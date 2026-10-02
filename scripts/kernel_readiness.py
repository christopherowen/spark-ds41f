"""Read-only kernel and memory-saver inventory, sent over SSH (no GPU work).

Installed artifacts establish preparation only, never successful boot or CUDA
qualification. GRUB identifiers contain node-local filesystem UUIDs; compare
their resolved kernel releases across nodes, not the identifiers themselves.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20,
                            env=dict(os.environ, LC_ALL="C"))
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def grub_inventory(text, environment):
    entries = {}
    submenu = ""
    entry = ""
    for line in text.splitlines():
        match = re.search(r"\$menuentry_id_option '([^']+)'", line)
        if match and line.startswith("submenu "):
            submenu = match[1]
        elif match and line.lstrip().startswith("menuentry "):
            entry = (submenu + ">" if line[:1].isspace() and submenu else "") + match[1]
        match = re.search(r"^\s*linux\s+\S*/vmlinuz-([^\s]+)", line)
        if match and entry:
            entries[entry] = match[1]
    defaults = re.findall(r'^\s*set default="([^"$]+)"', text, re.M)
    default = defaults[-1] if defaults else ""
    env = dict(line.split("=", 1) for line in environment.splitlines() if "=" in line)
    if 'set default="${saved_entry}"' in text and not default:
        default = env.get("saved_entry", "")
    next_entry = env.get("next_entry", "")
    # Numeric and title defaults are deliberately not assumed to be stable.
    return {"entries": entries, "default": default,
            "default_kernel": entries.get(default, ""), "next_entry": next_entry,
            "next_kernel": entries.get(next_entry, ""),
            "initrdfail": env.get("initrdfail", ""),
            "prev_entry": env.get("prev_entry", "")}


def key_enrolled(returncode, output):
    # Ubuntu's mokutil returns 1 for an already-enrolled certificate. Treat
    # its explicit result, not shell truthiness, as the enrollment evidence.
    return returncode in (0, 1) and output.strip().endswith(" is already enrolled")


def collect(policy):
    target = policy["candidate"]
    data = {"kernel": os.uname().release, "page_size": os.sysconf("SC_PAGE_SIZE"),
            "errors": [], "packages": {}, "modules": {}, "files": {}}
    def read(label, *args):
        try:
            return command(*args)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            data["errors"].append(f"{label}: {error}")
            return ""
    data["driver"] = read("loaded driver", "cat", "/sys/module/nvidia/version")
    # A locally built UVM module can report the same release as the packaged
    # driver. Compare the loaded binaries' source identities across nodes too.
    data["loaded_nvidia_sources"] = {
        module: read(module + " loaded source", "cat", f"/sys/module/{module}/srcversion")
        for module in ("nvidia", "nvidia_uvm")
    }
    if policy.get("rm_build_note"):
        data["rm_build_note"] = Path("/sys/module/nvidia/notes/.note.gnu.build-id").read_bytes().hex()
    packing = Path("/sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables")
    data["uvm_leaf_packing"] = packing.read_text().strip() if packing.exists() else "stock"
    for name in policy["packages"]:
        value = read(name, "dpkg-query", "-W", "-f=${Status}|${Version}", name)
        data["packages"][name] = value.removeprefix("install ok installed|") if value.startswith("install ok installed|") else ""
    for release in (policy["fallback"], target):
        for kind in ("vmlinuz", "initrd.img", "config"):
            path = f"/boot/{kind}-{release}"
            data["files"][path] = Path(path).is_file() and Path(path).stat().st_size > 0
    data["headers"] = Path(f"/lib/modules/{target}/build/Makefile").is_file()
    data["cpu_tools"] = Path(f"/usr/lib/linux-tools/{target}/cpupower").is_file()
    data["governors"] = sorted({p.read_text().strip() for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")})
    config = Path(f"/boot/config-{target}")
    data["config_64k"] = config.exists() and "CONFIG_ARM64_64K_PAGES=y" in config.read_text()
    for module in ("nvidia", "nvidia_uvm", "nvidia_drm", "mlx5_core", "mlx5_ib", "dgx_ec_fan_control"):
        data["modules"][module] = {
            "vermagic": read(module, "modinfo", "-k", target, "-F", "vermagic", module),
            "version": read(module, "modinfo", "-k", target, "-F", "version", module),
        }
    data["fan_dkms"] = read("fan DKMS", "dkms", "status", "-m", "dgx-spark-fan-control", "-k", target)
    if policy.get("memory_saver"):
        saver = policy["memory_saver"]
        data["memory_saver_dkms"] = read("memory-saver DKMS", "dkms", "status",
                                        "-m", saver["package"], "-k", target)
        data["memory_saver_disk"] = {
            field: read("memory-saver " + field, "modinfo", "-k", target,
                        "-F", field, "nvidia_uvm")
            for field in ("filename", "srcversion", "signer")
        }
    data["secure_boot"] = read("Secure Boot", "mokutil", "--sb-state")
    data["fan_signer"] = read("fan signature", "modinfo", "-k", target, "-F", "signer", "dgx_ec_fan_control")
    if data["secure_boot"] == "SecureBoot enabled":
        result = subprocess.run(["mokutil", "--test-key", policy["fan_signing_certificate"]],
                                capture_output=True, text=True, timeout=20,
                                env=dict(os.environ, LC_ALL="C"))
        data["fan_key_enrollment"] = (result.stdout + result.stderr).strip()
        data["fan_key_enrolled"] = key_enrolled(result.returncode, data["fan_key_enrollment"])
    grub = read("GRUB menu", "cat", "/boot/grub/grub.cfg")
    environment = read("GRUB environment", "grub-editenv", "list")
    data["grub"] = grub_inventory(grub, environment)
    data["thp"] = read("THP", "cat", "/sys/kernel/mm/transparent_hugepage/enabled")
    data["min_free_kbytes"] = read("memory reserve", "cat", "/proc/sys/vm/min_free_kbytes")
    swaps = read("active swap", "cat", "/proc/swaps")
    data["active_swap"] = [line.split()[0] for line in swaps.splitlines()[1:] if line.split()]
    if data["page_size"] == 65536:
        data["memory_service"] = read("kernel memory service", "systemctl", "is-active", "spark3-kernel-memory.service")
    return data


def findings(name, data, policy):
    """Warnings plus separate corrective commands; never execute corrections."""
    issues = []
    fixes = []
    target = policy["candidate"]
    running = data.get("kernel")
    expected_page = 65536 if running == target else 4096 if running == policy["fallback"] else None
    if expected_page is None or data.get("page_size") != expected_page:
        issues.append(f"{name}: running kernel/page size is outside the prepared pair: {running!r}/{data.get('page_size')!r}")
    if data.get("driver") != policy["driver"]:
        issues.append(f"{name}: loaded driver differs from {policy['driver']}")
    if policy.get("rm_build_note") and data.get("rm_build_note") != policy["rm_build_note"]:
        issues.append(f"{name}: loaded NVIDIA RM build differs from the required qualification build")
    if data.get("errors"):
        issues.append(f"{name}: incomplete kernel inventory: " + "; ".join(data["errors"]))
    missing = [p for p, version in policy["packages"].items() if data.get("packages", {}).get(p) != version]
    if missing:
        issues.append(f"{name}: 64 KiB package versions differ: {', '.join(missing)}")
        fixes.append(f"{name}: sudo apt-get install " + " ".join(f"{p}={policy['packages'][p]}" for p in missing))
    for path in [f"/boot/{kind}-{release}" for release in (policy["fallback"], target)
                 for kind in ("vmlinuz", "initrd.img", "config")]:
        if not data.get("files", {}).get(path):
            issues.append(f"{name}: missing or empty {path}")
    if not data.get("headers") or not data.get("config_64k"):
        issues.append(f"{name}: candidate headers or CONFIG_ARM64_64K_PAGES=y missing")
    if not data.get("cpu_tools"):
        issues.append(f"{name}: candidate cpupower tool is missing; nv-cpu-governor.service cannot run on that kernel")
    if data.get("governors") != ["performance"]:
        issues.append(f"{name}: CPU governors differ from performance: {data.get('governors')!r}")
        fixes.append(f"{name}: after installing matching linux-tools, sudo systemctl restart nv-cpu-governor.service")
    for module in ("nvidia", "nvidia_uvm", "nvidia_drm", "mlx5_core", "mlx5_ib", "dgx_ec_fan_control"):
        info = data.get("modules", {}).get(module, {})
        if info.get("vermagic", "").split()[:1] != [target]:
            issues.append(f"{name}: {module} is not prepared for {target}")
        if module.startswith("nvidia") and info.get("version") != policy["driver"]:
            issues.append(f"{name}: candidate {module} driver version differs from {policy['driver']}")
    if "installed" not in data.get("fan_dkms", ""):
        issues.append(f"{name}: fan-control DKMS is not installed for {target}")
        fixes.append(f"{name}: sudo dkms autoinstall -k {target}")
    if data.get("secure_boot") == "SecureBoot enabled" and (not data.get("fan_signer") or not data.get("fan_key_enrolled")):
        issues.append(f"{name}: Secure Boot is enabled but the candidate fan signature/enrolled key is unverified")
    grub = data.get("grub", {})
    if target not in grub.get("entries", {}).values():
        issues.append(f"{name}: candidate is missing from GRUB")
        fixes.append(f"{name}: sudo update-grub")
    default = policy.get("default", policy["fallback"])
    if grub.get("default_kernel") != default:
        issues.append(f"{name}: normal boot is not explicitly pinned to {default} (GRUB default {grub.get('default')!r})")
        fixes.append(f"{name}: restore the configured entry in /etc/default/grub.d/zz-spark-kernel-trial.cfg, then sudo update-grub")
    if grub.get("next_entry") or grub.get("initrdfail") or grub.get("prev_entry"):
        issues.append(f"{name}: a GRUB next-boot/recovery override is armed; kernel is not merely staged")
        fixes.append(f"{name}: inspect sudo grub-editenv list; cancel an unintended one-shot with sudo grub-editenv unset next_entry (review recovery state separately)")
    if data.get("page_size") == 65536:
        if "[never]" not in data.get("thp", "") or data.get("min_free_kbytes") != "45166":
            issues.append(f"{name}: 64 KiB THP/reserve policy differs from never/45166 KiB")
        if data.get("active_swap") != ["/swap-64k.img"] or data.get("memory_service") != "active":
            issues.append(f"{name}: 64 KiB swap or boot memory service is not active")
        if any("64 KiB THP" in x or "64 KiB swap" in x for x in issues):
            fixes.append(f"{name}: inspect journalctl -u spark3-kernel-memory.service; after correcting its inputs, sudo systemctl restart spark3-kernel-memory.service")
    saver_issues = memory_saver_problems(name, data, policy)
    issues.extend(saver_issues)
    if saver_issues:
        saver = policy["memory_saver"]
        fixes.append(f"{name}: register the pinned source using docs/memory-profiles.md#install-memory-saver; "
                     f"then sudo dkms install -m {saver['package']} -v {saver['version']} -k {target}. "
                     "For a loaded-module mismatch, stop GPU clients and reboot into the configured kernel.")
    return issues, fixes


def memory_saver_problems(name, data, policy):
    """Preparation warnings apply to both profiles; loaded checks apply at 64 KiB."""
    saver = policy.get("memory_saver")
    if not saver:
        return []
    issues = []
    expected = f"{saver['package']}/{saver['version']}, {policy['candidate']}, aarch64: installed"
    if not any(line == expected or line.startswith(expected + " (")
               for line in data.get("memory_saver_dkms", "").splitlines()):
        issues.append(f"{name}: memory-saver DKMS {saver['version']} is missing or not installed for {policy['candidate']}")
    disk = data.get("memory_saver_disk", {})
    if (disk.get("srcversion") != saver["srcversion"] or
            "/updates/dkms/" not in disk.get("filename", "")):
        issues.append(f"{name}: candidate UVM does not select the expected memory-saver DKMS module")
    if data.get("secure_boot") == "SecureBoot enabled" and not disk.get("signer"):
        issues.append(f"{name}: memory-saver module has no readable Secure Boot signer")
    if data.get("page_size") == 65536:
        if data.get("uvm_leaf_packing") != "Y":
            issues.append(f"{name}: memory-saver packing is not enabled in the loaded UVM module")
        if data.get("loaded_nvidia_sources", {}).get("nvidia_uvm") != saver["srcversion"]:
            issues.append(f"{name}: loaded UVM is not the expected memory-saver build; disk installation alone is insufficient")
    return issues


def profile_problems(name, data, profile, policy):
    """Hard launch requirements for a memory budget tied to a particular kernel."""
    if not profile:
        return []
    issues = []
    if data.get("kernel") != profile["release"] or data.get("page_size") != profile["page_size"]:
        issues.append(f"{name}: serving profile requires {profile['release']} / {profile['page_size']}-byte pages; "
                      f"running {data.get('kernel')!r} / {data.get('page_size')!r}")
    if policy.get("rm_build_note") and data.get("rm_build_note") != policy["rm_build_note"]:
        issues.append(f"{name}: serving profile requires the qualified NVIDIA RM build")
    if profile.get("memory_saver_required"):
        if not policy.get("memory_saver"):
            issues.append(f"{name}: serving profile requires a pinned memory-saver policy")
        else:
            issues.extend(memory_saver_problems(name, data, policy))
        if data.get("driver") != policy["driver"]:
            issues.append(f"{name}: serving profile requires NVIDIA {policy['driver']}")
    return issues


def alignment(inventories):
    issues = []
    for field in ("kernel", "page_size", "driver", "loaded_nvidia_sources", "uvm_leaf_packing", "thp", "min_free_kbytes", "secure_boot", "governors"):
        values = {name: data.get(field) for name, data in inventories.items()}
        if any(value in (None, "") for value in values.values()) or len({str(v) for v in values.values()}) > 1:
            issues.append("nodes differ or are unreadable for " + field + ": " + ", ".join(f"{n}={v!r}" for n, v in values.items()))
    return issues


if __name__ == "__main__":
    try:
        print(json.dumps(collect(json.loads(sys.argv[1]))))
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"errors": [str(error)]}))
