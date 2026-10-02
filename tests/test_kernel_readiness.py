"""Readiness means staged artifacts, never automatic boot or qualification."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kernel_readiness", ROOT / "scripts/kernel_readiness.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
POLICY = json.loads((ROOT / "config/kernel-trial.json").read_text())
OLD = POLICY["fallback"]
NEW = POLICY["candidate"]


def menu(uuid="node-a", default=None):
    old_id = f"gnulinux-advanced-{uuid}>gnulinux-{OLD}-advanced-{uuid}"
    return f'''if [ "${{next_entry}}" ]; then
 set default="${{next_entry}}"
else
 set default="{old_id if default is None else default}"
fi
submenu 'Advanced options' $menuentry_id_option 'gnulinux-advanced-{uuid}' {{
 menuentry 'old' $menuentry_id_option 'gnulinux-{OLD}-advanced-{uuid}' {{
  linux /boot/vmlinuz-{OLD} root=UUID={uuid}
 }}
 menuentry 'new' $menuentry_id_option 'gnulinux-{NEW}-advanced-{uuid}' {{
  linux /boot/vmlinuz-{NEW} root=UUID={uuid}
 }}
}}
'''


def ready(uuid="node-a"):
    return {"kernel": OLD, "page_size": 4096, "driver": POLICY["driver"],
            "thp": "always [madvise] never", "min_free_kbytes": "45166",
            "secure_boot": "SecureBoot enabled", "fan_signer": "Fleet key", "fan_key_enrolled": True,
            "errors": [], "packages": POLICY["packages"].copy(),
            "files": {f"/boot/{kind}-{release}": True for kind in ("config", "vmlinuz", "initrd.img") for release in (OLD, NEW)},
            "headers": True, "config_64k": True,
            "cpu_tools": True, "governors": ["performance"],
            "modules": {m: {"vermagic": NEW + " SMP", "version": POLICY["driver"]} for m in
                        ("nvidia", "nvidia_uvm", "nvidia_drm", "mlx5_core", "mlx5_ib", "dgx_ec_fan_control")},
            "fan_dkms": f"dgx-spark-fan-control/0.1.3, {NEW}, aarch64: installed",
            "grub": kernel.grub_inventory(menu(uuid), "")}


class ReadinessTest(unittest.TestCase):
    def test_staged_not_running_is_success(self):
        self.assertEqual(kernel.findings("dgx1", ready(), POLICY), ([], []))

    def test_missing_package_is_not_installed(self):
        data = ready()
        package = next(iter(data["packages"]))
        data["packages"][package] = ""
        issues, fixes = kernel.findings("dgx1", data, POLICY)
        self.assertTrue(any("package versions differ" in x for x in issues))
        self.assertIn(f"{package}={POLICY['packages'][package]}", fixes[0])

    def test_wrong_driver_and_vermagic(self):
        data = ready()
        data["modules"]["nvidia"] = {"version": "wrong", "vermagic": OLD}
        issues, _ = kernel.findings("dgx1", data, POLICY)
        self.assertEqual(sum("nvidia" in x for x in issues), 2)

    def test_missing_fan_and_initrd(self):
        data = ready()
        data["fan_dkms"] = "built"
        data["files"][f"/boot/initrd.img-{NEW}"] = False
        issues, fixes = kernel.findings("dgx1", data, POLICY)
        self.assertTrue(any("initrd" in x for x in issues))
        self.assertTrue(any("dkms autoinstall" in x for x in fixes))

    def test_numeric_default_is_not_a_safe_pin(self):
        data = ready()
        data["grub"] = kernel.grub_inventory(menu(default="0"), "")
        self.assertTrue(any("normal boot" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_one_shot_and_recovery_overrides(self):
        for env in ("next_entry=trial", "initrdfail=1\nprev_entry=trial"):
            with self.subTest(env=env):
                data = ready()
                data["grub"] = kernel.grub_inventory(menu(), env)
                self.assertTrue(any("override is armed" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_node_specific_grub_ids_do_not_cause_drift(self):
        self.assertEqual(kernel.alignment({"dgx1": ready("a"), "dgx2": ready("b")}), [])

    def test_mixed_pages_and_kernels_are_reported(self):
        second = ready()
        second.update(kernel=NEW, page_size=65536)
        self.assertEqual(len(kernel.alignment({"dgx1": ready(), "dgx2": second})), 2)

    def test_failed_probe_never_passes_alignment(self):
        issues = kernel.alignment({"dgx1": ready(), "dgx2": {"errors": ["SSH failed"]}})
        self.assertTrue(any("page_size" in x for x in issues))

    def test_absent_module_is_reported(self):
        data = ready()
        del data["modules"]["mlx5_ib"]
        self.assertTrue(any("mlx5_ib" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_identically_wrong_live_driver_still_warns(self):
        data = ready()
        data["driver"] = "wrong"
        self.assertTrue(any("loaded driver" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_kernel_page_size_disagreement(self):
        data = ready()
        data["page_size"] = 65536
        self.assertTrue(any("kernel/page size" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_unsigned_fan_is_not_ready_under_secure_boot(self):
        data = ready()
        data["fan_signer"] = ""
        self.assertTrue(any("candidate fan signature" in x for x in kernel.findings("dgx1", data, POLICY)[0]))

    def test_mokutil_enrollment_exit_status_is_not_shell_success(self):
        self.assertTrue(kernel.key_enrolled(1, "/root/key.der is already enrolled\n"))
        self.assertFalse(kernel.key_enrolled(0, "/root/key.der is not enrolled"))
        self.assertFalse(kernel.key_enrolled(1, "Failed to read key"))

    def test_64k_boot_requires_its_swap_and_memory_policy(self):
        data = ready()
        data.update(kernel=NEW, page_size=65536)
        issues, _ = kernel.findings("dgx1", data, POLICY)
        self.assertTrue(any("64 KiB swap" in x for x in issues))
        data.update(thp="always madvise [never]", active_swap=["/swap-64k.img"], memory_service="active")
        self.assertEqual(kernel.findings("dgx1", data, POLICY), ([], []))

    def test_promoted_default_is_distinct_from_retained_fallback(self):
        data = ready()
        new_id = next(k for k,v in data["grub"]["entries"].items() if v == NEW)
        data["grub"] = kernel.grub_inventory(menu(default=new_id), "")
        self.assertEqual(kernel.findings("dgx1", data, dict(POLICY, default=NEW)), ([], []))


if __name__ == "__main__":
    unittest.main()
