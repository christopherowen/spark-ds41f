"""Host-independent tests for doctor's node checks."""

from __future__ import annotations

import contextlib
import io
import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class DesktopTest(unittest.TestCase):
    def test_headless_node_passes(self) -> None:
        self.assertEqual(spark.desktop_problems("dgx1", "multi-user.target\ninactive\n"), [])

    def test_graphical_target_and_running_display_manager_are_reported(self) -> None:
        problems = spark.desktop_problems("dgx2", "graphical.target\nactive\n")
        self.assertEqual(len(problems), 2)
        self.assertIn("boots to graphical.target", problems[0])
        self.assertIn("display manager is active", problems[1])

    def test_display_manager_started_by_hand_is_reported(self) -> None:
        problems = spark.desktop_problems("dgx3", "multi-user.target\nactive\n")
        self.assertEqual(
            problems,
            ["dgx3: display manager is active; run sudo systemctl disable --now display-manager.service"],
        )

    def test_missing_output_is_reported(self) -> None:
        problems = spark.desktop_problems("dgx1", "")
        self.assertEqual(len(problems), 1)
        self.assertIn("unknown target", problems[0])


DGX1_BOOT = """BootCurrent: 0001
Timeout: 1 seconds
BootOrder: 0001,0003,0004,0005,0006
Boot0001* ubuntu\tHD(1,GPT,f9e7b4d5-76a3-46f4-8fbb-cea82cd05534,0x800,0x95000)/File(\\EFI\\ubuntu\\shimaa64.efi) File(.)
Boot0003* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller\tPcieRoot(0x7)/Pci(0x0,0x0)/Pci(0x0,0x0)/MAC(4cbb47e97f2a,0)
Boot0004* UEFI:CD/DVD Drive\tBBS(129,,0x0)
"""

# dgx3 after its 2026-06-12 firmware update, before the boot order was fixed.
DGX3_PXE_FIRST_BOOT = """BootCurrent: 0004
Timeout: 1 seconds
BootOrder: 0002,0004
Boot0002* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller\tPcieRoot(0x7)/Pci(0x0,0x0)/Pci(0x0,0x0)/MAC(4cbb47e97f2a,0)
Boot0004* ubuntu\tHD(1,GPT,88df73c8-497a-4b1d-be95-4e318905d1fb,0x800,0x95000)/File(\\EFI\\ubuntu\\shimaa64.efi) File(.)
"""


class BootOrderTest(unittest.TestCase):
    def test_installed_os_first_passes(self) -> None:
        self.assertEqual(spark.boot_order_problems("dgx1", DGX1_BOOT), [])

    def test_network_boot_first_is_reported(self) -> None:
        problems = spark.boot_order_problems("dgx3", DGX3_PXE_FIRST_BOOT)
        self.assertEqual(len(problems), 1)
        self.assertIn("UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller", problems[0])
        self.assertIn("BootOrder 0002,0004", problems[0])
        self.assertIn("run sudo efibootmgr --bootorder 0004,0002", problems[0])

    def test_fixed_order_passes(self) -> None:
        fixed = DGX3_PXE_FIRST_BOOT.replace("BootOrder: 0002,0004", "BootOrder: 0004,0002")
        self.assertEqual(spark.boot_order_problems("dgx3", fixed), [])

    def test_unreadable_boot_order_is_reported(self) -> None:
        self.assertEqual(
            spark.boot_order_problems("dgx2", ""),
            ["dgx2: cannot read the UEFI boot order"],
        )


FAN_WORKING = (
    "modeset=Y\n"
    "fbdev=Y\n"
    "drm_card=0\n"
    "drm_masters=\n"
    "console=tty1:0\n"
    "cmdline_splash=0\n"
    "grub_splash=0\n"
    "idle_services=\n"
    "kernel=7.0.0-1019-nvidia\n"
    "fan_dkms=dgx-spark-fan-control/0.1.3, 7.0.0-1019-nvidia, aarch64: installed\n"
    "fan_module=1\n"
    "fan_cooling_device=1\n"
    "fan_service=installed\n"
)


def facts(**changes: str) -> dict[str, str]:
    values = spark.host_facts(FAN_WORKING)
    values.update(changes)
    return values


class ModesetTest(unittest.TestCase):
    def test_kernel_mode_setting_passes(self) -> None:
        self.assertEqual(spark.modeset_problems("dgx1", facts()), [])

    def test_disabled_mode_setting_is_reported(self) -> None:
        problems = spark.modeset_problems("dgx1", facts(modeset="N"))
        self.assertEqual(len(problems), 1)
        self.assertIn("modeset is N, expected Y", problems[0])

    def test_unreadable_mode_setting_is_reported(self) -> None:
        problems = spark.modeset_problems("dgx1", facts(modeset=""))
        self.assertIn("modeset is unreadable", problems[0])


class ConsoleTest(unittest.TestCase):
    def test_working_console_passes(self) -> None:
        self.assertEqual(spark.fbdev_problems("dgx3", facts()), [])
        self.assertEqual(spark.drm_master_problems("dgx3", facts()), [])
        self.assertEqual(spark.console_mode_problems("dgx3", facts()), [])

    def test_fbdev_off_is_reported(self) -> None:
        problems = spark.fbdev_problems("dgx3", facts(fbdev="N"))
        self.assertEqual(len(problems), 1)
        self.assertIn("fbdev is N, expected Y", problems[0])

    def test_unreadable_fbdev_is_reported(self) -> None:
        problems = spark.fbdev_problems("dgx3", facts(fbdev=""))
        self.assertIn("fbdev is unreadable", problems[0])

    def test_drm_master_holder_is_reported(self) -> None:
        problems = spark.drm_master_problems(
            "dgx3", facts(drm_masters="VLLM::Worker_TP/2855221")
        )
        self.assertEqual(
            problems,
            [
                "dgx3: VLLM::Worker_TP/2855221 holds DRM master, so the text console cannot "
                "draw; stop or restart that process (the serving worker releases the device "
                "right after startup)"
            ],
        )

    def test_unreadable_drm_clients_are_reported(self) -> None:
        problems = spark.drm_master_problems("dgx3", facts(drm_masters="unreadable"))
        self.assertIn("cannot read the DRM clients", problems[0])
        missing = spark.host_facts("modeset=Y\n")
        self.assertIn("cannot read the DRM clients", spark.drm_master_problems("dgx3", missing)[0])

    def test_graphics_mode_console_is_reported(self) -> None:
        problems = spark.console_mode_problems("dgx3", facts(console="tty1:1"))
        self.assertEqual(len(problems), 1)
        self.assertIn("the active console (tty1) is in graphics mode", problems[0])

    def test_unreadable_console_mode_is_reported(self) -> None:
        problems = spark.console_mode_problems("dgx3", facts(console="tty1:"))
        self.assertEqual(problems, ["dgx3: cannot read the active console mode"])

    def test_splash_boot_is_reported_with_its_fix(self) -> None:
        problems = spark.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="1"))
        self.assertEqual(len(problems), 1)
        self.assertIn("scripts/host-recovery apply", problems[0])
        self.assertIn("zz-spark-console.cfg", problems[0])

    def test_configured_splash_removal_waits_for_a_reboot(self) -> None:
        problems = spark.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="0"))
        self.assertEqual(len(problems), 1)
        self.assertIn("reboot with the service stopped to apply", problems[0])

    def test_no_splash_passes(self) -> None:
        self.assertEqual(spark.splash_problems("dgx1", facts()), [])

    def test_every_console_problem_names_a_fix(self) -> None:
        for problems in (
            spark.fbdev_problems("dgx3", facts(fbdev="N")),
            spark.console_mode_problems("dgx3", facts(console="tty1:1")),
            spark.modeset_problems("dgx3", facts(modeset="N")),
        ):
            self.assertRegex(problems[0], r"; (run|remove) ")

    def test_modeset_message_names_the_carveout(self) -> None:
        problems = spark.modeset_problems("dgx1", facts(modeset="N"))
        self.assertIn("display carve-out cannot be allocated", problems[0])


class NvmeCoalescingTest(unittest.TestCase):
    def test_coalescing_off_and_service_masked_passes(self) -> None:
        self.assertEqual(spark.nvme_coalescing_warnings(
            "dgx1", facts(nvme_coalescing="nvme0:00000000", nvme_coalescing_service="masked")), [])

    def test_missing_facts_pass(self) -> None:
        self.assertEqual(spark.nvme_coalescing_warnings("dgx1", facts()), [])

    def test_coalescing_on_is_reported_with_its_fix(self) -> None:
        warnings = spark.nvme_coalescing_warnings(
            "dgx2", facts(nvme_coalescing="nvme0:0x00000107", nvme_coalescing_service="enabled"))
        self.assertEqual(len(warnings), 2)
        self.assertIn("nvme0=0x00000107", warnings[0])
        self.assertIn("systemctl mask --now nvidia-nvme-interrupt-coalescing.service", warnings[0])
        self.assertIn("next boot", warnings[1])
        self.assertTrue(all(isinstance(w, spark.Warn) for w in warnings))

    def test_enabled_service_alone_is_reported(self) -> None:
        warnings = spark.nvme_coalescing_warnings(
            "dgx3", facts(nvme_coalescing="nvme0:00000000", nvme_coalescing_service="enabled"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("next boot", warnings[0])

    def test_unreadable_controller_is_reported(self) -> None:
        warnings = spark.nvme_coalescing_warnings(
            "dgx4", facts(nvme_coalescing="nvme0:", nvme_coalescing_service="masked"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("cannot read", warnings[0])


class IdleServicesTest(unittest.TestCase):
    def test_idle_services_are_reported_with_their_fix(self) -> None:
        problems = spark.idle_service_warnings(
            "dgx2", facts(idle_services="bluetooth.service,snapd.socket")
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("bluetooth.service, snapd.socket enabled or running", problems[0])
        self.assertIn(
            "sudo systemctl disable --now bluetooth.service snapd.socket", problems[0]
        )
        self.assertIn("scripts/host-recovery apply", problems[0])

    def test_no_idle_services_pass(self) -> None:
        self.assertEqual(spark.idle_service_warnings("dgx2", facts()), [])
        self.assertEqual(
            spark.idle_service_warnings("dgx2", spark.host_facts("modeset=Y\n")), []
        )

    def test_warnings_do_not_fail_doctor(self) -> None:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark.report_doctor([], ["dgx2: snapd.service running"], live=True)
        self.assertEqual(status, 0)
        self.assertIn("WARN: dgx2: snapd.service running", output.getvalue())
        self.assertIn("configuration OK; live cluster matches with 1 warning", output.getvalue())

    def test_problems_fail_doctor_as_errors(self) -> None:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark.report_doctor(["dgx1: container is not running"], [], live=True)
        self.assertEqual(status, 1)
        self.assertIn("ERROR: dgx1: container is not running", output.getvalue())

    def test_query_lists_every_idle_unit(self) -> None:
        for unit in spark.IDLE_SERVICES:
            self.assertIn(unit, spark.HOST_FACTS_QUERY)
        # fwupd.service itself runs legitimately after a manual fwupdmgr call.
        self.assertIn("fwupd-refresh.timer", spark.IDLE_SERVICES)
        self.assertNotIn("fwupd.service", spark.IDLE_SERVICES)


class SeverityTest(unittest.TestCase):
    """Required capabilities are errors; latent or minor findings are warnings."""

    def test_missing_nvidia_card_is_an_error(self) -> None:
        self.assertEqual(spark.drm_card_problems("dgx3", facts()), [])
        self.assertEqual(spark.drm_card_problems("dgx3", facts(drm_card="1")), [])
        problems = spark.drm_card_problems("dgx3", facts(drm_card=""))
        self.assertEqual(len(problems), 1)
        self.assertNotIsInstance(problems[0], spark.Warn)
        self.assertIn("display carve-out cannot be allocated", problems[0])

    def test_required_display_pieces_are_errors(self) -> None:
        for problems in (
            spark.modeset_problems("dgx1", facts(modeset="N")),
            spark.fbdev_problems("dgx1", facts(fbdev="N")),
            spark.drm_master_problems("dgx1", facts(drm_masters="Xorg/123")),
            spark.console_mode_problems("dgx1", facts(console="tty1:1")),
        ):
            self.assertEqual(len(problems), 1)
            self.assertNotIsInstance(problems[0], spark.Warn)

    def test_latent_and_unreadable_findings_warn(self) -> None:
        for problems in (
            spark.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="1")),
            spark.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="0")),
            spark.drm_master_problems("dgx1", facts(drm_masters="unreadable")),
            spark.console_mode_problems("dgx1", facts(console="tty1:")),
            spark.idle_service_warnings("dgx1", facts(idle_services="cups.service")),
            spark.desktop_problems("dgx1", "graphical.target\ninactive\n"),
            spark.boot_order_problems("dgx1", ""),
        ):
            self.assertEqual(len(problems), 1)
            self.assertIsInstance(problems[0], spark.Warn)

    def test_running_display_manager_is_an_error(self) -> None:
        problems = spark.desktop_problems("dgx1", "multi-user.target\nactive\n")
        self.assertEqual(len(problems), 1)
        self.assertNotIsInstance(problems[0], spark.Warn)

    def test_split_findings_keeps_order(self) -> None:
        errors, warnings = spark.split_findings(
            ["a", spark.Warn("b"), "c", spark.Warn("d")]
        )
        self.assertEqual(errors, ["a", "c"])
        self.assertEqual(warnings, ["b", "d"])


class FanControlTest(unittest.TestCase):
    def test_working_fan_floor_passes(self) -> None:
        self.assertEqual(spark.fan_control_problems("dgx1", facts()), [])

    def test_missing_dkms_module_is_the_only_report(self) -> None:
        problems = spark.fan_control_problems(
            "dgx3", facts(fan_dkms="", fan_module="0", fan_cooling_device="0")
        )
        self.assertEqual(
            problems, ["dgx3: DKMS dgx-spark-fan-control is not installed for 7.0.0-1019-nvidia; run "
                "sudo dkms autoinstall -k 7.0.0-1019-nvidia"]
        )

    def test_unloaded_module_is_reported(self) -> None:
        problems = spark.fan_control_problems("dgx3", facts(fan_module="0"))
        self.assertIn("dgx_ec_fan_control is not loaded", problems[0])

    def test_refused_cooling_device_is_reported(self) -> None:
        # dgx3 on firmware 5.36_0ACUM027: the driver loads, then the EC rejects
        # its capability read and it refuses to register the cooling device.
        problems = spark.fan_control_problems("dgx3", facts(fan_cooling_device="0"))
        self.assertEqual(len(problems), 1)
        self.assertIn("cooling device is missing", problems[0])

    def test_missing_service_is_an_error(self) -> None:
        # bench needs it; whether it runs while serving is not checked.
        problems = spark.fan_control_problems("dgx2", facts(fan_service="absent"))
        self.assertEqual(len(problems), 1)
        self.assertNotIsInstance(problems[0], spark.Warn)
        self.assertIn("dgx-fan-control.service is not installed", problems[0])
        self.assertIn("sudo systemctl daemon-reload", problems[0])

    def test_missing_driver_layer_is_reported_before_the_service(self) -> None:
        problems = spark.fan_control_problems("dgx2", facts(fan_module="0", fan_service="absent"))
        self.assertEqual(len(problems), 1)
        self.assertNotIsInstance(problems[0], spark.Warn)
        self.assertIn("dgx_ec_fan_control is not loaded", problems[0])


class FanCurveTest(unittest.TestCase):
    """bench runs the performance curve while it measures, then restores each node."""

    def run_curve(self, services: dict[str, str], start_fails: tuple[str, ...] = (),
                  stop_fails: tuple[str, ...] = ()):
        calls: list[tuple[str, ...]] = []

        def fake_ssh(nodes, node, *command):
            name = node["name"]
            calls.append((name, *command))
            state = services[name]
            if command[:3] == ("systemctl", "is-active", "--quiet"):
                return spark.subprocess.CompletedProcess(command, 0 if state == "active" else 3, "", "")
            if command[:2] == ("systemctl", "cat"):
                return spark.subprocess.CompletedProcess(command, 1 if state == "absent" else 0, "", "")
            if command[3:5] == ("start", spark.FAN_SERVICE) and name in start_fails:
                return spark.subprocess.CompletedProcess(command, 1, "", "start-limit-hit\n")
            if command[3:5] == ("stop", spark.FAN_SERVICE) and name in stop_fails:
                return spark.subprocess.CompletedProcess(command, 1, "", "")
            return spark.subprocess.CompletedProcess(command, 0, "", "")

        original = spark.run_ssh
        spark.run_ssh = fake_ssh
        try:
            nodes = {"ssh_user": "u", "nodes": [{"name": n} for n in services]}
            with contextlib.redirect_stdout(io.StringIO()):
                records = spark.start_fan_curve(nodes, nodes["nodes"])
                started_calls = list(calls)
                failures = spark.stop_fan_curve(nodes, nodes["nodes"], records)
        finally:
            spark.run_ssh = original
        return records, failures, started_calls, calls[len(started_calls):]

    def test_installed_service_runs_for_the_bench_and_stops_after(self) -> None:
        records, failures, started, stopped = self.run_curve({"dgx1": "inactive"})
        self.assertEqual(records, [{"node": "dgx1", "service": "inactive", "started": True, "stopped": True}])
        self.assertEqual(spark.fan_curve_problems(records), [])
        self.assertEqual(failures, [])
        self.assertIn(("dgx1", "sudo", "-n", "systemctl", "start", spark.FAN_SERVICE), started)
        self.assertEqual(stopped, [("dgx1", "sudo", "-n", "systemctl", "stop", spark.FAN_SERVICE)])

    def test_running_service_is_left_running(self) -> None:
        records, failures, started, stopped = self.run_curve({"dgx1": "active"})
        self.assertEqual(records, [{"node": "dgx1", "service": "active", "started": False}])
        self.assertEqual(spark.fan_curve_problems(records), [])
        self.assertFalse(any(call[1] == "sudo" for call in started))
        self.assertEqual((failures, stopped), ([], []))

    def test_node_without_the_service_stops_the_bench(self) -> None:
        records, failures, started, stopped = self.run_curve({"dgx1": "absent", "dgx2": "inactive"})
        self.assertEqual(records[0], {"node": "dgx1", "service": "absent", "started": False})
        self.assertEqual(
            spark.fan_curve_problems(records),
            ["dgx1: dgx-fan-control.service is not installed (see doctor --live)"],
        )
        # The node it did start is still returned to its own control.
        self.assertTrue(records[1]["started"])
        self.assertEqual(stopped, [("dgx2", "sudo", "-n", "systemctl", "stop", spark.FAN_SERVICE)])
        self.assertFalse(any(call[0] == "dgx1" and call[1] == "sudo" for call in started))
        self.assertEqual(failures, [])

    def test_failed_start_is_recorded_and_not_stopped(self) -> None:
        records, failures, _, stopped = self.run_curve({"dgx3": "inactive"}, start_fails=("dgx3",))
        self.assertFalse(records[0]["started"])
        self.assertEqual(records[0]["error"], "start-limit-hit")
        self.assertEqual(
            spark.fan_curve_problems(records),
            ["dgx3: dgx-fan-control.service did not start: start-limit-hit"],
        )
        self.assertEqual((failures, stopped), ([], []))

    def test_failed_stop_is_reported(self) -> None:
        _, failures, _, _ = self.run_curve({"dgx4": "inactive"}, stop_fails=("dgx4",))
        self.assertEqual(failures, [f"dgx4: cannot stop {spark.FAN_SERVICE}"])


HEALTHY_GIDS = """rocep1s0f0 0 IB/RoCEv1 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 2 IB/RoCEv1 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
rocep1s0f0 3 RoCEv2 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
"""

# dgx3 after its peers rebooted while a starting service held RDMA resources.
SHIFTED_GIDS = """rocep1s0f0 0 IB/RoCEv1 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 2 IB/RoCEv1 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
rocep1s0f0 4 RoCEv2 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
"""


class RoceGidTest(unittest.TestCase):
    def test_ipv4_roce_v2_at_the_configured_index_passes(self) -> None:
        self.assertEqual(spark.roce_gid_problems("dgx1", 3, ["rocep1s0f0"], HEALTHY_GIDS), [])

    def test_shifted_gid_names_the_slot_and_interface(self) -> None:
        problems = spark.roce_gid_problems("dgx3", 3, ["rocep1s0f0"], SHIFTED_GIDS)
        self.assertEqual(len(problems), 1)
        self.assertIn("rocep1s0f0 GID index 3 is empty", problems[0])
        self.assertIn("index 4; re-activate enp1s0f0np0", problems[0])

    def test_ipv6_roce_v2_at_the_index_is_reported(self) -> None:
        problems = spark.roce_gid_problems("dgx1", 1, ["rocep1s0f0"], HEALTHY_GIDS)
        self.assertIn("GID index 1 is RoCEv2 fe80", problems[0])
        self.assertIn("at index 3", problems[0])

    def test_missing_device_and_missing_ipv4_are_reported(self) -> None:
        output = "rocep1s0f1 missing\nrocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0\n"
        problems = spark.roce_gid_problems("dgx2", 3, ["rocep1s0f0", "rocep1s0f1"], output)
        self.assertEqual(
            problems,
            [
                "dgx2: rocep1s0f0 GID index 3 is empty; it has no IPv4 RoCE v2 GID",
                "dgx2: RDMA device rocep1s0f1 is missing",
            ],
        )


class ClockLatchTest(unittest.TestCase):
    def test_serving_node_at_full_clock_passes(self) -> None:
        self.assertEqual(spark.clock_latch_problems("dgx1", facts(gpu="2411, 11.47"), True), [])

    def test_latched_serving_node_is_reported(self) -> None:
        # dgx3 on 2026-09-26: 520-565 MHz at about 10 W, ignoring nvidia-smi -lgc.
        problems = spark.clock_latch_problems("dgx3", facts(gpu="559, 9.90"), True)
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU clock is 559 MHz at 9.90 W while serving", problems[0])

    def test_idle_node_without_the_service_is_not_judged(self) -> None:
        self.assertEqual(spark.clock_latch_problems("dgx2", facts(gpu="208, 4.1"), False), [])

    def test_unreadable_clock_is_reported(self) -> None:
        self.assertEqual(
            spark.clock_latch_problems("dgx2", facts(gpu=""), True),
            ["dgx2: cannot read the GPU clock"],
        )


# HostConfig.Binds of the TP4 production container, as dgx1-4 reported it on 2026-10-06.
PRODUCTION_BINDS = [
    "/home/swank/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
    "dba1be0a40aa45a94ad051997016db3960a90277:/models:ro",
    "/home/swank/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/blobs:/blobs:ro",
    "/home/swank/projects/spark-ds41f/cache:/cache:rw",
]
SPARKNET_BIND = "/home/swank/sparknet-port/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro"


class LiveMountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = spark.json.loads((ROOT / "config" / "cluster-tp4.json").read_text())
        self.wanted = spark.expected_binds(self.cluster)

    def test_docker_run_is_given_the_compared_binds(self) -> None:
        node = spark.json.loads((ROOT / "config" / "nodes.example.json").read_text())["nodes"][0]
        command = spark.rendered_docker_command(self.cluster, node)
        volumes = [command[i + 1] for i, flag in enumerate(command) if flag == "--volume"]
        self.assertEqual(volumes, self.wanted)

    def test_production_container_passes_in_any_order(self) -> None:
        self.assertEqual(spark.mount_problems("dgx1", self.wanted, PRODUCTION_BINDS), [])
        self.assertEqual(spark.mount_problems("dgx1", self.wanted, PRODUCTION_BINDS[::-1]), [])

    def test_lab_arm_with_an_extra_mount_fails_the_live_match(self) -> None:
        # 2026-10-05: a measure bracket arm (cluster-tp4.json plus the sparknet
        # port) stayed serving because doctor --live matched it to production.
        problems = spark.mount_problems("dgx1", self.wanted, PRODUCTION_BINDS + [SPARKNET_BIND])
        self.assertEqual(problems, [f"dgx1: extra mount {SPARKNET_BIND!r}"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark.report_doctor(problems, [], live=True)
        self.assertEqual(status, 1)
        self.assertIn(f"ERROR: dgx1: extra mount {SPARKNET_BIND!r}", output.getvalue())
        self.assertNotIn("live cluster matches", output.getvalue())

    def test_missing_mount_is_reported(self) -> None:
        self.assertEqual(
            spark.mount_problems("dgx3", self.wanted, PRODUCTION_BINDS[:2]),
            [f"dgx3: missing mount {PRODUCTION_BINDS[2]!r}"],
        )

    def test_container_without_binds_misses_every_mount(self) -> None:
        problems = spark.mount_problems("dgx2", self.wanted, None)
        self.assertEqual(problems, [f"dgx2: missing mount {bind!r}" for bind in self.wanted])

    def test_different_source_or_mode_is_reported_against_the_wanted_mount(self) -> None:
        writable_models = PRODUCTION_BINDS[0].replace(":/models:ro", ":/models:rw")
        other_cache = "/home/swank/projects/spark-ds41f-lab/cache:/cache:rw"
        problems = spark.mount_problems(
            "dgx4", self.wanted, [writable_models, PRODUCTION_BINDS[1], other_cache]
        )
        self.assertEqual(
            problems,
            [
                f"dgx4: mount {writable_models!r}, expected {PRODUCTION_BINDS[0]!r}",
                f"dgx4: mount {other_cache!r}, expected {PRODUCTION_BINDS[2]!r}",
            ],
        )


# Config.Env of the r6 image (sha256:b5225c98...) as dgx1 reported it on 2026-10-06, abridged.
IMAGE_ENV = [
    "PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CUDA_VERSION=13.0.2",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x",
    "SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce",
    "VLLM_USAGE_SOURCE=production-docker-image",
]


def container_env(wanted: dict[str, str]) -> list[str]:
    """Config.Env as docker run builds it: the --env values, then the image's other entries."""
    return [f"{key}={value}" for key, value in sorted(wanted.items())] + [
        item for item in IMAGE_ENV if item.partition("=")[0] not in wanted
    ]


class LiveEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = spark.json.loads((ROOT / "config" / "cluster-tp4.json").read_text())
        site = spark.json.loads((ROOT / "config" / "examples" / "nodes-ring4.json").read_text())
        self.node = site["nodes"][0]
        self.wanted = spark.expected_environment(self.cluster, self.node)
        self.production = container_env(self.wanted)

    def test_docker_run_is_given_the_compared_environment(self) -> None:
        command = spark.rendered_docker_command(self.cluster, self.node)
        values = [command[i + 1] for i, flag in enumerate(command) if flag == "--env"]
        self.assertEqual(dict(value.split("=", 1) for value in values), self.wanted)

    def test_production_container_passes_in_any_order(self) -> None:
        self.assertEqual(spark.environment_problems("dgx1", self.wanted, self.production, IMAGE_ENV), [])
        self.assertEqual(
            spark.environment_problems("dgx1", self.wanted, self.production[::-1], IMAGE_ENV), []
        )

    def test_lab_arm_with_an_extra_variable_fails_the_live_match(self) -> None:
        # 2026-10-06: a prefetch arm (cluster-tp4.json plus one prefetch variable)
        # stayed serving because doctor --live matched it to production.
        self.assertNotIn("VLLM_DS41_L2_PREFETCH_FFN_MB", self.wanted)
        arm = spark.json.loads(spark.json.dumps(self.cluster))
        arm["environment"]["VLLM_DS41_L2_PREFETCH_FFN_MB"] = "8"
        live = container_env(spark.expected_environment(arm, self.node))
        problems = spark.environment_problems("dgx1", self.wanted, live, IMAGE_ENV)
        self.assertEqual(problems, ["dgx1: extra variable VLLM_DS41_L2_PREFETCH_FFN_MB='8'"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark.report_doctor(problems, [], live=True)
        self.assertEqual(status, 1)
        self.assertIn("ERROR: dgx1: extra variable VLLM_DS41_L2_PREFETCH_FFN_MB='8'", output.getvalue())
        self.assertNotIn("live cluster matches", output.getvalue())

    def test_changed_value_is_reported(self) -> None:
        live = container_env({**self.wanted, "VLLM_DS41_KERNEL_BACKEND": "b12x"})
        self.assertEqual(
            spark.environment_problems("dgx2", self.wanted, live, IMAGE_ENV),
            ["dgx2: VLLM_DS41_KERNEL_BACKEND='b12x', expected 'tilelang'"],
        )

    def test_missing_variable_is_reported(self) -> None:
        live = container_env({k: v for k, v in self.wanted.items() if k != "NCCL_PROTO"})
        self.assertEqual(
            spark.environment_problems("dgx3", self.wanted, live, IMAGE_ENV),
            ["dgx3: NCCL_PROTO=None, expected '^LL128'"],
        )

    def test_image_variables_are_ignored(self) -> None:
        # PYTHONPATH is set by both; the rest only by the image.
        image_only = [item for item in IMAGE_ENV if item.partition("=")[0] not in self.wanted]
        self.assertTrue(image_only)
        self.assertEqual(spark.environment_problems("dgx4", self.wanted, self.production, IMAGE_ENV), [])
        # Without the image's entries, the same variables are not the profile's.
        self.assertEqual(
            spark.environment_problems("dgx4", self.wanted, self.production, None),
            [f"dgx4: extra variable {item.partition('=')[0]}={item.partition('=')[2]!r}"
             for item in image_only],
        )

    def test_image_variable_set_to_another_value_is_reported(self) -> None:
        live = self.production + ["NVIDIA_VISIBLE_DEVICES=0"]
        live.remove("NVIDIA_VISIBLE_DEVICES=all")
        self.assertEqual(
            spark.environment_problems("dgx1", self.wanted, live, IMAGE_ENV),
            ["dgx1: NVIDIA_VISIBLE_DEVICES='0', expected the image's 'all'"],
        )


class SiteNodesTest(unittest.TestCase):
    def test_missing_site_file_points_at_the_example(self) -> None:
        import tempfile

        original = spark.ROOT
        with tempfile.TemporaryDirectory() as directory:
            spark.ROOT = Path(directory)
            try:
                with self.assertRaises(SystemExit) as raised:
                    spark.site_nodes()
            finally:
                spark.ROOT = original
        self.assertIn("config/nodes.example.json", str(raised.exception))

    def test_example_is_a_complete_site_file(self) -> None:
        example = spark.json.loads((ROOT / "config" / "nodes.example.json").read_text())
        self.assertEqual(sum(node["head"] for node in example["nodes"]), 1)
        for node in example["nodes"]:
            self.assertEqual(set(node), {"name", "rank", "management_ip", "head", "roce_peer_hcas"})


class CoolingTest(unittest.TestCase):
    """Benchmarks start with every node below the cooling threshold."""

    def setUp(self) -> None:
        # cool_nodes reports each cooling step; keep it out of the test output.
        progress = contextlib.redirect_stdout(io.StringIO())
        self.progress = progress.__enter__()
        self.addCleanup(progress.__exit__, None, None, None)

    def run_cooling(self, temps: dict[str, list[float]], service_active: bool = True,
                    timeout: float = 600.0):
        calls: list[tuple[str, ...]] = []
        readings = {name: list(values) for name, values in temps.items()}

        def fake_ssh(nodes, node, *command):
            calls.append((node["name"], *command))
            if command[:2] == ("sh", "-c"):
                values = readings[node["name"]]
                value = values.pop(0) if len(values) > 1 else values[0]
                return spark.subprocess.CompletedProcess(command, 0, f"{int(value * 1000)}\n", "")
            if command[:3] == ("systemctl", "is-active", "--quiet"):
                return spark.subprocess.CompletedProcess(command, 0 if service_active else 3, "", "")
            return spark.subprocess.CompletedProcess(command, 0, "", "")

        original = spark.run_ssh
        spark.run_ssh = fake_ssh
        try:
            nodes = {"ssh_user": "u", "nodes": [{"name": n} for n in temps]}
            records = spark.cool_nodes(nodes, nodes["nodes"], 55.0, timeout, poll_s=0)
        finally:
            spark.run_ssh = original
        return records, calls

    @staticmethod
    def fan_commands(calls, node: str) -> list[tuple[str, ...]]:
        return [call[1:] for call in calls if call[0] == node and call[1] == "sudo"]

    def test_all_cool_nodes_are_left_alone(self) -> None:
        records, calls = self.run_cooling({"dgx1": [48.0], "dgx2": [54.9]})
        self.assertEqual([r["cooled"] for r in records], [False, False])
        self.assertFalse(any(call[1] == "sudo" for call in calls))

    def test_one_hot_node_cools_every_node_until_all_are_below(self) -> None:
        records, calls = self.run_cooling({"dgx1": [58.0, 56.0, 54.5], "dgx2": [50.0]})
        expected = [
            ("sudo", "-n", "systemctl", "stop", "dgx-fan-control.service"),
            ("sudo", "-n", "dgx-fan-control", "set-state", "12"),
            ("sudo", "-n", "systemctl", "reset-failed", "dgx-fan-control.service"),
            ("sudo", "-n", "systemctl", "start", "dgx-fan-control.service"),
        ]
        self.assertEqual(self.fan_commands(calls, "dgx1"), expected)
        self.assertEqual(self.fan_commands(calls, "dgx2"), expected)
        self.assertEqual([r["cooled"] for r in records], [True, True])
        self.assertEqual(records[0]["final_c"], 54.5)
        self.assertEqual(records[0]["restored"], "dgx-fan-control.service")
        self.assertIn("fans restored", self.progress.getvalue())

    def test_without_the_service_fans_return_to_firmware_automatic(self) -> None:
        records, calls = self.run_cooling({"dgx3": [60.0, 50.0]}, service_active=False)
        self.assertIn(("sudo", "-n", "dgx-fan-control", "automatic"), self.fan_commands(calls, "dgx3"))
        self.assertEqual(records[0]["restored"], "automatic")

    def test_timeout_still_restores_every_node(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_ssh(nodes, node, *command):
            calls.append((node["name"], *command))
            out = "70000\n" if command[:2] == ("sh", "-c") else ""
            return spark.subprocess.CompletedProcess(command, 0, out, "")

        original = spark.run_ssh
        spark.run_ssh = fake_ssh
        try:
            nodes = {"ssh_user": "u", "nodes": [{"name": "dgx1"}, {"name": "dgx4"}]}
            with self.assertRaises(RuntimeError):
                spark.cool_nodes(nodes, nodes["nodes"], 55.0, 0.0, poll_s=0)
        finally:
            spark.run_ssh = original
        for node in ("dgx1", "dgx4"):
            self.assertEqual(self.fan_commands(calls, node)[-1],
                             ("sudo", "-n", "systemctl", "start", "dgx-fan-control.service"))

    def test_unreadable_node_stops_before_touching_fans(self) -> None:
        def fake_ssh(nodes, node, *command):
            return spark.subprocess.CompletedProcess(command, 0, "", "")

        original = spark.run_ssh
        spark.run_ssh = fake_ssh
        try:
            nodes = {"ssh_user": "u", "nodes": [{"name": "dgx2"}]}
            with self.assertRaises(RuntimeError):
                spark.cool_nodes(nodes, nodes["nodes"], 55.0, 600.0, poll_s=0)
        finally:
            spark.run_ssh = original

class MountTestScriptTest(unittest.TestCase):
    def test_one_script_reports_exactly_the_unavailable_sources(self) -> None:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as home:
            Path(home, "ok").write_text("")
            cluster = {"host": {"home": home}, "container": {"mounts": [
                ["{home}/ok", "/a", "ro"],
                ["{home}/missing dir", "/b", "rw"],
                ["{home}/ok", "/c", "rw"],
            ]}}
            result = subprocess.run(["sh", "-c", spark.mount_test_script(cluster)],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.splitlines(), [f"rw {home}/missing dir"])


class ClusterReplaceTest(unittest.TestCase):
    """start --replace stops every node's old container (concurrently) and rolls back on failure."""

    def run_start(self, fail_rename_on=None):
        import argparse
        from unittest import mock

        nodes = {"ssh_user": "u", "nodes": [
            {"name": "dgx1", "head": True, "rank": 0, "management_ip": "1", "roce_peer_hcas": []},
            {"name": "dgx2", "head": False, "rank": 1, "management_ip": "2", "roce_peer_hcas": []},
            {"name": "dgx3", "head": False, "rank": 2, "management_ip": "3", "roce_peer_hcas": []}]}
        cluster = {"deployment": {"launch_enabled": True}, "container": {"name": "c"}}
        calls = []
        guards = set()

        def run_ssh(_nodes, node, *command):
            calls.append((node["name"], command[:2]))
            code = 0
            if command[:2] == ("docker", "rename") and node["name"] == fail_rename_on:
                code = 1
            return subprocess.CompletedProcess(command, code, stdout="id\n", stderr="")

        import subprocess
        patches = [
            mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})),
            mock.patch.object(spark, "cluster_config_path", return_value=(None, "config/cluster.json")),
            mock.patch.object(spark, "rendered_docker_command", return_value=["docker", "run"]),
            mock.patch.object(spark, "require_local_deployment_state", return_value="rev"),
            mock.patch.object(spark, "remote_runtime_problems", return_value=[]),
            mock.patch.object(spark, "stop_memguard",
                              side_effect=lambda c, n, node: guards.discard(node["name"])),
            mock.patch.object(spark, "start_memguard",
                              side_effect=lambda c, n, node, *rest, **kw: guards.add(node["name"])),
            mock.patch.object(spark, "start_memguards"),
            mock.patch.object(spark, "wait_for_api"),
            mock.patch.object(spark, "memguard_is_active",
                              side_effect=lambda c, n, node: node["name"] in guards),
            mock.patch.object(spark, "run_ssh", side_effect=run_ssh),
            mock.patch.object(spark, "preserve_failed_start_logs", return_value=None),
            mock.patch.object(spark, "restore_containers"),
        ]
        for patch in patches:
            patch.start()
        try:
            args = argparse.Namespace(apply=True, replace=True, cluster_config="config/cluster.json")
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                code = spark.command_cluster_start(args)
            restore = spark.restore_containers
            return code, calls, restore
        finally:
            for patch in patches:
                patch.stop()

    def test_every_old_container_is_stopped_and_its_backup_removed(self) -> None:
        code, calls, restore = self.run_start()
        self.assertEqual(code, 0)
        stopped = sorted(node for node, command in calls if command == ("docker", "stop"))
        self.assertEqual(stopped, ["dgx1", "dgx2", "dgx3"])
        removed = sorted(node for node, command in calls if command == ("docker", "rm"))
        self.assertEqual(removed, ["dgx1", "dgx2", "dgx3"])
        restore.assert_not_called()

    def test_a_failed_rename_rolls_back_the_stopped_nodes(self) -> None:
        code, calls, restore = self.run_start(fail_rename_on="dgx2")
        self.assertEqual(code, 1)
        backups = restore.call_args.args[2]
        self.assertEqual(sorted(backups), ["dgx1", "dgx3"])
        self.assertNotIn(("dgx1", ("docker", "run")), calls)


class KernelPolicyPathTest(unittest.TestCase):
    def test_default_and_experiment(self):
        self.assertEqual(spark.kernel_policy_path({}), ROOT / "config/kernel-trial.json")
        self.assertEqual(spark.kernel_policy_path({"host": {"kernel_policy": "experiments/policy.json"}}),
                         ROOT / "experiments/policy.json")

    def test_outside_repository_rejected(self):
        for path in ("../policy.json", "/tmp/policy.json"):
            with self.assertRaises(ValueError):
                spark.kernel_policy_path({"host": {"kernel_policy": path}})
