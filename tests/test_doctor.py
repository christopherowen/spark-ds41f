"""Host-independent tests for doctor's node checks."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)


class DesktopTest(unittest.TestCase):
    def test_headless_node_passes(self) -> None:
        self.assertEqual(spark3.desktop_problems("dgx1", "multi-user.target\ninactive\n"), [])

    def test_graphical_target_and_running_display_manager_are_reported(self) -> None:
        problems = spark3.desktop_problems("dgx2", "graphical.target\nactive\n")
        self.assertEqual(len(problems), 2)
        self.assertIn("boots to graphical.target", problems[0])
        self.assertIn("display manager is active", problems[1])

    def test_display_manager_started_by_hand_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx3", "multi-user.target\nactive\n")
        self.assertEqual(
            problems,
            ["dgx3: display manager is active; run sudo systemctl disable --now display-manager.service"],
        )

    def test_missing_output_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx1", "")
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
        self.assertEqual(spark3.boot_order_problems("dgx1", DGX1_BOOT), [])

    def test_network_boot_first_is_reported(self) -> None:
        problems = spark3.boot_order_problems("dgx3", DGX3_PXE_FIRST_BOOT)
        self.assertEqual(len(problems), 1)
        self.assertIn("UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller", problems[0])
        self.assertIn("BootOrder 0002,0004", problems[0])
        self.assertIn("run sudo efibootmgr --bootorder 0004,0002", problems[0])

    def test_fixed_order_passes(self) -> None:
        fixed = DGX3_PXE_FIRST_BOOT.replace("BootOrder: 0002,0004", "BootOrder: 0004,0002")
        self.assertEqual(spark3.boot_order_problems("dgx3", fixed), [])

    def test_unreadable_boot_order_is_reported(self) -> None:
        self.assertEqual(
            spark3.boot_order_problems("dgx2", ""),
            ["dgx2: cannot read the UEFI boot order"],
        )


FAN_WORKING = (
    "modeset=Y\n"
    "fbdev=Y\n"
    "drm_masters=\n"
    "console=tty1:0\n"
    "cmdline_splash=0\n"
    "grub_splash=0\n"
    "idle_services=\n"
    "kernel=7.0.0-1019-nvidia\n"
    "fan_dkms=dgx-spark-fan-control/0.1.3, 7.0.0-1019-nvidia, aarch64: installed\n"
    "fan_module=1\n"
    "fan_cooling_device=1\n"
    "fan_service=enabled/active\n"
)


def facts(**changes: str) -> dict[str, str]:
    values = spark3.host_facts(FAN_WORKING)
    values.update(changes)
    return values


class ModesetTest(unittest.TestCase):
    def test_kernel_mode_setting_passes(self) -> None:
        self.assertEqual(spark3.modeset_problems("dgx1", facts()), [])

    def test_disabled_mode_setting_is_reported(self) -> None:
        problems = spark3.modeset_problems("dgx1", facts(modeset="N"))
        self.assertEqual(len(problems), 1)
        self.assertIn("modeset is N, expected Y", problems[0])

    def test_unreadable_mode_setting_is_reported(self) -> None:
        problems = spark3.modeset_problems("dgx1", facts(modeset=""))
        self.assertIn("modeset is unreadable", problems[0])


class ConsoleTest(unittest.TestCase):
    def test_working_console_passes(self) -> None:
        self.assertEqual(spark3.fbdev_problems("dgx3", facts()), [])
        self.assertEqual(spark3.drm_master_problems("dgx3", facts()), [])
        self.assertEqual(spark3.console_mode_problems("dgx3", facts()), [])

    def test_fbdev_off_is_reported(self) -> None:
        problems = spark3.fbdev_problems("dgx3", facts(fbdev="N"))
        self.assertEqual(len(problems), 1)
        self.assertIn("fbdev is N, expected Y", problems[0])

    def test_unreadable_fbdev_is_reported(self) -> None:
        problems = spark3.fbdev_problems("dgx3", facts(fbdev=""))
        self.assertIn("fbdev is unreadable", problems[0])

    def test_drm_master_holder_is_reported(self) -> None:
        problems = spark3.drm_master_problems(
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
        problems = spark3.drm_master_problems("dgx3", facts(drm_masters="unreadable"))
        self.assertIn("cannot read the DRM clients", problems[0])
        missing = spark3.host_facts("modeset=Y\n")
        self.assertIn("cannot read the DRM clients", spark3.drm_master_problems("dgx3", missing)[0])

    def test_graphics_mode_console_is_reported(self) -> None:
        problems = spark3.console_mode_problems("dgx3", facts(console="tty1:1"))
        self.assertEqual(len(problems), 1)
        self.assertIn("the active console (tty1) is in graphics mode", problems[0])

    def test_unreadable_console_mode_is_reported(self) -> None:
        problems = spark3.console_mode_problems("dgx3", facts(console="tty1:"))
        self.assertEqual(problems, ["dgx3: cannot read the active console mode"])

    def test_splash_boot_is_reported_with_its_fix(self) -> None:
        problems = spark3.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="1"))
        self.assertEqual(len(problems), 1)
        self.assertIn("scripts/host-recovery apply", problems[0])
        self.assertIn("zz-spark-console.cfg", problems[0])

    def test_configured_splash_removal_waits_for_a_reboot(self) -> None:
        problems = spark3.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="0"))
        self.assertEqual(len(problems), 1)
        self.assertIn("reboot with the service stopped to apply", problems[0])

    def test_no_splash_passes(self) -> None:
        self.assertEqual(spark3.splash_problems("dgx1", facts()), [])

    def test_every_console_problem_names_a_fix(self) -> None:
        for problems in (
            spark3.fbdev_problems("dgx3", facts(fbdev="N")),
            spark3.console_mode_problems("dgx3", facts(console="tty1:1")),
            spark3.modeset_problems("dgx3", facts(modeset="N")),
        ):
            self.assertRegex(problems[0], r"; (run|remove) ")

    def test_modeset_message_names_the_carveout(self) -> None:
        problems = spark3.modeset_problems("dgx1", facts(modeset="N"))
        self.assertIn("display carve-out cannot be allocated", problems[0])


class IdleServicesTest(unittest.TestCase):
    def test_idle_services_are_reported_with_their_fix(self) -> None:
        problems = spark3.idle_service_warnings(
            "dgx2", facts(idle_services="bluetooth.service,snapd.socket")
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("bluetooth.service, snapd.socket enabled or running", problems[0])
        self.assertIn(
            "sudo systemctl disable --now bluetooth.service snapd.socket", problems[0]
        )
        self.assertIn("scripts/host-recovery apply", problems[0])

    def test_no_idle_services_pass(self) -> None:
        self.assertEqual(spark3.idle_service_warnings("dgx2", facts()), [])
        self.assertEqual(
            spark3.idle_service_warnings("dgx2", spark3.host_facts("modeset=Y\n")), []
        )

    def test_warnings_do_not_fail_doctor(self) -> None:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark3.report_doctor([], ["dgx2: snapd.service running"], live=True)
        self.assertEqual(status, 0)
        self.assertIn("WARN: dgx2: snapd.service running", output.getvalue())
        self.assertIn("configuration OK; live cluster matches with 1 warning", output.getvalue())

    def test_problems_fail_doctor_as_errors(self) -> None:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = spark3.report_doctor(["dgx1: container is not running"], [], live=True)
        self.assertEqual(status, 1)
        self.assertIn("ERROR: dgx1: container is not running", output.getvalue())

    def test_query_lists_every_idle_unit(self) -> None:
        for unit in spark3.IDLE_SERVICES:
            self.assertIn(unit, spark3.HOST_FACTS_QUERY)
        # fwupd.service itself runs legitimately after a manual fwupdmgr call.
        self.assertIn("fwupd-refresh.timer", spark3.IDLE_SERVICES)
        self.assertNotIn("fwupd.service", spark3.IDLE_SERVICES)


class SeverityTest(unittest.TestCase):
    """Required capabilities are errors; latent or minor findings are warnings."""

    def test_required_display_pieces_are_errors(self) -> None:
        for problems in (
            spark3.modeset_problems("dgx1", facts(modeset="N")),
            spark3.fbdev_problems("dgx1", facts(fbdev="N")),
            spark3.drm_master_problems("dgx1", facts(drm_masters="Xorg/123")),
            spark3.console_mode_problems("dgx1", facts(console="tty1:1")),
        ):
            self.assertEqual(len(problems), 1)
            self.assertNotIsInstance(problems[0], spark3.Warn)

    def test_latent_and_unreadable_findings_warn(self) -> None:
        for problems in (
            spark3.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="1")),
            spark3.splash_problems("dgx1", facts(cmdline_splash="1", grub_splash="0")),
            spark3.drm_master_problems("dgx1", facts(drm_masters="unreadable")),
            spark3.console_mode_problems("dgx1", facts(console="tty1:")),
            spark3.idle_service_warnings("dgx1", facts(idle_services="cups.service")),
            spark3.desktop_problems("dgx1", "graphical.target\ninactive\n"),
            spark3.boot_order_problems("dgx1", ""),
        ):
            self.assertEqual(len(problems), 1)
            self.assertIsInstance(problems[0], spark3.Warn)

    def test_running_display_manager_is_an_error(self) -> None:
        problems = spark3.desktop_problems("dgx1", "multi-user.target\nactive\n")
        self.assertEqual(len(problems), 1)
        self.assertNotIsInstance(problems[0], spark3.Warn)

    def test_split_findings_keeps_order(self) -> None:
        errors, warnings = spark3.split_findings(
            ["a", spark3.Warn("b"), "c", spark3.Warn("d")]
        )
        self.assertEqual(errors, ["a", "c"])
        self.assertEqual(warnings, ["b", "d"])


class FanControlTest(unittest.TestCase):
    def test_working_fan_floor_passes(self) -> None:
        self.assertEqual(spark3.fan_control_problems("dgx1", facts()), [])

    def test_missing_dkms_module_is_the_only_report(self) -> None:
        problems = spark3.fan_control_problems(
            "dgx3", facts(fan_dkms="", fan_module="0", fan_cooling_device="0", fan_service="/inactive")
        )
        self.assertEqual(
            problems, ["dgx3: DKMS dgx-spark-fan-control is not installed for 7.0.0-1019-nvidia; run "
                "sudo dkms autoinstall -k 7.0.0-1019-nvidia"]
        )

    def test_unloaded_module_is_reported(self) -> None:
        problems = spark3.fan_control_problems("dgx3", facts(fan_module="0"))
        self.assertIn("dgx_ec_fan_control is not loaded", problems[0])

    def test_refused_cooling_device_is_reported(self) -> None:
        # dgx3 on firmware 5.36_0ACUM027: the driver loads, then the EC rejects
        # its capability read and it refuses to register the cooling device.
        problems = spark3.fan_control_problems(
            "dgx3", facts(fan_cooling_device="0", fan_service="disabled/inactive")
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("cooling device is missing", problems[0])

    def test_stopped_daemon_is_reported(self) -> None:
        problems = spark3.fan_control_problems("dgx1", facts(fan_service="enabled/failed"))
        self.assertEqual(
            problems, ["dgx1: dgx-fan-control.service is enabled/failed, expected enabled/active; "
                "run sudo systemctl enable --now dgx-fan-control.service"]
        )


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
        self.assertEqual(spark3.roce_gid_problems("dgx1", 3, ["rocep1s0f0"], HEALTHY_GIDS), [])

    def test_shifted_gid_names_the_slot_and_interface(self) -> None:
        problems = spark3.roce_gid_problems("dgx3", 3, ["rocep1s0f0"], SHIFTED_GIDS)
        self.assertEqual(len(problems), 1)
        self.assertIn("rocep1s0f0 GID index 3 is empty", problems[0])
        self.assertIn("index 4; re-activate enp1s0f0np0", problems[0])

    def test_ipv6_roce_v2_at_the_index_is_reported(self) -> None:
        problems = spark3.roce_gid_problems("dgx1", 1, ["rocep1s0f0"], HEALTHY_GIDS)
        self.assertIn("GID index 1 is RoCEv2 fe80", problems[0])
        self.assertIn("at index 3", problems[0])

    def test_missing_device_and_missing_ipv4_are_reported(self) -> None:
        output = "rocep1s0f1 missing\nrocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0\n"
        problems = spark3.roce_gid_problems("dgx2", 3, ["rocep1s0f0", "rocep1s0f1"], output)
        self.assertEqual(
            problems,
            [
                "dgx2: rocep1s0f0 GID index 3 is empty; it has no IPv4 RoCE v2 GID",
                "dgx2: RDMA device rocep1s0f1 is missing",
            ],
        )


class ClockLatchTest(unittest.TestCase):
    def test_serving_node_at_full_clock_passes(self) -> None:
        self.assertEqual(spark3.clock_latch_problems("dgx1", facts(gpu="2411, 11.47"), True), [])

    def test_latched_serving_node_is_reported(self) -> None:
        # dgx3 on 2026-09-26: 520-565 MHz at about 10 W, ignoring nvidia-smi -lgc.
        problems = spark3.clock_latch_problems("dgx3", facts(gpu="559, 9.90"), True)
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU clock is 559 MHz at 9.90 W while serving", problems[0])

    def test_idle_node_without_the_service_is_not_judged(self) -> None:
        self.assertEqual(spark3.clock_latch_problems("dgx2", facts(gpu="208, 4.1"), False), [])

    def test_unreadable_clock_is_reported(self) -> None:
        self.assertEqual(
            spark3.clock_latch_problems("dgx2", facts(gpu=""), True),
            ["dgx2: cannot read the GPU clock"],
        )


if __name__ == "__main__":
    unittest.main()
