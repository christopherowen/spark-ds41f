# Host recovery policy

These files are the source of truth for the small management-plane recovery
policy installed on every Spark. They do not start, stop, or otherwise manage
the vLLM container.

The policy has five deliberately narrow responsibilities:

1. arm the Spark's existing SBSA hardware watchdog through systemd;
2. preserve a small amount of reclaim protection for SSH and Tailscale;
3. prevent a handful of stuck pre-authentication SSH children from exhausting
   the listener;
4. check the local SSH listener and Tailscale control socket once per minute,
   restarting only the affected management service when it is unresponsive;
   and
5. bound Plymouth's multi-user boot gate so a missing display-manager quit event
   cannot leave an otherwise healthy host indefinitely stuck in `starting`.

It does not panic on an ordinary host or cgroup OOM, panic on a generic hung
task, reset networking, restart Docker, or reboot merely because the gateway or
Internet is unavailable. The hardware watchdog is the last resort for a real
PID-1/system stall.

The Plymouth drop-in retains the normal graphical handoff for 30 seconds. If
that event is lost, it explicitly quits the splash daemon and lets the boot
continue. The unit has a 40-second outer bound and treats only GNU `timeout`'s
expected 124 status as successful; it does not mask Plymouth or alter kernel
command-line splash settings.

`grub-console.cfg` becomes `/etc/default/grub.d/zz-spark-console.cfg`. It sorts
after every other drop-in and removes `splash` from the kernel command line that
`dgxstation-grub`'s `menu.cfg` sets, keeping `quiet`. With the splash, Plymouth
leaves the active console in graphics mode after it quits, so an attached display
or KVM shows no text console. `apply` regenerates the boot menu when it still
passes `splash` and switches the active console to text mode at once; the
command-line change takes effect at the next reboot, which must follow a stopped
service. `check` fails until that reboot.

Install or verify the policy from a clean, published checkout on each node:

```sh
scripts/host-recovery apply
scripts/host-recovery check
```

`apply` validates `sshd` before changing either listener. Fresh SSH and
Tailscale processes are scheduled after the command exits so the OOM policy is
active without stranding an installation performed through Tailscale.

`scripts/host-recovery apply` also disables the display manager (`gdm3`) and
desktop services that serve nothing on a headless inference node: Bluetooth,
CUPS with cups-browsed, snapd and its repair timer (every installed snap is a
desktop application; Docker, the NVIDIA driver, Tailscale and DKMS come from
apt), and NVIDIA's DGX Dashboard, whose periodic update checks
(`/opt/nvidia/spark-ota-check`) peak near 850 MiB and wake fwupd and
PackageKit. Re-enable the dashboard with
`sudo systemctl enable --now dgx-dashboard.service dgx-dashboard-admin.service`
when its web interface is wanted.
`check` and `bin/spark doctor --live` warn about any that is enabled or running;
a warning never fails either command or blocks a benchmark.
fwupd stays installed: it delivers the embedded-controller, UEFI, ConnectX-7
and NVMe firmware from LVFS. `apply` disables only its daily refresh timer and
stops the resident daemon; `sudo fwupdmgr refresh`, `get-updates` and `update`
start the daemon on demand over D-Bus (stop it again with
`sudo systemctl stop fwupd` afterwards).

