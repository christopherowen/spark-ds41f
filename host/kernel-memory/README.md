# Base-page memory policy

The 4 KiB and 64 KiB kernels need different swap-file headers. The boot service
selects `/swap.img` on 4 KiB or `/swap-64k.img` on 64 KiB. The original swap file
is retained unchanged; its fstab entry is `noauto` because the service owns
activation. Rollback to the old kernel therefore restores the original swap.

On 64 KiB only, the service disables transparent huge pages and sets
`vm.min_free_kbytes=45166`, the reserve observed on the original 4 KiB profile.
This avoids the large PMD-THP watermark while preserving a free-memory reserve.
This policy change is recorded as part of the host trial; it is not a claim that
changing the base page size alone determines every memory or timing difference.
The 4 KiB THP policy remains unchanged.

Installation and the initial swap formatting are performed by the published
`experiments/2026-10-02-kernel-64k/install-boot-support.sh`. It does not reboot,
activate the service, or swap off a live file. The service runs at boot before
swap.target and Docker, with no GPU operations or serving control.

To remove this policy after returning to 4 KiB, disable the service, restore
the saved `/var/lib/spark3/kernel-64k-boot/fstab.before`, reload systemd and
verify the original swap is active. Never remove an active swap file.
