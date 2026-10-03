# TP3 triangle revalidation

Owner-authorized testing after restoring the dgx1–dgx2–dgx3 triangle.
The generated `tp3` control uses the existing r5o image and transport settings.
The installed image ID is `aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`, already recorded as the base of the TP4 experiments. Its three source-tree labels match r5o, but its ID differs from the historical promoted artifact; the fresh run records this explicitly.

Network changes replace the former dgx1–dgx4 and dgx3–dgx4 addresses with the two 10.13 subnets. Only the affected fabric connections are activated; management networking is preserved. Each original netplan file is saved under `/root/spark3-tp3-revalidation-20261003/`. The ignored `config/nodes-tp3.local.json` records GID slots, subnets and peer interfaces.

Sequence: verify every directed fabric lane, publish/sync the generated control to the isolated qualification checkout, run model-free collective correctness including graph replay, then coordinated serving startup and quality/decode/source-prefill/prefix measurements. Record all failures and distinguish fresh TP3 measurements from the earlier TP4 run. The historical baseline is immutable.

Status: preparation; results pending.
