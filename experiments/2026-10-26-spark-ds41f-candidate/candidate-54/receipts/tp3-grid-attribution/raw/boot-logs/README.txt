L2 prefetch plan lines ([l2_prefetch] ...) copied read-only from the lab's boot logs.

1107-production-tp3-switched-sp{5,6,7}-l2plans.txt: sp5 results/private/lab/candidate-54-screen-tp3-switched/
  boot-candidate-54.engine-sp{5,6,7}.log. Despite the file name these lines are dated 11-10 01:42-01:44Z,
  before candidate-54's first (failed, "image is unavailable") start at 01:50Z, and show the pre-#961
  kernel (l2_prefetch.py:267 "grid=2 block=128 chunk=4096"): the lab captured the logs of the container
  then running, the restored spark-ds41f:1107 service. The plan sizes are the model's at TP3 and no
  candidate-54 patch changes them (#961 changes the kernel, not the plans).
candidate-54-again-tp4-switched-sp1-l2plans.txt: sp1 results/private/lab/candidate-54-screen-tp4-switched/
  boot-candidate-54-again.engine-sp1.log, candidate-54's own boot (l2_prefetch.py:300, "grid=2 block=32
  chunk=65536", the #961 kernel).

Per normal decoder layer, rank 0:
  TP3: WO 20.0 MB (wo_a 12.6 + wo_b head 7.4: capped), FFN 16.0 MB (capped), NEXT 20.0 MB (capped) = 56.0 MB
  TP4: WO 18.9 MB, FFN 14.8 MB, NEXT 14.8 MB = 48.5 MB
  In 64 KiB chunks (#961): TP3 306 + 245 + 306 = 857 chunks, TP4 289 + 226 + 226 = 741 chunks a layer
  (+16%); at the 200 ns pace that is a floor of ~171 us of side-stream time a layer at TP3, ~148 us at TP4.
  ENGRAM (the layer before an Engram layer) adds 20.0 MB at both.
