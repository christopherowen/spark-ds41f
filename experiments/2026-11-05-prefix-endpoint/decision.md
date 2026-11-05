# Decision

Keep the focused patch as an unpromoted candidate and submit it upstream in draft.
The native cache regression and ownership tests support retaining the missing
encoder endpoint. The broad version also changed ordinary replay retention and was
rejected; the final patch is limited to a multi-token replay suffix.

Do not claim a TTFT improvement, production memory improvement or full-model
equivalence from CPU results. Leave launch disabled until an authorized serving
window can run the documented boot/repeat-boot gates with the exact built image.
No production configuration, published baseline or recorded benchmark is changed.

The alternatives audit in `alternatives.md` supports retaining exactly the two
reachable tail anchors for the current 128/256 replay/page geometry. Dense retention
passes correctness but has a growing cache cost; the selected policy matches its
append hit with one extra block per encoder group at contexts up to 1,000,256 tokens.
The best policy for middle-prefix forks and all live performance claims remain open.
