# Native source with compact dense selection

Base `a5468795`, runtime parent `4210b137`. The direct-merge RULER26 screen
finds one newly wrong naturally completed128K QA answer. Native scores26/26,
compact dense/source selection25/26. All official inputs and scores remain
unchanged; lower wall time cannot qualify that candidate.

One intended variable: `VLLM_DS41_PAGE_NATIVE_SOURCE=1`, an existing dispatch
setting, keeps native source selection and bypasses unused source summary
maintenance. Dense compact80 selection, budgets, fine arithmetic, checkpoint and
generation protocol remain unchanged. No benchmark-specific serving logic or
answer references enter the recipe. The exact fusion/parallel maintenance
experiments are not added simultaneously.

Before model inference, check the native cache reader on the compact80 prefix
with deliberately unwritten summary suffixes. Require all12 frozen actual
batches' native keys and source handoff to match Native68 exactly under strict
initialization checking. The source dispatcher and its declarations must select
native consistently. Then face the same frozen26-task RULER screen with two
fresh boots per arm. A newly wrong or incomplete answer rejects it. All later
full-context, independent held-out, coding and performance/admission gates remain
required even if this lean pilot passes. Keep every failure and restore receipt.

Source `727595b8` passes all12 actual native key/handoff comparisons under strict
initialization checking with unwritten compact suffixes (`runs/prefix-01`). CPU
inspection also verifies both page sizes route source and reindex plans and their
declarations to native, while dense plans retain the compact route. This is
cache/dispatch qualification only. The four-boot RULER26 pilot is separate and
must retain all score, completion, timing and restoration observations.
