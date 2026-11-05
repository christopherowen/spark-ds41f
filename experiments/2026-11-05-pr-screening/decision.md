# Current decision after owner clarification

Continue every PR through a defensible disposition. C42 built successfully but
failed one required IKET integration test before any serving measurement. Its
failed image is rejected; the compiler idea has no performance verdict. The
earlier closure of875/879–883 applied the bounded rework rule too broadly:
pre-existing fixes had not corrected this new integration defect. All six PRs
are reopened at their unchanged heads. One focused #880 correction is assigned,
with a distinct new source/image and full gates before measurement. C44's
metadata-only proposal remains withdrawn. The [update and scope](runs/compiler-rework-disposition.json)
supersede the historical idea-drop and closure decisions below. C40 remains the
validated default-enabled base. C43 frontend cleanup stays independent on C40.

## Previous decisions retained as history

# Current decision

The PR queue is not complete. Retain unique work, merge reviewed evidence and
correctness controls, and qualify each selected runtime source before accepting
its claim. No source-only or component-only pass stands in for full image and
serving evidence. Continue from corrected default-enabled C40; rejected C41 is
preserved and excluded from subsequent bases.

#865 and #911 are retained as prior observations, without a new performance or
release claim. #865 used site policies outside local qualification (including
warning-only memory handling, no installed fan control and unqualified-shape
startup); none are adopted here. #911 retains the deliberate partial cohort and
original exit1 separately from the current recorded-collector foreign run.

#851 failed same-layer/different-runtime-configuration repair. Its one focused
correction uses existing store-independent content identity and fails closed on
inspection errors. Corrected source passes57 independently repeated focused
checks and755 repository tests with5 skips. It is merged without a serving-image
change. Original failure/setup logs and the corrected audit remain preserved.

Compiler refresh #875 has authored parent correctness failures and five separate
fixes. The existing exact patches are combined explicitly in C42, with fresh
canonical source proof and no silent omission. Image, feature/release gates and
two serving boots remain pending; neither combined quality nor performance is
claimed yet. Other draft/active changes are not closed simply to empty the list.

C42 compiler integration is rejected at the release gate (8073pass/360skip/1fail,
zero serving points). The failed IKET prepared/frozen call is traced to #880's
wrapper bypass, independent of the separate shared-cache identity gap. Metadata
proposal C44 cannot fix it and is withdrawn unmeasured. Close875/879–883 as not
adopted with this rejected dependency; standalone positive results for the other
fixes remain valid within their own scope and are retained. No source branches
or native images are deleted. Continue separate C43 frontend cleanup on C40.
