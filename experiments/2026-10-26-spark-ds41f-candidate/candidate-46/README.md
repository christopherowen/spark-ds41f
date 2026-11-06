# Candidate46: frontend cleanup with isolated fixture teardown

Unpromoted source proposal. Preserve C43’s failed image and verdict. The runtime
is unchanged from C43: the two frontend lifetime fixes on corrected default-copy
C40. The only new source delta is the reviewed one-file test-fixture correction.

Canonical source: d92f9c26e7e749ed15850d24a9e419ebad90a91f / 8056f1ef9761065bf9c66cb82b008ff71b6f310a; 577-patch identity aa971421ba2bb74223b8dbc046222700ec9dc2049b0547ae1da686e3ced94ffe.

The fresh remote replay and independent correction reapplication passed under
one CPU, 8 GiB and no swap. Actual-C43-image pytest witness, new image, complete
138-file release gates, strict top-k initcheck, all functional outcomes and both
full performance boots remain pending. Disabled TP3/TP4 profiles are source/build
controls. Named TP3 qualification profiles require the authorized shared window;
no TP4 serving launch or promotion is authorized.
