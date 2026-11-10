import numpy as np
def budget(conf, rc, scale, spill, tau, nondraft_per_req=1, knee=16):
    conf = np.asarray(conf, np.float64)
    n = conf.shape[0]
    S = np.cumprod(conf, axis=1)
    valid = np.ones_like(S, bool)
    if tau > 0: valid &= S >= tau
    sc = np.sort(S[valid])[::-1]
    E = np.concatenate(([n], n + np.cumsum(sc)))
    R = n*nondraft_per_req + np.arange(len(sc)+1)
    s = spill if n <= 1 else scale
    eff = np.minimum(R, knee) + s*np.maximum(R-knee, 0)
    U = E/(1+rc*eff)
    b = int(np.argmax(U))
    return b, b + n*nondraft_per_req, U
rc=0.035
tau_cal = min(0.2, 2*rc/(1+3*rc))
print("tau_cal", tau_cal)
c8 = np.tile([0.78,0.72,0.71,0.71,0.73],(8,1))
for sc in (1.0,0.55,0.85):
    b,t,U = budget(c8, rc, sc, 2.0, 0.2)
    print("c8 tau0.2 scale",sc,b,t)
    b,t,U = budget(c8, rc, sc, 2.0, max(0.12,tau_cal))
    print("c8 floor scale",sc,b,t)
typ=[[0.677,0.556,0.396,0.460,0.638]]
print("c1 typical S", np.cumprod(typ))
for tau in (tau_cal, max(0.12,tau_cal)):
    print("c1 tau", tau, budget(typ, rc, 0.55, 2.0, tau)[:2], budget(typ, rc, 0.85, 2.0, tau)[:2])
c1copy = np.full((1,16),0.95)
for sc in (0.55,0.85):
    print("c1 copy 0.95 spill2.0 scale",sc, budget(c1copy, rc, sc, 2.0, 0.0)[:2], "with tau floor", budget(c1copy, rc, sc, 2.0, 0.12)[:2], "if 0.85 applied to c1", budget(c1copy, rc, sc, sc, 0.0)[:2])
c2copy = np.full((2,16),0.95)
for sc in (0.55,0.85):
    print("c2 copy 0.95 scale",sc, budget(c2copy, rc, sc, 2.0, 0.0)[:2])
