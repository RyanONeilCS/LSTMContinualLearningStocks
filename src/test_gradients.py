"""
Gradient check for the backward pass

Compares every analytical gradient against a central finite difference
If BPTT is right the two numbers agree to about 1e-7

Run with
    python src/test_gradients.py
"""

from __future__ import annotations

import numpy as np

from lstm import parameters, forward, backward


#Small sizes so the check runs fast
#Real training uses D=3 H=32 B=64 T=60
D, H, B, T = 3, 8, 4, 12

rng = np.random.default_rng(1)

#One set of parameters and one fake batch
p = parameters(D, H, seed = 0)
X = rng.standard_normal((B, T, D))
y = rng.standard_normal((B, 1)) * 0.01

#Run forward then backward to get the analytical gradients
y_hat, caches, h_T = forward(X, p)
loss, grads = backward(y_hat, y, h_T, caches, p)

#Every gradient must match the shape of the thing it moves
print("shapes  ", {k: (p[k].shape, grads[k].shape) for k in p})
print("loss    ", loss)
print("finite  ", all(np.isfinite(v).all() for v in grads.values()))
print()

#Nudge one weight up and down and see how the loss actually moves
#Central difference since it is more accurate than a one sided one
eps = 1e-6
worst = 0.0

for name in ["W", "b", "W_y", "b_y"]:

    A = p[name]
    flat = A.reshape(-1)

    #Checking every weight is slow so sample a few from each array
    idxs = rng.choice(flat.size, size = min(8, flat.size), replace = False)
    errs = []

    for k in idxs:

        orig = flat[k]

        flat[k] = orig + eps
        yh, _, _ = forward(X, p)
        lp = np.mean((yh - y) ** 2)

        flat[k] = orig - eps
        yh, _, _ = forward(X, p)
        lm = np.mean((yh - y) ** 2)

        #Put it back before moving on
        flat[k] = orig

        #Measured slope against what backward reported
        num = (lp - lm) / (2 * eps)
        ana = grads[name].reshape(-1)[k]

        #Relative error so small gradients are not unfairly punished
        denom = max(abs(num), abs(ana), 1e-12)
        errs.append(abs(num - ana) / denom)

    m = max(errs)
    worst = max(worst, m)
    print(f"{name:4s} max relative error {m:.3e}")

print()
print("WORST", f"{worst:.3e}", "->", "PASS" if worst < 1e-4 else "FAIL")
