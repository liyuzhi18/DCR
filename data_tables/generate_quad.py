import numpy as np
from numpy.polynomial.laguerre import laggauss

n = 128
x, w = laggauss(n)            # points on [0, ∞)
with open("data_tables/quad_max_128.in", "w") as f:
    for i, (xi, wi) in enumerate(zip(x, w), 1):
        f.write(f"{i:3d}\t{xi: .14E}\t{wi: .14E}\n")
