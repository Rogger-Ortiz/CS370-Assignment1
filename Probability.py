import pandas as pd
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

# Helper functions #


def halfround(num):
    return round(num*2)/2

### Problem 1 (80 points) ###


# 1a) Write the code for generating the gs variable. This is the simplest random variable of the problem and can be generated independent of the others.
gs = []
for i in range(0, 10000):
    num = np.random.normal(loc=7.25, scale=0.875)
    gs.append(halfround(num))

# 1b) We have three variables, ak, pp, and ptime. Write the code for generating these variables from Multivariate Gaussian distribution and replicate the associated plots.
U = [[1.0, 0.6, -0.9],
     [0.6, 1.0, -0.5],
     [-0.9, -0.5, 1.0]]

APT = np.random.multivariate_normal(mean=(0, 0, 0), cov=U, size=10000)

figure, axis = plt.subplots(3, 3)
axis[0, 0].set_title("ak")
axis[1, 0].plot(APT[:, 0], APT[:, 1])
axis[2, 0].plot(APT[:, 0], APT[:, 2])

axis[0, 1].plot(APT[:, 1], APT[:, 0])
axis[1, 1].set_title("pp")
axis[2, 1].plot(APT[:, 1], APT[:, 2])

axis[0, 2].plot(APT[:, 2], APT[:, 0])
axis[1, 2].plot(APT[:, 2], APT[:, 1])
axis[2, 2].set_title("ptime")

plt.show()
>>>>>>> 5691336c792ce29e9268e2b1ca18748843094c07

# 1c) Perform the probability integral transform and replicate the associated plots.

APT = sc.stats.norm.cdf(APT)

figure, axis = plt.subplots(3, 3)
axis[0, 0].set_title("ak")
axis[1, 0].plot(APT[:, 0], APT[:, 1])
axis[2, 0].plot(APT[:, 0], APT[:, 2])

axis[0, 1].plot(APT[:, 1], APT[:, 0])
axis[1, 1].set_title("pp")
axis[2, 1].plot(APT[:, 1], APT[:, 2])

axis[0, 2].plot(APT[:, 2], APT[:, 0])
axis[1, 2].plot(APT[:, 2], APT[:, 1])
axis[2, 2].set_title("ptime")

plt.show()

# 1d) Perform the inverse transform sampling.
# 1e) Replicate the final plot showcasing the correlations between the variables.


### Problem 2 (20 Points) ###

# 2a) Write the expression of the sample correlation matrix.
# 2b) Write the expression of the sample correlation matrix that can be estimated recursively and plot the elements of the sample correlation matrix from i=1 to i=100.
