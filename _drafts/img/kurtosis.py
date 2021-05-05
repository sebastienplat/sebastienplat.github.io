import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

# distributions (mean=0, variance=1, skewness=0)
laplace = stats.laplace(scale=1 / np.sqrt(2))
norm = stats.norm()
uniform = stats.uniform(-np.sqrt(12) / 2, 2*np.sqrt(12) / 2)

# plot
x = np.linspace(-5, 5, 100)
ax = plt.subplot()

# moments (stats uses the Fisher kurtosis by default so we add 3 to get the Pearson kurtosis)
for name, distr in zip(['laplace', 'norm', 'uniform'], [laplace, norm, uniform]):
    kurt = distr.stats(moments='k')
    ax.plot(x, distr.pdf(x), label='{:<8}: {:.1f}'.format(name, kurt+3))

# add details
_ = ax.set_title('Kurtosis of three distributions\nwith mean=0, std=1, skew=0')
_ = ax.legend()
