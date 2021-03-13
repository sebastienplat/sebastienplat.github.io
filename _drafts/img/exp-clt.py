#modules
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
sns.set()

# lambda
lb = 5
expn = stats.expon(scale=lb)
norm = stats.norm(lb, lb)

# fig
fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (4 * 5, 4))
fig.suptitle('Sampling distribution of the sample mean for the Exponential distribution with lambda=5')

# pdf
x1 = np.linspace(0, 30, 101)
axes[0].plot(x1, expn.pdf(x1), label='exp')
axes[0].plot(x1, stats.norm.pdf(x1, lb, lb), label='normal')
axes[0].legend()
axes[0].set_title('PDF of exponential vs normal\nof same mean (5) and variance (25)')

# sampling dist of sample mean
x2 = np.linspace(0, 10, 101)
for (idx, size) in enumerate([10, 40, 100]):

    # samples
    np.random.seed(42 + idx)
    means = np.zeros(1000)
    for i in range(1000):
        sample = expn.rvs(size=size)
        means[i] = sample.mean()

    # dist of sample means
    axes[idx+1].hist(means, range=(0,10), bins=30, density=True, label='sample')
    axes[idx+1].plot(x2, stats.norm.pdf(x2, 5, 5/np.sqrt(size)), lw=2, label='CLT')
    axes[idx+1].legend(loc='upper left')
    axes[idx+1].set_title('Sampling distribution vs CLT Normal\nSample size {} - sampling mean {:.2f}'.format(size, means.mean()))

# labels
for ax in axes:
    _ = ax.set_xlabel('x')
    _ = ax.set_ylabel('probability')

# titles
plt.tight_layout(rect=[0, 0, 0.9, 0.9])

# show fig
plt.show()
