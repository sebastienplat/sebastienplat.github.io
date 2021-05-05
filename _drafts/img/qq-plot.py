import numpy as np
import statsmodels.api as sm

# distributions
np.random.seed(42)
norm_expon = stats.norm(loc=1)
norm = stats.norm()
expon = stats.expon()
t = stats.t(df=3)

# plot
sd = np.linspace(-4, 4, 30)
sd_expon = np.linspace(-3, 5, 30)
percentiles = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(3, 3, figsize=(11,10))

# loop over dists
dist_names = ['Standard Normal', 'T-distribution\nwith 3 Degrees of Freedom', 'Standard Exponential']
for (i, dist) in enumerate([norm, t, expon]):
    if dist == expon:
        norm_dist = norm_expon
        sd_dist = sd_expon
        xlim_adjust = 1
    else:
        norm_dist = norm
        sd_dist = sd
        xlim_adjust = 0

    # pdf
    ax[0,i].plot(sd_dist, norm_dist.pdf(sd_dist), label='norm')
    ax[0,i].plot(sd_dist, dist.pdf(sd_dist), label='dist')
    ax[0,i].set_title(dist_names[i])
    ax[0,i].legend()

    # cdf
    ax[1,i].plot(sd_dist, norm_dist.cdf(sd_dist), label='norm')
    ax[1,i].plot(sd_dist, dist.cdf(sd_dist), label='dist')
    ax[1,i].legend()
 
    # ppf
    sm.qqplot(dist.ppf(percentiles), norm_dist, line='45', ax=ax[2,i])
    ax[2,i].set_xlim(-4+xlim_adjust, 4+xlim_adjust)

_ = fig.suptitle('PDF, CDF and QQ-Plot of several distributions\ncompared to a Normal distribution of same mean and variance')
fig.tight_layout(rect=[0, 0.03, 1, 0.92])
