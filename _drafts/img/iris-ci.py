# ---------- MODULES ---------- #

# modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import datasets
from statsmodels.distributions import ECDF

sns.set()

# colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# ---------- DATA ---------- #

# load & format iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
species = ['setosa', 'versicolor', 'virginica']


# ---------- PLOTS ---------- #

# fig
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize = (3 * 5, 1 * 4))

# boxplots
_ = sns.boxplot(y='species', x='petal length (cm)', data=iris_df, ax=ax1, order=reversed(species), palette=colors[2::-1])

# sample statistics for all three species
for i, species in enumerate(species):

    petal_length = iris_df.loc[iris_df['species'] == species, 'petal length (cm)'].to_numpy()
    
    # ECDF
    ecdf = ECDF(petal_length)
    ax2.plot(ecdf.x, ecdf.y, marker='.', linestyle='none', label=species)

    # sample stats
    size = len(petal_length)
    mean = np.mean(petal_length)
    std = np.std(petal_length)
    median = np.median(petal_length)

    # CI for pop mean
    tdist = stats.t(df=size-1, loc=mean, scale=std)
    interval = tdist.interval(0.95)
    ax3.hlines(species, interval[0], interval[1], color=colors[i], lw=2.5)
    ax3.vlines([interval[0], mean, interval[1]], i - 0.1, i + 0.1, color=colors[i], lw=2.5)

    # summary
    #print('{: <10}: median: {:0.2} - mean: {:0.3} - std: {:0.2} - interval: ({:0.2f}, {:0.2f})'.format(species, median, mean, std, interval[0], interval[1]))

# labels
_ = ax2.set_xlabel('petal length (cm)')
_ = ax2.set_ylabel('ECDF')
_ = ax3.set_xlabel('petal length (cm)')
_ = ax3.set_ylabel('species')

# titles
_ = ax1.set_title('Boxplots')
_ = ax2.set_title('ECDFs')
_ = ax3.set_title('Confidence Intervals of the pop. mean')

fig.suptitle('Comparizon of the distribution of petal lenghts across species')

# legend
handles, labels = ax2.get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 0.9, 0.9])
_ = fig.legend(handles, labels, loc='right', bbox_to_anchor=(1.02, 0.5))
