

import json
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
from matplotlib import pyplot as plt
from scipy import stats

data_path = '../../data/'

def load_undata():
    """
    This function:
    + loads the UN countries data from csv
    + converts cols to the relevant dtype
    + split columns in two with the relevant dtype
    """

    # columns to split
    cols_to_split = [
        {
            'col_name': 'Life expectancy at birth (females/males, years)',
            'resulting_col_names': ['Life Exp - F', 'Life Exp - M'],
            'resulting_dtype': 'float'
        }
    ]

    # load dtypes from json
    undata_json_name = 'undata_country_profile_variables.json'
    with open(data_path + undata_json_name) as json_file:
        col_dtypes = json.load(json_file)
        
    # load UN Countries data
    undata_name = 'undata_country_profile_variables.csv'
    undata = pd.read_csv(
        data_path + undata_name, 
        header=0,
        names=col_dtypes.keys(),
        usecols=range(len(col_dtypes)), 
        #dtype=col_dtypes
    )

    # convert cols to relevant dtypes
    for col, dtype in col_dtypes.items():
        if dtype in ('integer', 'float'):
            undata[col] = pd.to_numeric(undata[col], errors='coerce', downcast=dtype)
        
        if dtype == 'integer':
            undata[col] = undata[col].astype('Int32')

    # split columns
    for col in cols_to_split:

        col_split = undata[col['col_name']].str.split('/', 1, expand=True)
        col_split.columns = col['resulting_col_names']

        if col['resulting_dtype'] in ('integer', 'float'):
            col_split = col_split.apply(pd.to_numeric, errors='coerce', downcast=dtype)

        if col['resulting_dtype'] == 'integer':
            col_split = col_split.astype('Int32')

        undata = undata.join(col_split)

    # add continents
    def continent(row):
        if (row['Region'].endswith('Europe')) and (row['country'] != 'Holy See'):
            return 'Europe'
        elif row['Region'].endswith('Asia'):
            return 'Asia'
        elif row['Region'].endswith('Africa'):
            return 'Africa'
        elif row['Region'] in ('CentralAmerica', 'SouthAmerica'):
            return 'Latam'
        elif row['Region'] == 'NorthernAmerica':
            return 'NorthernAmerica'
        else:
            return ('Rest Of World')

    undata['Continent'] = undata.apply(continent, axis=1)

    return undata


def life_exp_plot(df1, df2, vs_str, labels):
    """
    Plots compared life expectancies
    and influence of vs feature.
    """

    # variable to plot life exp against
    life_exp_f = 'Life Exp - F'
    if vs_str == 'GDP per capita':
        vs = 'GDP per capita (current US$)'
    elif vs_str == 'Unemployment':
        vs = 'Unemployment (% of labour force)'
    else:
        vs = vs_str

    
    # init subplots
    _, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize = (12, 5), tight_layout=True)

    # density plots for life exp (df1 vs df2)
    _ = sns.kdeplot(df1[life_exp_f], ax=ax1, label=labels[0], shade=True)
    _ = sns.kdeplot(df2[life_exp_f], ax=ax1, label=labels[1], shade=True)
    _ = ax1.set_title('Life Expectancy')

    # density plots for other variable (df1 vs df2)
    _ = sns.kdeplot(df1[vs], ax=ax2, label=labels[0], shade=True)
    _ = sns.kdeplot(df2[vs], ax=ax2, label=labels[1], shade=True)
    _ = ax2.set_xlim([0, None])
    _ = ax2.set_title(vs_str)

    # correlation between life exp vs variable
    df = pd.concat((df1, df2))
    _ = sns.scatterplot(x=vs, y=life_exp_f, data=df, hue='Continent', ax=ax3)
    _ = ax3.set_title('Life Exp vs ' + vs_str)


# --------------------

def load_olymp():
    """
    This function:
    + loads the athletes data from csv
    + converts cols to the relevant dtype
    """

    # load athletes data
    athletes_name = 'olyathswim.csv'
    athletes = pd.read_csv(
        data_path + athletes_name,
        usecols=range(1, 14)
    )

    # convert cols to relevant dtypes
    for col in ['Age', 'Height', 'Weight']:
        athletes[col] = pd.to_numeric(athletes[col], errors='coerce', downcast='integer')
        
    # medals flag
    athletes['MedalTF'] = athletes['Medal'].apply(lambda x: False if pd.isna(x) else True)

    # zoom on some races
    ath100m = ["Athletics Men's 100 metres", "Athletics Women's 100 metres"]
    ath10k = ["Athletics Men's 10,000 metres", "Athletics Women's 10,000 metres"]
    athletes['ShortEvent'] = ['100m' if x in ath100m else '10k' if x in ath10k else x for x in athletes['Event']]

    return athletes
    

def boxplot_with_factors(df, factors, outcome):
    """
    Draws two boxplots for weight distribution:
    + Sport only
    + Sport and Sex
    """
    _ = plt.figure(figsize = (12, 6), tight_layout=True)
    ax1 = plt.subplot2grid((1,4), (0,0), colspan=1)
    ax2 = plt.subplot2grid((1,4), (0,1), colspan=1, sharey=ax1)
    ax3 = plt.subplot2grid((1,4), (0,2), colspan=2, sharey=ax1)

    _ = sns.boxplot(y=outcome, x=factors[0], data=df, ax=ax1)
    _ = sns.boxplot(y=outcome, x=factors[1], data=df, ax=ax2)
    _ = sns.boxplot(y=outcome, x=factors[0], hue=factors[1], data=df, ax=ax3)


def kdeplot_wt_sport_sex(df):
    """
    Draws three density plots for weight distribution:
    + Sport only
    + Sport for men
    + Sport for women
    """

    athl = df.loc[df['Sport'] == 'Athletics']
    swim = df.loc[df['Sport'] == 'Swimming']

    _, (ax1, ax2, ax3) = plt.subplots(figsize = (14, 4), ncols=3)
    _ = sns.kdeplot(athl['Weight'], ax=ax1, label='Athletics')
    _ = sns.kdeplot(swim['Weight'], ax=ax1, label='Swimming')

    _ = sns.kdeplot(athl.loc[athl['Sex'] == 'M', 'Weight'], ax=ax2, label='Athl - M')
    _ = sns.kdeplot(swim.loc[swim['Sex'] == 'M', 'Weight'], ax=ax2, label='Swim - M')

    _ = sns.kdeplot(athl.loc[athl['Sex'] == 'F', 'Weight'], ax=ax3, label='Athl - F')
    _ = sns.kdeplot(swim.loc[swim['Sex'] == 'F', 'Weight'], ax=ax3, label='Swim - F')



class PowerTest:
    """
    Performs different power calculations
    based on two samples.
    """
    def __init__(self, samp1, samp2, alpha, power):
        """
        We initialize the class with our
        two samples and both alpha and power.
        """

        # samples
        self.samp1 = samp1
        self.samp2 = samp2

        # initialize formula
        self.analysis = sm.stats.power.TTestIndPower()

        # set parameters
        self.power = 0.8
        self.alpha = 0.05
        self.ratio = len(samp2) / len(samp1)
        self.effect_size = self.get_cohend(verbose=False)


    def get_sample_size(self, effect_size):
        """
        Returns the sample sizes for 
        capturing a given effect size.
        """

        samp1_size = self.analysis.solve_power(
            alpha=self.alpha,
            power=self.power,
            ratio=self.ratio,
            effect_size=effect_size
        )

        print('min sample size to detect effect size = {}: samp1: {:0.0f} - samp2: {:0.0f}'.format(
            effect_size,
            samp1_size, 
            samp1_size * self.ratio)
        )


    def get_effect_size(self, samp1_size):
        """
        Returns the effect size that can be 
        captured using a given sample size.
        """

        effect_size = self.analysis.solve_power(
            alpha=self.alpha,
            power=self.power,
            ratio=self.ratio,
            nobs1=samp1_size
        )

        print('   min effect size of full dataset: {:0.5f}'.format(effect_size))



    def get_power(self, samp1, samp2):
        """
        Returns the power of a given sample size
        for the dataset efect size.
        """

        power = self.analysis.solve_power(
            alpha=self.alpha,
            ratio=len(samp2) / len(samp1),
            nobs1=len(samp1),
            effect_size=self.effect_size
        )

        return power


    def get_cohend(self, verbose=True):
        """
        Returns Cohen's d of the two samples.
        """

        # Calculate difference between means and pooled standard deviation
        diff = self.samp1.mean() - self.samp2.mean()
        pooledstdev = np.sqrt((self.samp1.std()**2 + self.samp2.std()**2)/2 )

        # Calculate Cohen's d
        cohend = diff / pooledstdev

        if verbose:
            print('actual effect size of full dataset: {:0.5f}'.format(cohend))
        else:
            return cohend


    def rnd_sample_ttest(self, n, seed):
        """
        Draws one sample of size [n] from the two samples.
        Performs a t-test from the sample.
        Returns power for dataset effect size.
        """

        # subset sample
        samp1 = pd.DataFrame(self.samp1)
        samp1['sample'] = 'samp1'
        samp2 = pd.DataFrame(self.samp2)
        samp2['sample'] = 'samp2'
        subset = pd.concat((samp1, samp2)).sort_index().sample(n=n, random_state= seed)

        # split sample
        samp1 = subset.loc[subset['sample'] == 'samp1', 'Weight']
        samp2 = subset.loc[subset['sample'] == 'samp2', 'Weight']

        # power
        power = self.get_power(samp1, samp2)

        # Perform the two-sample t-test
        t_result = stats.ttest_ind(samp1, samp2)
        print('n={:>4}: pvalue = {:0.5f} / power = {:0.2f}'.format(n, t_result[1], power))




def wt_sample_ttest(df, n, seed):
    """
    Draws two random samples of size [n]: one for
    Athletics and one for Swimming (seeded for reproducibility).
    Performs a t-test on the two samples.
    """
    
    # Create two subsets, one for the athletics competitors and one for the swimmers
    s_athl = df.loc[df['Sport'] == 'Athletics'].sample(n=n, random_state= seed)
    s_swim = df.loc[df['Sport'] == 'Swimming'].sample(n=n, random_state= seed)

    # Perform the two-sample t-test
    t_result = stats.ttest_ind(s_athl['Weight'], s_swim['Weight'])
    print(t_result)


def wt_blocked_sample_ttest(df, n, seed):
    """
    Draws two random samples of size [n]: one for
    Athletics and one for Swimming (seeded for reproducibility).
    Blocked samples: draws n/2 F and n/2 M.
    Performs a t-test on the two samples.
    """
    
    # sports
    athl = df.loc[df['Sport'] == 'Athletics']
    swim = df.loc[df['Sport'] == 'Swimming']

    # subset blocks
    s_athl_m = athl.loc[athl['Sex'] == 'M'].sample(n=n//2, random_state= seed)
    s_athl_f = athl.loc[athl['Sex'] == 'F'].sample(n=n//2, random_state= seed)
    s_swim_m = swim.loc[swim['Sex'] == 'M'].sample(n=n//2, random_state= seed)
    s_swim_f = swim.loc[swim['Sex'] == 'F'].sample(n=n//2, random_state= seed)

    # Combine blocks
    s_athl = pd.concat([s_athl_m, s_athl_f])
    s_swim = pd.concat([s_swim_m, s_swim_f])
    
    # Perform the two-sample t-test
    t_result = stats.ttest_ind(s_athl['Weight'], s_swim['Weight'])
    print(t_result)
