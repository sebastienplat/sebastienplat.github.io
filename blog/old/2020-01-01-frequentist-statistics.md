---
title:  "Frequentist Statistics"
date:   2020-01-20
categories: statistics
---

# Frequentist Statistics

Statistics help us understand the **relationship that exist between events**, if any. It allows us to **predict future behavior** and guide business decisions.

**Frequentist** statistics assume that:
+ the parameter $\theta$ of a population variable is set but unknown
+ the data $X$ we collect is randomly generated via the variable distribution
+ we can estimate the parameter value $\hat{\theta}$ by leveraging the CLT:

$$\hat{\theta} = argmax_\theta P(X | \theta)$$

**Bayesian** statistics assume that:
+ the parameter is a random variables that has a probability distribution
+ data is set from observations (there is no randomness after the fact)
+ we can model the probability distribution of the parameter value (called **posterior distribution**) based on the data:

$$P(\theta | X)$$

_Note: this notebook focuses on frequentist statistics._

## Data Types

We leverage several kinds of data to do statistical analysis:

+ Categorical
    + nominal: cannot be sorted (flavors of ice cream)
    + ordinal: can be sorted; lack scale (survey responses)
+ Continuous
    + interval: lacks a "zero" point (temperature)
    + ratio: true "zero" point (age, salary)

A distribution shows the probable outcomes of a variable.
+ **discrete**: probability of being exactly equal to some value. The sum of all probabilities is one. Its probability distribution is called the **Probability Mass Function of PMF**.
+ **continuous**: probability of falling within any interval of values. The area under the probability curve is equal to one. Its probability distribution is called the **Probability Density Function or PDF**

For both cases, the probability $P(X<=x)$ is given by the **Cumulative Distribution Function** or CDF.

___

## Discrete Data

### Uniform distribution

Probabilities are **evenly distributed** across the sample space, ie each value has the same probability of occuring (example: roll of a fair dice).

### Binomial distribution

There are **two mutually exclusive outcomes**. Examples include heads vs tail, success vs failure, healthy vs sick. The associated experiments are called a **Bernoulli Trial** if the probability of success is constant.

The probability of observing $k$ successes for $n$ independant Bernoulli trials with a probability of success $p$ is given by the PMF of the Binomial Distribution. The highest probability is for $k = n p$.


```python
# example: probability of rolling a five three times out of 16 dice rolls
from scipy.stats import binom
print('{:.1%}'.format(binom.pmf(k=3, n=16, p=1/6)))
```

> 24.2%
    

### Poisson distribution

Very similar to the Binomial Distribution, but measures the number of times an event occurs **in an interval of time or space**. It assumes a constant average rate of occurence $\lambda$.

The probability of observing $k$ events in an interval is given by the PMF of the Poisson Distribution. The highest probability of the Poisson Distribution is for $k = \lambda$.


```python
# example: probability of having only four deliveries between 4PM and 5PM this friday when the average is 8
from scipy.stats import poisson
print('{:.1%}'.format(poisson.pmf(k=4, mu=8)))
```

    5.7%
    

We can also calculate the probabilities of having fewer than $k$ successes with the **cumulative distribution function**.


```python
# example: probability of having four deliveries or less between 4PM and 5PM this friday when the average is 8
from scipy.stats import poisson
print('{:.1%}'.format(poisson.cdf(k=4, mu=8)))
```

    10.0%
    

We can also apply the same logic to smaller intervals if we can assume that events are equally distributed inside each interval. For instance, if $\lambda_{minute} = \lambda_{hour} / 60$


```python
# example: probability of having zero deliveries between 4:00PM and 4:05PM this friday when the hourly average is 8
from scipy.stats import poisson
print('{:.1%}'.format(poisson.pmf(k=0, mu=8/(60/5))))
```

    51.3%
    

___

##  Continuous Data

Continuous data are typically associated with their:

+ **Central Tendency**: average value a variable can take
+ **Dispersion**: spread from the average

### Median and IQR

+ median: median value; insensitive to outliers
+ range: maximum - minimum; extremely sensitive to outliers
+ interquartile range (IQR): difference between the third and first quartile

**Quartiles** divide a rank-ordered data set into four parts of equal size; each value has a 25% probability of falling into each quartile. The **Interquartile range (IQR)** is equal to Q3 minus Q1. This is why it is sometimes called the midspread or **middle 50%**. Q2 is the median.
    
**Box-plots** are a good way to summarize a distribution using its quartiles:

+ IQR with the median inside
+ whiskers (Tukey boxplot): values within 1.5 IQR of Q1 and Q3
+ remaining outliers

### Mean and Standard Deviation

+ mean: calculated average; sensitive to outliers
+ variance: average squared distance from the mean; emphasis on higher distances
+ standard deviation: square root of variance; same unit as the measure itself


_Note: sample variance is calculated with $n-1$ where $n$ is the sample size, in what is called the **Bessel's correction**. It reduces the bias due to finite sample size: you need to account for the variance of sample mean minus population mean. More details [here](https://en.wikipedia.org/wiki/Bessel%27s_correction)._

### Normal Distribution

Many real-life variables roughly follow the Gaussian or **Normal Distribution**: height, weight, test scores, etc. We can use it to model their behavior.

A normal distribution:
+ is centered around its mean
+ has 2/3 of all its values inside +/- 1SD from the mean
+ has 95% of all its values inside +/- 2SD from the mean
+ had 99.5% of all its values inside +/- 3SD from the mean

_Note1: the Standard Normal Distribution has a mean of 0 and and a standard deviation of 1._
_Note2: we can convert a normal distribution to the **standard normal distribution** by using its z-scores: $z = (x - \mu) / \sigma$._


```python
# example: probability of having the value of a standard normal distribution below z
from scipy.stats import norm
print('{:.1%}'.format(norm.cdf(x=0.7)))
```

    75.8%
    


```python
# example: z-score of a given probability
from scipy.stats import norm
print('{:.3}'.format(norm.ppf(q=0.95)))
```

    1.64
    


```python
# example: percentile of a student for a normal distribution of mean 75 and sd 7
from scipy.stats import norm
print('{:.1%}'.format(norm.cdf(x=87, loc=75, scale=7)))
```

    95.7%
    

___

## Statistics

### Statistical Inference

We usually do not have access to the entire **population** of a group. We draw our conclusions on a **sample** of this group: a subset of members that time and resources allow us to measure. This process is called **statistical inference** and is associated with a measure of **uncertainty**: how likely are these conclusions to be true.

We infer **parameters** from **statistics**:
+ a parameter is a characteristic of the population
+ a statistic is a characteristic of the sample
+ a variable is a characteristic of each member of the sample

### Sampling

A **reasonably sized random sample** (around 30 members) will almost always reflect the population. 

There are several ways to do sample a population:

+ [**Simple random sample**](https://en.wikipedia.org/wiki/Simple_random_sample) – each subject in the population has an equal chance of being selected. Some demographics might be missed.
+ [**Stratified random sample**](https://en.wikipedia.org/wiki/Stratified_sampling) – the population is divided into groups based on some characteristic (e.g. sex, geographic region). Then simple random sampling is done for each group based on its size in the actual population.
+ [**Cluster sample**](https://en.wikipedia.org/wiki/Cluster_sampling) – a random cluster of subjects is selected from the population (e.g. certain neighborhoods instead of the entire city).

Sampling must be probabilistic in order to make inference about the whole population. Otherwise, the inference can only be made about the sample itself.

There are several forms of **sampling bias**:
+ selection bias: not fully representative of the entire population
    + people who answer surveys
    + people from specific segments of the population (polling about health at fruit stand)
+ survivorship bias: population improving over time by having lesser members leave due to death
    + head injuries with metal helmets increasing vs cloth caps because less lethal
    + damage in WWII planes: not uniformally distributed in planes that came back, but only in non-critical areas

### Central Limit Theorem

A group of samples having the same size $n$ will have mean values **normally distributed** around the population mean $\mu$, regardless of the original distribution.

This normal distribution has:
+ the **same mean** $\mu$ as the population
+ a standard deviation called **standard error** equal to $\sigma / \sqrt(n)$, where $\sigma$ is the SD of the population

Because it is normally distributed, 95% of all sample means should fall within two standard errors of the actual population mean.

In other words: we can say with a 95% confidence level that the **population parameter** lies within a confidence interval of plus-or-minus two standard errors of the **sample statistic**.

___

## Hypothesis Testing

### Experimentation Design

Hypothesis testing is used to **make decisions about a population** using sample data. 

+ We start with a **null hypothesis** that we we asssume to be true.
+ We run an **experiment** to test this hypothesis.
+ Based on the experimental results, we can either **reject** or **fail to reject** this null hypothesis. 
+ If we reject it, we say that the data supports another, mutually exclusive, **alternate hypothesis**.

### Null and Alternate Hypothesis

Some example of **null hypothesis**: 
+ the average human weight is 60kg: average weight = 60kg
+ this prep course improves test scores: old scores $\geq$ new scores _(we cannot assume the claim is true, so we test the opposite)_

Given some sample statistic $\mu$ and the population parameter $\mu_0$, there are three possible **alternate hypotheses**:

1. two-sided test: $\mu \neq \mu_0$
2. left-tailed test: $\mu \lt \mu_0$
3. right-tailed test: $\mu \gt \mu_0$

The experiment compares the sample statistic to the $H_0$ parameter. We **reject the null hypothesis** if the probability of observing the experimental results is **very small** under its assumption. The cutoff probability is called the **level of significance** and is typically 5%.

### Types of Errors

There are four possible outcomes for our hypothesis testing, with two [**types of errors**](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors):

| Decision          | $$H_0$$ is True                      | $$H_0$$ is False                     |
|-------------------:|:---------------------------------:|:---------------------------------:|
| **Reject H0** | **Type I error**: False Positive   | Correct inference: True Positive |
| **Fail to reject H0** | Correct inference: True Negative | **Type II error**: False Negative |

<br>

The probabilities of making these two kinds of errors are related:
+ Decreasing the Type I error increased the probability of the Type II error.
+ If you want both to decrease, you have to increase the sample size.


<br>

When we **reject** the null hypothesis, there is still a very small likelihood that our sample belongs to the population but with extreme values: This probably is called the **Type I error** and is equal to the **significance level** $\alpha$. It falsely states that the alternate hypothesis is true, hence its name of **False Positive**.
  
<br>

The **Type II error** $\beta$ is the probability of incorrectly failing to reject a false $H_0$: **False Negative**.

### Statistical Power

[**Power**](https://en.wikipedia.org/wiki/Statistical_power) is the probability of correctly rejecting a false $H_0$. It is equal to $1 - \beta$. It is also called the **sensitivity**. High power means less chances of having false negatives.

_Note: A test with more statistical power needs less extreme values of the test statistic to be statistically significant._

Two key things impact statistical power:
+ the effect size: a large difference between groups is easier to detect
+ the sample size: it directly impacts the test statistic and the p-value

A typical formula to asses sample size is:

$$N = 16 \times \frac{\sigma^2}{\delta^2} \text{    where }\sigma\text{ is the variance of data and }\delta\text{ the minimum difference to detect}$$

The caveat is that you need to collect data for the sample of size $N$ calculated above before being able to draw conclusions.

___

## Selecting a test

Identifying wich test to use starts with two questions:
+ how many categories of population do you have (example: male vs female, three different trial groups)
+ what is the type of the variable to analyze: nominal, measurement or ranked

Note: 
+ Z-tests for mean serve the same purpose as ANOVA
+ Z-tests for proportions serve the same purpose as chi-square tests

Both tests compare a sample to a given population. Formally, the population SD needs to be known, but we can use a t-test with the sample standard deviation if not.

Additional information can be found [here](http://www.biostathandbook.com/).

### T-test & ANOVA

The variable to analyze is **measurement**: you want to compare the **mean** among categories.
+ **T-test** for roughly Gaussian-distributed data; the test statistic follows a **t-distribution**. 
    + only one measurement variable
    + only one or two samples _(note: for one-sample, you will test against a number that needs to be known beforehand)_
    + both one-tail and two-tailed alternate hypothesis are possible
+ **ANOVA** for roughly Gaussian-distributed data; the test statistic follows an **F-distribution**. 
    + one-way ANOVA when only one measurement variable
    + N-ways otherwise
    + The only alternate hypothesis is that the different categories have different means.
+ **non-parametric** tests otherwise _(typically with less assumptions so less statistical power)_

### Chi-Square

The variable to analyze is **nominal**: you want to compare the **frequencies** among categories.
+ **Chi-Square** test; the test statistic follows a **Chi-Square distribution**.
    + One-way when the nominal variable only has one value (ex: repartition of patient discharges per day of the week)
    + Two-way otherwise: test of independance or conformity (ex: comparing proportion among population categories)
    + The only alternate hypothesis is that the different categories have different frequencies.
+ **Fisher's exact test** if the sample size is small

### Kruskal–Wallis

The variable to analyze is **ranked**: you want to compare the **ranks** among categories.
+ **Kruskal–Wallis** test

### Bonferroni Corrections

The chance of capturing rare event increases when testing multiple hypothesis. It means the likelihood of incorrectly rejecting a null hypothesis (false positive) increases. 

The Bonferroni correction rejects the null hypothesis for each $p_{i} \leq \frac {\alpha}{m}$. This ensures the [Family Wise Error Rate](https://en.wikipedia.org/wiki/Family-wise_error_rate) stays below the significance level $\alpha$. More information can be found [here](https://stats.stackexchange.com/questions/153122/bonferroni-correction-for-post-hoc-analysis-in-anova-regression).

It is useful for post-hoc tests after performing one-way ANOVA or Chi-Square tests that reject the null hypothesis. When comparing $N$ multiple groups, we can either do:
+ pairwise tesing. In that case, $m$ will be ${N \choose 2}$.
+ one vs the rest. In that case, $m$ will be $N$.

___

## Assessing a test

### P-Value

The [p-value](https://en.wikipedia.org/wiki/P-value) is the probability that our sample produces such a statistic or one more extreme under $H_0$. A low p-value means that it is unlikely that the $H_0$ probability distribution actually describe the population: we reject the null hypothesis.

+ $P\leq\alpha$: we **reject** the null hypothesis. The observed effect is **statistically significant**
+ $P\gt\alpha$: we **fail to reject** the null hypothesis. The observed effect is **not statistically significant**

### P-Values vs Distribution Tails

Hypothesis testing leverages the **Central Limit Theorem**, as seen in the examples below. The p-value being smaller than $\alpha$ would mean that the sample statistic under $H_0$ is in the blue areas of the **sampling distribution of sample statistic**, depending on the alternate hypothesis:

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/21a0a7a855f51f6426dfbf6115b872161490032937519"/>

### Z-Scores

We can use two factors to assess the probability of observing the experimental results under the null hypothesis:
+ The [**Z-score**](https://en.wikipedia.org/wiki/Standard_score) represents the number of standard deviations an observation is from the mean.
+ The sampling distribution of sample statistic is centered around the population parameter and has a standard error linked to the population variance. 

It means that we can calculate the z-score of our sample statistic to calculate its p-value.

___

## Z-tests

### Mean Tests

A **mean test** look for a specific value, typically the average of a population parameter. Its null hypothesis states that $\mu = \mu_0$.

+ According to the CLT, the mean of our sample is part of a normal distribution centered around its population mean
+ The null hypothesis states that the sample belongs to the initial population

It means that we can calculate the position of the sample mean in the sampling distribution that follows $H_0$, provided we know the population standard deviation (see formula in code). 

We calculate its p-value based on the alternate hypothesis to draw our conclusions.


```python
# example: test if a website redesign improved load time. H0: old_load >= new_load, alpha = 0.01
# old_load_mean = 3.125, old_load_sd = 0.700
# new_load_mean = 2.875, new_load_sample_size = 40
# the sample mean of new_load_mean under H0 is in the 1.2% percentile, outside of our cutoff area: we fail to reject H0

import numpy as np
from scipy.stats import norm
z_score = (2.875 - 3.125) / (0.700 / np.sqrt(40))
print('p-value: {:.1%} > 1% cutoff'.format(norm.cdf(x=z_score)))
```

    p-value: 1.2% > 1% cutoff
    

### Proportion tests

A **proportion test** looks for a share of a population that have a specific trait. We assume that the population follows a binomial distribution with a probability $p_0$ of having a given trait, and the sample has a measured probability $\hat{p}$.

It also leverages the CLT, with the following tweaks:
+ proportions are equal to $\hat{p} - p_0$
+ population variance equals $p_0 \times(1 - p_0)$ according to the binomial distribution


```python
# example: test if most customers of a website are teenagers. H0: teen_proportion <= 0.5 
# sample_teen_proportion = 0.58, sample_size = 400
# H0_teen_proportion = 0.5, HO_variance = sample_size * H0_teen_proportion * (1 - H0_teen_proportion)

# the sample_teen_proportion under H0 is in the 99.9% percentile, inside our cutoff area: we reject H0

import numpy as np
from scipy.stats import norm
z_score = (0.58 - 0.5) / (np.sqrt(0.5 * (1 - 0.5)) / np.sqrt(400))
print('p-value: {:.1%} < 5% cutoff'.format(norm.sf(x=z_score))) # right-tail so survival function = 1 - cdf
```

    p-value: 0.1% < 5% cutoff
    

___

## T-tests

### Assumptions

The CLT and Z-scores assume **we know the population standard deviation**. Using them does not work when:

+ **$\sigma$ is unknown**
+ the sample size **$n$ is small**

We can substitute the normal distribution with the [Student’s t distribution](https://en.wikipedia.org/wiki/Student's_t-distribution) to represent the **sampling distribution** of the sample statistic when:

+ the sample size **is large** (30+ observations), OR
+ the **population** is roughly **normal** (very small samples)

The tails of the Student Distribution are **thicker than normal** to reflect the **additional uncertainty** introduced by using the sample standard deviation. They get closer to the normal distribution as the degrees of freedom increase (ie. when the sample size increases).

_Note: when the sample size is large (30+ observations), the Student Distribution becomes extremely close to the normal distribution._

Notes:
+ If the sample size is very small, we can use normal probability plots to check whether the sample may come from a normal distribution.
+ If the t-distribution cannot be used, we can use more robust procedures like the one-sample [**Wilcoxon procedure**](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test).

### T-statistic

A t-statistic will be larger (i.e. less likely to happen by chance) if:
+ the compared statistic values are very different 
+ the pooled standard deviation is small, ie. the compared distributions do not overlap much
+ the samples are large

### Types of t-tests

There are three main types of t-tests:
+ **one-sample** t-test: check if a sample is part of a population
+ **dependent paired-samples** t-test: check if the statistic of the same sample evolve over time
+ **independent two-samples** t-test: check if two samples are part of the same population
    + same sample size, equal variance
    + different sample size, equal variance
    + different variance (Welch's t-test)

### One-sample


```python
# we want to check if the following sample belongs to a population of mean 120
from scipy import stats

sample = [120.6,116.4,117.2,118.1,114.1,116.9,113.3,121.1,116.9,117.0]
stats.ttest_1samp(sample, 120)
```




    Ttest_1sampResult(statistic=-3.6923618691956284, pvalue=0.004979126870063981)



### Independent two-samples

We want to compare the productivity of two plants over the same 10 days:


```python
from scipy import stats

# number of cars produced in the two plants over the same 10 days
cars_plant1 = [1184, 1203, 1219, 1238, 1243, 1204, 1269, 1256, 1156, 1248]
cars_plant2 = [1136, 1178, 1212, 1193, 1226, 1154, 1230, 1222, 1161, 1148]

stats.ttest_ind(cars_plant1, cars_plant2, equal_var=False) # t > 0 and p/2 > 0.05 so plant1 performs significantly better
```




    Ttest_indResult(statistic=2.2795770510504845, pvalue=0.03504506467283038)



___

## ANOVA

ANOVA allows us to:
+ check if **more than two samples** belong to the same population.
+ check if two samples come from populations that have the **same variance**.

### Assumptions

The ANOVA is mathematically considered a [generalized linear model (GLM)](https://pythonfordatascience.org/anova-python/). It means that its assumptions are the same as for linear regression:

+ Normality
+ Homogeneity of variance
+ Independent observations

If group sizes are equal, the F-statistic is robust to violations of normality and homogeneity of variance. If these assumptions are not met, we can use either the Kruskal-Wallis H-test or the Welch’s ANOVA.

### F-distribution

The [F-Distribution](https://www.geo.fu-berlin.de/en/v/soga/Basics-of-statistics/Continous-Random-Variables/F-Distribution/index.html) has two numbers of degrees of freedom: the denominator (sample size) and numerator (number of samples). 


```python
# example of F-distribution
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt

# degrees of freedom
dfn, dfd = 30, 8

# 100 x points between the first and 99th percentile of the f-distribution & corresponding f values
x = np.linspace(f.ppf(0.01, dfn, dfd), f.ppf(0.99, dfn, dfd), 100)
y = f.pdf(x, dfn, dfd)

# plot
fig, ax = plt.subplots(1, 1)
ax.plot(x, f.pdf(x, dfn, dfd), 'r-', lw=2, alpha=0.6, label='f pdf')
plt.show()
```


![png](output_94_0.png)


### F-value

ANOVA compares two types of variance:

+ between groups: how far group means stray from the total mean
+ within groups: how far individual values stray from their respective group mean

The **F-value** is the variance between groups divided by the variance within groups, where:

+ the variance between groups equals the sum of squares group divided by the degrees of freedom (groups)
+ the variance within groups equals the sum of squares errors divided by the degrees of freedom (error)

The groups belong to the **same population** if the **variance between groups** (numerator) is **small** compared to the **variance within groups** (denominator).

### One-way ANOVA


```python
# example: number of days each customer took to pay an invoice based on a percentage of discount if early payment
disc_0p = [14, 11, 18, 16, 21]
disc_1p = [21, 15, 23, 10, 16]
disc_2p = [11, 16,  9, 14, 10]

stats.f_oneway(disc_0p, disc_1p, disc_2p) # p-value > 0.05, the discounts make no significant difference
```




    F_onewayResult(statistic=2.121212121212121, pvalue=0.16262315311926887)



### Two-way ANOVA

The **two-way ANOVA** is an extension of the one-way ANOVA to test **two independant variables** at the same time, taking interactions between these variables into account. 

_Note: this can be further generalized to N-way ANOVA._


```python
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels as sm

# data - same as before, but checking if the amount has an impact
df = pd.DataFrame({
    'discount': ['2p','2p','2p','2p','2p','1p','1p','1p','1p','1p','0p','0p','0p','0p','0p'],
    'amount': [50,100,150,200,250,50,100,150,200,250,50,100,150,200,250],
    'days': [16,14,11,10,9,23,21,16,15,10,21,16,18,14,11]
})


# fit without interaction factor
model = ols('days ~ C(discount) + C(amount)', df).fit()

# discount has now become significant
sm.api.stats.anova_lm(model, typ=2)

# model.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C(discount)</td>
      <td>70.0</td>
      <td>2.0</td>
      <td>11.666667</td>
      <td>0.004249</td>
    </tr>
    <tr>
      <td>C(amount)</td>
      <td>174.0</td>
      <td>4.0</td>
      <td>14.500000</td>
      <td>0.000975</td>
    </tr>
    <tr>
      <td>Residual</td>
      <td>24.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Note on `model.summary()`:
+ Durban-Watson detects the presence of autocorrelation
+ Jarque-Bera tests the assumption of normality
+ Omnibus tests the assumption of homogeneity of variance
+ Condition Number assess multicollinearity (should be < 20)


```python
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels as sm

# data - three fertilizers, warm vs cold, size of plant
df = pd.DataFrame({
    'fertilizer': ['A','A','A','A','A','A','B','B','B','B','B','B','C','C','C','C','C','C'],
    'temperature': ['W','W','W','C','C','C','W','W','W','C','C','C','W','W','W','C','C','C'],
    'size': [13,14,12,16,18,17,21,19,17,14,11,14,18,15,15,15,13,8]
})


# fit with interaction factor
model = ols('size ~ C(fertilizer) * C(temperature)', df).fit()

# discount has now become significant
sm.api.stats.anova_lm(model, typ=2)

# model.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>C(fertilizer)</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>1.44</td>
      <td>0.275087</td>
    </tr>
    <tr>
      <td>C(temperature)</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>4.32</td>
      <td>0.059786</td>
    </tr>
    <tr>
      <td>C(fertilizer):C(temperature)</td>
      <td>84.0</td>
      <td>2.0</td>
      <td>10.08</td>
      <td>0.002699</td>
    </tr>
    <tr>
      <td>Residual</td>
      <td>50.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



___

## Chi-Square Analysis 

The [Chi-Square Analysis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) determines the **probability of an observed frequency of events** given an expected frequency. If we get heads ten times in a row, how likely is it to happen if we assume the coin to be fair?

$\chi^2 = \sum (O - E)^2 / E$ for each possible outcome of an experiment

_Note: the expected value is not a probability._

### Chi-Square Distribution

The Chi-Square Distribution depends on its degrees of freedom, which are the number of possible outcomes minus one.


```python
# example of Chi-Square-distribution
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# plot
fig, ax = plt.subplots(1, 1)

# degrees of freedom
for df in [3,4,5,6]:

    # 100 x points between the first and 99th percentile of the f-distribution & corresponding f values
    x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
    y = chi2.pdf(x, df)

    ax.plot(x, y, lw=2, alpha=0.6)

plt.show()
```


![png](output_108_0.png)


### One-way Chi-Square

We get 12 heads out of 18 coin tosses. Is the coin fair?


```python
# example: we get 12 heads out of 18 coin tosses. Is the coin fair?
from scipy.stats import chisquare
chisquare([6, 12], f_exp=[9,9]) # p-value is > 5%: we fail to reject the null hypothesis
```




    Power_divergenceResult(statistic=2.0, pvalue=0.15729920705028105)



 

Our company has six server that should fail at the same rate. Is it true?

In the example below, we have 240 failures. If the null hypothesis is true, the probability of failure should be the same for all the six servers: 1/6 or 40 failures per server. Let's test this: 


```python
import numpy as np
from scipy.stats import chisquare

obs_failures = [46,36,52,26,42,38]
mean_failure = np.mean(obs_failures)

chisquare(obs_failures, f_exp=mean_failure) # p-value is > 5%: we fail to reject the null hypothesis
```




    Power_divergenceResult(statistic=10.0, pvalue=0.07523524614651217)



Are most customers of a website teenagers?

In the example below, we have a sample of 400 visitors, 58% of which are teenagers. If the null hypothesis is true, the probability of having teenagers is 50% or less.


```python
import numpy as np
from scipy.stats import chisquare

obs_values = [232, 400 - 232]
exp_values = [200, 200]

chisquare(obs_values, f_exp=exp_values) # p-value is < 5%: we reject the null hypothesis
```




    Power_divergenceResult(statistic=10.24, pvalue=0.0013742758758316976)



###  Two-way Chi-Square

Suppose there is a city of 1,000,000 residents with four neighborhoods: $A$, $B$, $C$, and $D$. A random sample of 650 residents of the city is taken and their occupation is recorded as "white collar", "blue collar", or "no collar". 

|                | $A$   | $B$   | $C$   | $D$   | **Total** |
|----------------|-------|-------|-------|-------|-----------|
|White Collar    |  90   |  60   | 104   |  95   |  **349**  |
|Blue Collar     |  30   |  50   |  51   |  20   |  **151**  |
|No Collar       |  30   |  40   |  45   |  35   |  **150**  |
|**Total**       |**150**|**150**|**200**|**150**|  **650**  |


The null hypothesis is that each person's neighborhood of residence is independent of the person's occupational classification. By the assumption of independence under the hypothesis we should "expect" the number of white-collar workers in neighborhood $A$ to be:

$WC_A = 150\times\frac{349}{650} \approx 80.54$

So the contribution of this cell to $\chi^2$ is $\frac{(90 - 80.54)^2}{80.54} \approx 1.11$

The sum of these quantities over all of the cells is the test statistic; in this case, $\approx 24.6$.  Under the null hypothesis, this sum has approximately a chi-squared distribution whose number of degrees of freedom are:

$(\text{number of rows}-1)(\text{number of columns}-1) = (3-1)(4-1) = 6$

If the test statistic is improbably large according to that chi-squared distribution, then one rejects the null hypothesis of 'independence'. In this example, the neighborhood and occupation are linked.


```python
import pandas as pd
import scipy.stats as stats

df = pd.DataFrame({
    'A': [ 90,30,30], 
    'B': [ 60,50,40], 
    'C': [104,51,45], 
    'D': [ 95,20,35]}, index=['White Collar', 'Blue Collar', 'No Collar']
)

display(df)

chi2, p, _, _ = stats.chi2_contingency(df)
print('Chi-Square test: {:.3} - p-value: {:.3}'.format(chi2, p)) # we reject the null hypothesis of independence
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>White Collar</td>
      <td>90</td>
      <td>60</td>
      <td>104</td>
      <td>95</td>
    </tr>
    <tr>
      <td>Blue Collar</td>
      <td>30</td>
      <td>50</td>
      <td>51</td>
      <td>20</td>
    </tr>
    <tr>
      <td>No Collar</td>
      <td>30</td>
      <td>40</td>
      <td>45</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>


    Chi-Square test: 24.6 - p-value: 0.00041
    

Another example: is the proportion of kids taking swimming lessons depend on their ethnicity:
+ 247 Black kids. 36.8% take swimming lessons
+ 308 Hispanic kids. 38.9% take swimming lessons

The null hypothesis is that the proportion of kids taking swimming lessons does not depend on their ethnicity.


```python
# contingency matrix
df = pd.DataFrame({'black': [91, 156], 'hisp': [120, 188]}, index=['Swim', 'No Swim'])
display(df)

chi2, p, _, _ = stats.chi2_contingency(df)
print('Chi-Square test: {:.3} - p-value: {:.3}'.format(chi2, p)) # we fail to reject the null hypothesis of independence
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>black</th>
      <th>hisp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swim</td>
      <td>91</td>
      <td>120</td>
    </tr>
    <tr>
      <td>No Swim</td>
      <td>156</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>


    Chi-Square test: 0.179 - p-value: 0.672
    

___

## Correlation between variables

Correlation is the measure of dependance between two variables; It typically indicates their linear relationship, but more broadly measures how in sync they vary. This is expressed by their **covariance**. 

A more common measure is the **[Pearson product-moment correlation coefficient](https://en.wikipedia.org/wiki/Correlation_and_dependence#Pearson's_product-moment_coefficient)**, built on top of the covariance. It's akin to the standard variation vs the variance for bivariate data and represents how far the relationship is from the line of best fit.

The correlation coefficient divides the covariance by the product of the standard deviations. This normalizes the covariance into a unit-less variable whose values are between -1 and +1.

The line of best fit has a slope equal to the Pearson coefficient multiplied by SDy / SDx.

___

## Linear Regression

+ only incude variables that are correlated to the outcome.
+ check for collinearity.

### Independent two-samples - proportion

Is there a significant difference in the proportion of Black vs Hispanic children taking swimming lessons?


___

## Appendix - Formulas

For $n$ independant Bernoulli trials having a probability of success $p$, the probability of observing $k$ successes is:

$$P(k)={n \choose k} p^k (1 - p)^{n-k}$$

Note: ${n \choose k}$ is called the **Bernoulli coefficient**.

The probability of observing $k$ events in an interval is:

$$P(k \text{ events in interval}) = \frac{\lambda^k e^{-\lambda}}{k!}$$

The p-value calculation will vary slightly based on the alternate hypothesis, if $x$ is the measured sample statistic and $X$ the sampling distribution of the sample statistic:

+ $2\min\{\Pr(X\leq x|H),\Pr(X\geq x|H_0)\}$ for double tail events
+ $\Pr(X \leq x | H_0)$ for left tail events
+ $\Pr(X \geq x | H_0)$ for right tail events


```python

```
