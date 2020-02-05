## Introduction

> The goal of statistical inference is to make **generalizations about the population** when **only a sample is available**.

We usually do not have access to an entire **population**, but only to a subset of its members that time and resources allow us to measure. We use this **sample data** to draw **probabilistic conclusions** about the population; this process is called [**statistical inference**](https://www.encyclopediaofmath.org/index.php/Statistical_inference).

The population variable can take many values that are more or less likely to happen. This **probability distribution** has a numerical feature of interest called a **parameter**. The data we collect is **randomly generated** from this probability distribution. 

We draw our conclusions about the parameter from the sample **statistic**. Three important limitations:
+ Because a sample is only part of the population, the numerical value of the statistic will not be the exact value of the parameter.
+ The observed value of the statistic depends on the particular sample selected.
+ Some variability in the values of a statistic, over different samples, is unavoidable.

___


## Sampling

### Methods

A **reasonably sized random sample** (around 30 members) will almost always reflect the population. 

There are several ways to sample a population:

+ [**Simple random sample**](https://en.wikipedia.org/wiki/Simple_random_sample) – each subject in the population has an equal chance of being selected. Some demographics might be missed.
+ [**Stratified random sample**](https://en.wikipedia.org/wiki/Stratified_sampling) – the population is divided into groups based on some characteristic (e.g. sex, geographic region). Then simple random sampling is done for each group based on its size in the actual population.
+ [**Cluster sample**](https://en.wikipedia.org/wiki/Cluster_sampling) – a random cluster of subjects is selected from the population (e.g. certain neighborhoods instead of the entire city).

### Representativity

Sampling must be probabilistic in order to make inference about the whole population. Otherwise, the inference can only be made about the sample itself.

There are several forms of [**sampling bias**](https://en.wikipedia.org/wiki/Sampling_bias):
+ selection bias: not fully representative of the entire population
    + people who answer surveys
    + people from specific segments of the population (polling about health at fruit stand)
+ survivorship bias: population improving over time by having lesser members leave due to death
    + head injuries with metal helmets increasing vs cloth caps because less lethal
    + damage in WWII planes: not uniformally distributed in planes that came back, but only in non-critical areas
    
_Note: other [**criteria**](https://en.wikipedia.org/wiki/Selection_bias) can also impact the representativity of our sample._


___


## Inference Problems

### Estimation of parameter

The estimation can either be:
+ **point estimate**: a particular value that best approximates some parameter.
+ **interval estimate**: interval of plausible values for the parameter.

This can also include prediction intervals for future observations.

### Hypothesis Testing

A test of hypotheses provides a yes/no answer as to whether the parameter lies in a specified region of values.

### Limitations of Statistical Inference

Because statistical inferences are based on a sample, they will sometimes be in error. The actual value of the parameter is unknown, so a test of hypotheses may yield the **wrong yes/no answer** and the interval of plausible values **may not contain the true value** of the parameter.

_Note: Large samples approximate the population distribution more closely, so we can be more confident in their conclusion._


___


## Frequentist  vs Bayesian Paradigms

Both paradigms are based on the likelihood but their frameworks are entirely different.

### Frequentist Paradigm

In the frequentist paradigm, the **parameter** is **set but unknown**. 

Due to the random nature of sampling, some samples are not representative of the population. It means that a small proportion of samples, typically noted $\alpha$, will produce incorrect inferences. This probability of errors can be controlled to build **(1 - $\alpha$) Percent Confidence Intervals**. 

This means that for (1 - $\alpha$) percents of all samples, the calculated interval will actually include the parameter value.

_Note: this is not a probability. The interval either includes the parameter value or it doesn't._

### Bayesian Paradigm

In the Bayesian paradigm, the **parameter** is a **random variable**. 

It is assigned a **prior distribution** based on already available (prior) data. This distribution is updated by the likelihood of the sample values to obtain its **posterior distribution**. From it, both **point estimate** and **region of highest posterior density** (or credible intervals) can be derived.

### Considerations

A rigorous approach to frequentist statistics assume that the conditions of the experiment are well-defined even before any dat a is actually collected. 

Baysesian statistics, on the other hand, make no such assumptions. They are especially useful when new data is constently collected: our beliefs are constantly updated, older data being used as prior to the new data that comes in. 

___


## Parameter estimation

The goal of [parameter estimation](https://en.wikipedia.org/wiki/Point_estimation) is to find the best approximation of the population parameter given our sample data; there are many possibilities. We'll focus on the following two: 

+ the Maximum Likelihood Estimate, or MLE, in frequentist inference.
+ the Maximum A Posteriori, or MAP, in Bayesian inference.

### Maximum Likelihood Estimate

The [**Maximum Likelihood Estimator**](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is the value of the parameter space (i.e. the set of all values the parameter can take) that is the **most likely** to have **generated our sample**. As the sample size increases, the MLE converges towards the true value of the population parameter.

+ for discrete distributions, the MLE of the probability of success is equal to successes / total trials.
+ for continuous distributions:
    + the MLE of the population mean is the sample mean. 
    + the MLE of the population variance is the sample variance.
    
    
_Note1: the sample variance needs to be [slightly adjusted](https://en.wikipedia.org/wiki/Bessel%27s_correction) to become [unbiased](https://dawenl.github.io/files/mle_biased.pdf)._

_Note2: in more complex problems, the MLE can only be found via numerical optimization._

### Maximum A Posteriori

The Maximum A Posteriori is [very similar to the MLE](https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/), but for the posterior distribution of the population parameter. It applies weigth coming from the prior to the likelihood of the new sample data.

_Note1: for constant priors, the MAP is equal to the MLE._

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

Given some sample statistic $\mu$ and the population parameter $\mu_0$, there are three possible **alternate hypotheses**:

| Left-tailed  | Two-sided     | Right-tailed    |
|-----------------:|:-----------------:|:-------------------:|
| $\mu \lt \mu_0$ | $\mu \neq \mu_0$   | $\mu \gt \mu_0$     |


<br>

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

Given the variance of data $\sigma$ and the minimum difference to detect $\delta$,  a typical formula to asses sample size is:

$$N = 16 \times \frac{\sigma^2}{\delta^2}$$

The caveat is that you need to collect data for the sample of size $N$ calculated above before being able to draw conclusions.

___

## Central Limit Theorem

### Definition

A group of samples having the same size $N$ will have mean values **normally distributed** around the population mean $\mu$, regardless of the original distribution.

This normal distribution has:
+ the **same mean** $\mu$ as the population
+ a standard deviation called **standard error** equal to $\sigma / \sqrt(n)$, where $\sigma$ is the SD of the population

### Confidence Intervals

Because the sampling distribution of sample statistic is **normally distributed**, 95% of all sample means fall within two standard errors of the actual population mean. In other words: we can say with a 95% confidence level that the **population parameter** lies within a confidence interval of plus-or-minus two standard errors of the **sample statistic**.

### Hypothesis Testing

Hypothesis testing also leverages the **Central Limit Theorem**. If we assume that the population parameter holds a specific value under $H_0$, we can use the CLT to measure the probability that our sample produces such a statistic or one more extreme under $H_0$. This probability is called the [p-value](https://en.wikipedia.org/wiki/P-value).

A low p-value means that it is unlikely that the $H_0$ probability distribution actually describe the population: we reject the null hypothesis.

+ $P\leq\alpha$: we **reject** the null hypothesis. The observed effect is **statistically significant**
+ $P\gt\alpha$: we **fail to reject** the null hypothesis. The observed effect is **not statistically significant**

The p-value being smaller than $\alpha$ would mean that the sample statistic under $H_0$ is in the blue areas of the **sampling distribution of sample statistic**, depending on the alternate hypothesis:

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/21a0a7a855f51f6426dfbf6115b872161490032937519"/>

_Note: for two-tailed tests, we use $\alpha/2$ for each tail. This ensures the total probability of extreme values is $\alpha$._

### Z-Scores

We can use two factors to assess the probability of observing the experimental results under the null hypothesis:
+ The [**Z-score**](https://en.wikipedia.org/wiki/Standard_score) represents the number of standard deviations an observation is from the mean.
+ The sampling distribution of sample statistic is centered around the population parameter and has a standard error linked to the population variance. 

It means that we can calculate the z-score of our sample statistic to calculate its p-value.

___


## Further Reads

A few interesting Wikipedia articles:

Generalities
+ https://en.wikipedia.org/wiki/Sampling_distribution
+ https://en.wikipedia.org/wiki/Statistical_hypothesis_testing 

Probabilities
+ https://en.wikipedia.org/wiki/Probability_interpretations
+ https://en.wikipedia.org/wiki/Frequentist_probability
+ https://en.wikipedia.org/wiki/Bayesian_probability

Inference paradigms:
+ https://en.wikipedia.org/wiki/Frequentist_inference
+ https://en.wikipedia.org/wiki/Bayesian_inference
+ https://en.wikipedia.org/wiki/Lindley%27s_paradox
+ https://www.stat.berkeley.edu/~stark/Preprints/611.pdf
