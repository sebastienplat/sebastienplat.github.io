## Introduction

> The goal of [**statistical inference**](https://www.encyclopediaofmath.org/index.php/Statistical_inference) is to make **generalizations about the population** when **only a sample is available**.

## Sampling

We want to study the **probability distribution** of a population variable: how likely its possible values are to happen. The data we collect must be **randomly generated** from this probability distribution in order to make inference about **the whole population**. 

### Methods

There are several ways to sample a population:

+ [**Simple random sample**](https://en.wikipedia.org/wiki/Simple_random_sample) – each subject in the population has an equal chance of being selected. Some demographics might be missed.
+ [**Stratified random sample**](https://en.wikipedia.org/wiki/Stratified_sampling) – the population is divided into groups based on some characteristic (e.g. sex, geographic region). Then simple random sampling is done for each group based on its size in the actual population.
+ [**Cluster sample**](https://en.wikipedia.org/wiki/Cluster_sampling) – a random cluster of subjects is selected from the population (e.g. certain neighborhoods instead of the entire city).

### Bias

There are several forms of [**sampling bias**](https://en.wikipedia.org/wiki/Sampling_bias) that can lead to incorrect inference:
+ selection bias: not fully representative of the entire population.
    + people who answer surveys.
    + people from specific segments of the population (polling about health at fruit stand).
+ survivorship bias: population improving over time by having lesser members leave due to death.
    + head injuries with metal helmets increasing vs cloth caps because less lethal.
    + damage in WWII planes: not uniformally distributed in planes that came back, but only in non-critical areas.
    
_Note: other [**criteria**](https://en.wikipedia.org/wiki/Selection_bias) can also impact the representativity of our sample._

### Representativity

Due to the random nature of sampling, some samples are **not representative** of the population and will produce incorrect inference. This uncertainty is reflected in the **confidence level** of statistical conclusions:
+ a small proportion of samples, typically noted $\alpha$, will produce incorrect inferences.
+ for 1 - $\alpha$ percents of all samples, the conclusions will be correct.
+ the confidence level is therefore expressed as 1 - $\alpha$.

_Note: 0.01 and 0.05 are the most common values of $\alpha$. This translates to 99% and 95% confidence intervals._

___

## Probability Distribution

### Point Estimate

It is often interesting to summarize the **probability distribution** with a single numerical feature of interest: the population **parameter**. We draw our conclusions about the parameter from the sample **statistic**. 

A few important limitations:
+ a sample is only part of the population; the numerical value of its statistic will not be the exact value of the parameter.
+ the observed value of the statistic depends on the selected sample.
+ some variability in the values of a statistic, over different samples, is unavoidable.

The [**Maximum Likelihood Estimator**](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is the value of the parameter space (i.e. the set of all values the parameter can take) that is the **most likely** to have **generated our sample**. As the sample size increases, the MLE converges towards the true value of the population parameter.

+ for discrete distributions, the MLE of the probability of success is equal to successes / total trials.
+ for continuous distributions:
    + the MLE of the population mean is the sample mean. 
    + the MLE of the population variance is the sample variance.
    
    
_Note1: the sample variance needs to be [slightly adjusted](https://en.wikipedia.org/wiki/Bessel%27s_correction) to become [unbiased](https://dawenl.github.io/files/mle_biased.pdf)._

_Note2: in more complex problems, the MLE can only be found via numerical optimization._

___

## Hypothesis Testing

### Experimental Design

Hypothesis testing is used to **make decisions about a population** using sample data. 

+ We start with a **null hypothesis $H_0$** that we we asssume to be true:
    + the sample parameter is equal to a given value.
    + samples with different characteristics are drawn from the same population.
+ We run an **experiment** to test this hypothesis:
    + **collect data** from a sample of predetermined size _(see [Statistical Power](#Statistical-Power) below)_.
    + perform the appropriate **statistical test**.
+ Based on the experimental results, we can either **reject** or **fail to reject** this null hypothesis. 
+ If we reject it, we say that the data supports another, mutually exclusive, **alternate hypothesis**.


### P-Value

We **reject the null hypothesis** if the probability of observing the experimental results, called the **p-value**, is **very small** under its assumption. The cutoff probability is called the **level of significance $\alpha$** and is typically 5%. 

More specifically, we measure the probability that our sample(s) produce such a test statistic or one more extreme under the $H_0$ probability distribution. A low p-value means that $H_0$ is unlikely to actually describe the population: we reject the null hypothesis.

+ $P\leq\alpha$: we **reject** the null hypothesis. The observed effect is **statistically significant**.
+ $P\gt\alpha$: we **fail to reject** the null hypothesis. The observed effect is **not statistically significant**.


### Types of Errors

There are four possible outcomes for our hypothesis testing, with two [**types of errors**](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors):

| Decision          | $$H_0$$ is True                      | $$H_0$$ is False                     |
|-------------------:|:---------------------------------:|:---------------------------------:|
| **Reject H0** | **Type I error**: False Positive   | Correct inference: True Positive |
| **Fail to reject H0** | Correct inference: True Negative | **Type II error**: False Negative |

<br>

The **Type I error** is the probability of incorrecly rejecting the null hypothesis when the sample belongs to the population but with extreme values; this probability is equal to the **level of significance** $\alpha$. It is also called **False Positive**: falsely stating that the alternate hypothesis is true.

The **Type II error** $\beta$ is the probability of incorrectly failing to reject a null hypothesis; it is also called **False Negative**.


_Note: The probabilities of making these two kinds of errors are related. Decreasing the Type I error increases the probability of the Type II error._


### Statistical Power

[**Power**](https://en.wikipedia.org/wiki/Statistical_power), also called the **sensitivity**, is the probability of correctly rejecting a false $H_0$; It is equal to $1 - \beta$.

Two key things impact statistical power:
+ the **effect size**: a large difference between groups is easier to detect.
+ the **sample size**: it directly impacts the test statistic and the p-value.

Given the variance of data $\sigma$ and the minimum difference to detect $\delta$, a typical formula to assess [sample size](https://en.wikipedia.org/wiki/Sample_size_determination) is:

$$N = (z_\alpha + z_\beta)^2 \times \frac{\sigma^2}{\delta^2}$$

Where $z_\alpha$ and $z_\beta$ are the z-score of $\alpha$ and $\beta$, respectively. 

___

## Choosing a Test

Choosing a [statistical test](http://www.biostathandbook.com/) depends on:
+ what hypothesis is tested.
+ the type of the variable of interest & its probability distribution.


![png](../../assets/images/posts/statistical-inference/stat_tests_overview.png)

_Note: relationship modelling will be covered in another article._

### Population Inference

We can infer the value of the **population parameter** based on the sample statistics. Which parameter represents the population the best depends on the probability distribution.

![png](../../assets/images/posts/statistical-inference/stat_tests_inference.png)

### Difference Between Samples

Comparing samples aims to determine if some characteristics of the population have an impact on the variable of interest. More specifically, we check if different values of some **categorical variable(s)** lead to **different probability distributions** for the variable of interest.


![png](../../assets/images/posts/statistical-inference/stat_tests_diff_between_samples.png)

### Correlation Coefficients

A **correlation coefficient** quantifies the **goodness of fit** between **two continuous or ordinal variables**. 

![png](../../assets/images/posts/statistical-inference/stat_tests_correlation.png)

___

## Assumptions of Parametric Tests

Both t-tests and ANOVA compare means between samples. They require specific assumptions for their conclusions to be statistially sound.

### T-Tests

In its most common form, a t-test **compare means**.

+ one-sample null hypothesis: the mean of a population has a specific value.
+ two-sample null hypothesis: the means of two populations are equal. 


T-tests make the following **assumptions**:
+ the sample **mean(s)** follow a **normal distribution** (this is always the case for large samples under the CLT).
+ the sample **variance(s)** follow a **$\chi^2$ distribution** (this is always the case for normally distributed data).

In practice, t-tests can be used when:
+ the sample size **is large** (30+ observations), OR
+ the **population** is roughly **normal** (very small samples - use normal probability plots to assess normality).


### ANOVA

In its most common form, an ANalysis Of VAriance ([ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance)) **compare means**.

+ one-way ANOVA null hypothesis: the means of three or more populations are equal _(see example [here](https://en.wikipedia.org/wiki/One-way_analysis_of_variance#Example))._
+ repeated measures ANOVA null hypothesis: the average difference between in-sample values is null.


ANOVA is mathematically a [generalized linear model (GLM)](https://pythonfordatascience.org/anova-python/), where the factors of all the categorical variables have been one-encoded. In particular, factorial ANOVA include interaction terms between categorical factors and should therefore be interpreted like traditional linear models.

ANOVA being a GLM, assumptions are the same as for linear regression:

+ Normality
+ Homogeneity of variance
+ Independent observations

_Note: If group sizes are equal, the F-statistic is robust to violations of normality and homogeneity of variance._

### Non-Parametric Alternatives

**Non-parametric** tests should be used when:
+ the **assumptions are not met**.
+ the **mean** is **not the most appropriate** parameter to describe the population.


___

## Appendix - Central Limit Theorem

### Definition

A group of samples having the same size $N$ will have mean values **normally distributed** around the population mean $\mu$, regardless of the original distribution. This normal distribution has:
+ the **same mean** $\mu$ as the population
+ a standard deviation called **standard error** equal to $\sigma / \sqrt(n)$, where $\sigma$ is the SD of the population

### Confidence Intervals

Because the sampling distribution of sample statistic is **normally distributed**, 95% of all sample means fall within two standard errors of the actual population mean. In other words: we can say with a 95% confidence level that the **population parameter** lies within a confidence interval of plus-or-minus two standard errors of the **sample statistic**. 


Given some sample statistic $\mu$ and the population parameter $\mu_0$, there are three possible **alternate hypotheses**:

| Left-tailed  | Two-sided     | Right-tailed    |
|-----------------:|:-----------------:|:-------------------:|
| $\mu \lt \mu_0$ | $\mu \neq \mu_0$   | $\mu \gt \mu_0$     |

The p-value being smaller than $\alpha$ would mean that the sample statistic under $H_0$ is in the blue areas of the **sampling distribution of sample statistic**, depending on the alternate hypothesis.

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/21a0a7a855f51f6426dfbf6115b872161490032937519"/>

_Note: for two-tailed tests, we use $\alpha/2$ for each tail. This ensures the total probability of extreme values is $\alpha$._

### Z-Scores

We can use two factors to assess the probability of observing the experimental results under the null hypothesis:
+ The [**Z-score**](https://en.wikipedia.org/wiki/Standard_score) represents the number of standard deviations an observation is from the mean.
+ The sampling distribution of sample statistic is centered around the population parameter and has a standard error linked to the population variance. 

It means that we can calculate the z-score of our sample statistic to calculate its p-value.

___

## Appendix - Further Reads

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
