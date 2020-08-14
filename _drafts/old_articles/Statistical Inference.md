# OVERVIEW

#### Goal of Statistical Inference

The goal of Statistical Inference is to draw conclusion about a whole population by studying only a sample. 

#### Sampling

Sampling must be probabilistic in order to make inference about the whole population. Otherwise, the inference can only be made about the sample itself.

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/ac0925d493878aa2e4c640dfa7dad1741489593445684" style="max-width:300px"/>

There are several ways to do it:

+ [**Simple random sample**](https://en.wikipedia.org/wiki/Simple_random_sample) – each subject in the population has an equal chance of being selected
+ [**Stratified random sample**](https://en.wikipedia.org/wiki/Stratified_sampling) – the population is divided into groups based on some characteristic (e.g. sex, geographic region). Then simple random sampling is done for each group
+ [**Cluster sample**](https://en.wikipedia.org/wiki/Cluster_sampling) – a random cluster of subjects is selected from the population


#### Types of studies

Studies look for relationships between different parameters of a population:

+ **Explanatory** variables called **predictors** (or independent variables) 
+ **Response** variable called **outcome** (or dependent variable)

There are two main types of studies:

+ **Observational** – measures the relationship between variables
+ **Experimental** – involves the random assignment of a treatment; researchers can draw causal conclusions

Experimental studies must be carefully designed to conclude that differences in the results of an experiment, not reasonably attributable to chance, are likely caused by the treatments.

+ **Control**: control for effects due to factors other than the ones of primary interest
+ **Random selection**: subjects should be randomly selected from the population
+ **Random assignment**: subjects should be randomly divided into the different groups
+ **Replication**: there should be enough subjects participating to the study, so the randomization creates groups that resemble each other closely and to increase the chances of detecting differences among the treatments if they exist

**Random selection** ensures that the study results can be **extended** to the entire population.

**Random assignment** of treatment avoid unintentional selection bias. Significant results can be concluded as **causal**, ie. the treatment caused the result.  

A study with both random selection and random assignment is a **completely randomized experiment**.




# RANDOM VARIABLES

#### Definitions

A variable is a quantity whose value changes. 
 
+ a **discrete** variable is a variable whose value is obtained by counting
+ a **continuous** variable is a variable whose value is obtained by measuring

A **random variable** is a variable whose value is a numerical outcome of a physical phenomenon: 

+ an experiment: result of a coin toss, etc. 
+ an industrial process: default rate in manufacturing, etc.
+ a naturally occuring event: people's height, etc.
+ ...

There are many ways to describe a random variable. Here are the most common ones.

#### Probability Distribution

A [**probability distribution**](https://en.wikipedia.org/wiki/Probability_distribution) of a random variable X tells what the **possible values** of X are and what **probability** each value has.

For example, let's consider the outcome of a coin toss X. The probability distribution of X would take the value 0.5 for Heads and 0.5 for Tails.

There are several probability distributions:

+ [**Probability Mass Function**](https://en.wikipedia.org/wiki/Probability_mass_function), or PMF, for **discrete** variables
+ [**Categorical Distribution**](https://en.wikipedia.org/wiki/Categorical_distribution) for **discrete** variables with a **finite** set of values
+ [**Probability Density Function**](https://en.wikipedia.org/wiki/Probability_density_function), or PDF, for **continuous** variables

The [**Cumulative Distribution Function**](https://en.wikipedia.org/wiki/Cumulative_distribution_function) is the probability that a random variable X will take a value less than or equal to x. It is the cumulative sum (or integral) of the probability distribution.


#### Central Tendancy

There are three common ways to measure [central tendency](https://en.wikipedia.org/wiki/Central_tendency).

+ [**mode**](https://en.wikipedia.org/wiki/Mode_(statistics&#41;): value that occurs most often in the data
+ [**median**](https://en.wikipedia.org/wiki/Median): middle value of the ordered data
+ [ **mean**](https://en.wikipedia.org/wiki/Mean): average of the data

The median is not skewed much by extremely large or small values. So it may give a better idea of a 'typical' value than the mean. This is also why it is considered a [**robust statistic**](https://en.wikipedia.org/wiki/Robust_statistics).

The image below illustrates the definition of mode, median and mean:

<img class="center-block" style="max-height:300px;" src="https://sebastienplat.s3.amazonaws.com/e4b73954b4944120752f1082b78ed3cc1489605907921"/>

The image below illustrates the behaviour of mode, median and mean for several distributions:

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/8557fe4fbefaa3f408ce0207d2169a791489605358369"/>

Many applied statistical methods require the assumption that the data is normal, or very near bell-shaped. This is why data transformation is often necessary for skewed distributions (log transform, etc.).


#### Variability

In addition to the centreal tendency, it is useful to measure the [**variability**](https://en.wikipedia.org/wiki/Statistical_dispersion) of data, also called **spread** or statistical dispersion. It is zero if all the data are the same and increases as the data become more diverse. 

Most measures of dispersion have the same units as the quantity being measured. 

There are three common ways to measure it:

+ [**Range**](https://en.wikipedia.org/wiki/Range_(statistics&#41;): maximum minus minimum (see also [this article](https://en.wikipedia.org/wiki/Sample_maximum_and_minimum))
+  [**Interquartile range (IQR)**](https://en.wikipedia.org/wiki/Interquartile_range): the difference between the thrid and first quartile
+ [**Standard Deviation**](https://en.wikipedia.org/wiki/Standard_deviation): square root of the variance, the average squared distance from the mean

**Range** is easy to calculate, but is extremely sensitive to outliers and therefore the least robust measure of variability.

##### Interquartile Range

A [**percentile**](https://en.wikipedia.org/wiki/Percentile) indicates the value below which falls a given percentage of observations in a data set. The **median** is the **50th percentile**: Fifty percent or the observations fall at or below it.

For continuous variables, percentiles represent the area under the PDF. The following graph shows the example of a normal distribution:

<img class="center-block" style="max-width: 500px;" src="https://sebastienplat.s3.amazonaws.com/8453b91f3e6f877d096d69951b5c78051489661658479"/>

**Quartiles** divide a rank-ordered data set into **four equal parts**. Each value has a 25% probability of falling into either of the four parts.
 
The values that divide each part are called the first, second, and third quartiles; they are denoted by Q1, Q2, and Q3, respectively. The [**Interquartile range (IQR)**](https://en.wikipedia.org/wiki/Interquartile_range) is equal to Q3 minus Q1. This is why it is sometimes called the **midspread** or **middle 50%**.

The following graph shows the quartiles of a bell-shaped distribution:

<img class="center-block" style="max-width: 500px;" src="https://sebastienplat.s3.amazonaws.com/4ffc9f174b46ac4bec5fcd6df42b65d81489662410273"/>


##### Box-Plot

[**Box-plots**](https://en.wikipedia.org/wiki/Box_plot) are a good way to summarize a distribution using its quartiles. 

+ the **box** always shows the **IQR with the median** inside
+ the **whiskers** representation can vary

The most commonly used by statistical softwares is the **Tukey boxplot**:

+ the whiskers extend until the smallest and highest values within **1.5 IQR** of Q1 and Q3 respectively
+ the remaining values are considered as **outliers** and represented individually


##### Standard Deviation

**Variance** is the average squared distance from the mean. **Standard deviation** is the square root of the variance. It has the **same unit** as the variable itself and its typical notation is $$\sigma$$.

The [**68–95–99.7 rule**](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule) states that "nearly all" values lie within three standard deviations of the mean. For a normal distribution, the probability that any value falls within 3 $$\sigma$$ of the mean is approx. 99.7%.

<img class="center-block" style="max-width: 500px;" src="https://sebastienplat.s3.amazonaws.com/ae16863c2895a36e3d00721cefd9d0921489748975006"/>

<br>
The following graph shows the relation between the probability distribution, box-plots and the 68–95–99.7 rule for a normal distribution:

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/fd9f7fa6f5723f925b1877b88b2fd9fc1489751308953"/>

##### Coefficient of Variation

The [**coefficient of variation**](https://en.wikipedia.org/wiki/Coefficient_of_variation) is the ratio of the standard deviation $$\sigma$$ to the mean $$\mu$$.  This is a unit-free measure of dispersion.

$$CV = \sigma / \mu$$

##### Z-Score

The [**Z-score**](https://en.wikipedia.org/wiki/Standard_score) represents the number of standard deviations an observation is from the mean:

$$Z = (y - \mu) / \sigma$$

It **requires** the mean and standard deviation of the **complete population**.




# PROBABILITY DISTRIBUTIONS

#### Probability Mass Function

The [**Probability Mass Function**](https://en.wikipedia.org/wiki/Probability_mass_function) gives the probability that a discrete random variable is exactly equal to some value.

The following graph shows an example of probability mass function. All the values of this function must be non-negative and sum up to 1.

<img class="center-block" style="max-width: 300px;" src="https://sebastienplat.s3.amazonaws.com/6add1f412d4e4b481ec8ac81421d54371489766747520"/>



For discrete variables, the variance $${\sigma}^2$$ of a population of size $$N$$ and mean $$\mu$$ is: 

$${\sigma}^2 =  \frac{1}{N}  \sum_{i=1}^{N} (y_i - \mu)^2$$

The variance $$s^2$$ of a sample of size $$n$$ and mean $$\bar{y}$$ is: 

$$s^2 =  \frac{1}{n-1}  \sum_{i=1}^{n} (y_i - \bar{y})^2$$

The sample variance $$s^2$$ is an unbiased estimate the population variance $${\sigma}^2$$. 

Why use $$n-1$$ instead of $$n$$ ? It is called the [Bessel's correction](https://en.wikipedia.org/wiki/Bessel's_correction). The intuition is that in a sample, data points are closer to $$\bar{y}$$ than to $$\mu$$. To compensate, we divide by the smaller number $$n-1$$.

In more details:

+ by using $$\bar{y}$$ instead of $$\mu$$,  you underestimate each $$y_i - \mu$$ by $$\bar{y} - \mu$$. 
+ For uncorrelated variables, the variance of a sum is the sum of the variances. 
+ So the gap with the unbiased variance is the variance of $$\bar{y} - \mu$$. 
+ This is just the [variance of $$\bar{y}$$](https://en.wikipedia.org/wiki/Variance#Sum_of_uncorrelated_variables_.28Bienaym.C3.A9_formula.29), which is $$\sigma^2/n$$. 
+ So we expect the biased estimator to underestimates $$\sigma^2$$ by $$\sigma^2/n$$. 
+ So the biased estimator = (1 − 1/n) × the unbiased estimator = (n − 1)/n × the unbiased estimator.


When working with a **random sample**, the computation yields the [**Student's t-statistic**](https://en.wikipedia.org/wiki/Student%27s_t-statistic) instead.


#### Probability Density Function

The [**Probability Density Function**](https://en.wikipedia.org/wiki/Probability_density_function) (or density curve) describes the **shape of the distribution**. It is used to calculate the probability that a variable falls within any interval of values, by integrating the PDF over the interval.

Most of the time it is **unknown**, and we try to **estimate it** based on a **sample** of points taken from that distribution.

The distribution of sample data can be represented by an [**histogram**](https://en.wikipedia.org/wiki/Histogram): divide the entire range of values into a series of intervals (or bins), then count how many values fall into each interval. 

An histogram can be normalized to show the proportion of values that fall into each bin, with the sum of the heights equaling 1. This gives a rough assessment of the [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) of the variable.

But histograms have a few caveats. They:

+ are not smooth
+ depend on end points of bins
+ depend on bins width

A [**Kernel Density Estimation**](https://en.wikipedia.org/wiki/Kernel_density_estimation) (or KDE) is used to smooth the histogram into a curve that approximates the hypothesized PDF. More details about KDE can be found in  [this article](http://www.mvstat.net/tduong/research/seminars/seminar-2001-05/).

The following graph shows a sample of a normal distribution. The histogram has been normalized to show densities. The red line is the KDE. The blue line is the theoretical PDF of the underlying normal distribution. 

<img class="center-block" style="max-width: 500px;" src="https://sebastienplat.s3.amazonaws.com/13ac3ab4813448e1554ed5c89c7ebdfe1489656601333"/>

See also:

+ [KDE in Python](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/)




# SAMPLING DISTRIBUTIONS

Most of the time the true value of a population's parameter is **unknown**, and we try to **estimate it** based on a **sample** of points taken from that population.

But how does the sample statistics relate to the actual population parameter ?

+ the **true value** of a population's parameter is **fixed**
+ **samples** from the same population have **different statistic values**

As it depends on the sample, the statistic is random: it has a **sampling distribution** that we can study. 

For the rest of this page, we consider a population of mean $$\mu$$ and standard deviation $$\sigma$$.

#### Law of Large Numbers

The [**Law of Large Numbers**](https://en.wikipedia.org/wiki/Law_of_large_numbers) states that:

> The **mean and standard variation of a sample** get **closer to their expected value** as **the sample size increases**.

<br>
The example below shows the sampling distribution of 1000 samples of 40 exponential values ([exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) of mean=5 and sd=25). The mean of both distributions is close to the value they estimate.

![samping dist.png](https://sebastienplat.s3.amazonaws.com/1481d044baf7373620d967594e8509b31490003412437)


#### Central Limit Theorem

The [**Central Limit Theorem**](https://en.wikipedia.org/wiki/Central_limit_theorem) states that, according to the Law of Large Numbers:

> For large samples (n ≥ 30), the **sampling distribution of the sample mean** $$\bar{y}$$ is approximately **normal**, **regardless of the original distribution**:

>+ mean $$\mu$$
>+ standard error $$\sigma / \sqrt{n}$$

The **standard error** is the **standard deviation** of the **sampling distribution** of the sample mean.

<br>
The example below shows:

+ the PDF of the exponential distribution (mean=5 and sd=25), compared to the normal distribution of same mean and sd
+ the empirical distribution of samples mean VS the one predicted by the CLT

The exponential distribution is not even close to being normal, but the empirical distribution is very close to being normal, as predicted by the CLT.

![clt.png](https://sebastienplat.s3.amazonaws.com/e26192439d8acfe388ec2b7f3f61ebd91490003678751)


#### Confidence Interval

We usually have only **one sample** to study a population, and we do not know **how close** its mean is from the actual population mean.

We estimate the population mean by using the **sample mean** plus or minus a **margin of error**. The result is called a **confidence interval** for the population mean.

A [**Confidence Interval**](https://en.wikipedia.org/wiki/Confidence_interval) is expressed with a confidence percentage $$1 - \alpha$$. It means that for $$1 - \alpha$$ of all samples of size $$n$$ that could be drawn from the population, the Confidence Interval will include $$\mu$$.

It means that there is an $$\alpha$$ probability that our sample mean is so far from the actual mean that its Confidence Interval does not include $$\mu$$.

<br>
The 95% Confidence Interval is the most commonly used. As the sampling distribution of the sample mean is roughly normal, our sample mean $$\bar{y}$$ has a 95% probability of being between -2 and +2 standard errors of $$\mu$$:

$$P(\bar{y} \in [\space\mu \pm 2 se\space] ) = P(\bar{y} \in [\space\mu \pm 2 \sigma/\sqrt{n}\space] ) \simeq 0.95$$ 

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/9ec352c1ff3263bdd17c8407d30c1f0b1490007929308"/>

We can deduce that:

$$P(\mu \in [\space\bar{y} \pm 2 \sigma/\sqrt{n}\space] ) \simeq 0.95$$


#### T-Distribution

The CLT Confidence intervals **does not works** when:

+ **$$\sigma$$ is unknown**
+ the sample size **$$n$$ is small**

The [Student’s t distribution](https://en.wikipedia.org/wiki/Student's_t-distribution) is used instead. 

Its tails are **thicker than normal**, so its Confidence Interval is **wider** for the same Confidence Level. This is because **estimating the population standard deviation** introduces more **uncertainty**.

The **sampling distribution** of the sample mean has to be roughly **normal** for the t-distribution to work well. It means that either:

+ the sample size **$$n$$ is large**, OR
+ the **population** is roughly **normal** (very small samples)

If the sample size is very small, we can use normal probability plots to check whether the sample may come from a normal distribution.

It the t-distribution cannot be used, it is possible to use a more robust procedure such as the one-sample [**Wilcoxon procedure**](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test).


<br>
Back to the Exponential Distribution, Fig.6 shows the experimental sample mean distribution vs T vs Normal for different sample sizes: 2, 5, 10 and 20. The t-distribution gets close to normal even for relatively small sample sizes. But it does not approximate the empirical distribution very well for smaller sample sizes.

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/dc954f3e9562d53b7829a2adcd2854ff1490011103173"/>

<br>
The $$1 - \alpha$$ T Confidence Interval is:

> $$\bar{y} \pm T_{\alpha/2, n-1} \times SD_{Est} / \sqrt{n}$$

> where $$T_{\alpha/2, n-1}$$ is the distance from the mean of the t-distribution with n-1 degrees of freedom above which lay $$\alpha/2$$ percent of all observations

The Confidence Interval is narrower (ie. more precise) when:

+ the confidence level is low
+ the sample size is large
+ the standard deviation of the population is small



#### Sample size

In a normal distribution, 95% of the data is between --2 and +2 standard deviations from the mean. Even for skewed data, going two standard deviations away from the mean often captures nearly all of the data.

If we know the minimum and maximum values that the population is likely to take (excluding outliers), we can suppose they represent this interval of four standard deviations.

It means the standard deviation of a population $$\sigma$$ can be approximated by:

$$\sigma \simeq 1/4 \times \Delta_{range}$$

If we know the margin of error $$E$$ we are ready to accept at $$1 - \alpha$$ confidence, the sample size we need can be approximated by:

$$n \simeq [Z_{\alpha/2} \times \sigma / E]^2 \simeq [Z_{\alpha/2} \times  \Delta_{range} / 4 E]^2 $$

A more accurate method to estimate the sample size: iteratively evaluate the following formula, until the $$n$$ value chosen to calculate the t-value matches the resulting $$n$$.

$$n \simeq [t_{\alpha/2, n-1} \times  \Delta_{range} / 4 E]^2 $$




# 