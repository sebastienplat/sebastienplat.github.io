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

    24.2%
    

For $n$ independant Bernoulli trials having a probability of success $p$, the probability of observing $k$ successes is:

$$P(k)={n \choose k} p^k (1 - p)^{n-k}$$

Note: ${n \choose k}$ is called the **Bernoulli coefficient**.

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
    

The probability of observing $k$ events in an interval is:

$$P(k \text{ events in interval}) = \frac{\lambda^k e^{-\lambda}}{k!}$$

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
    
