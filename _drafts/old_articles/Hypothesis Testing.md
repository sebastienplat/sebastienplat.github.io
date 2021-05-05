# HYPOTHESIS TESTING

#### Procedure

Hypothesis testing is the use of statistics to determine the **probability that a given hypothesis is true**. It is used to make decisions about a population using sample data.

Hypothesis testing has several steps:

1. **Formulate** two competing hypotheses about a population parameter
  + $$H_0$$ or null hypothesis
  + $$H_a$$ or alternate hypothesis

2. Fix a **level of significance** $$\alpha$$: probability cutoff for making decisions about $$H_0$$

3. Use a **test statistic** to compare the sample statistic to the $$H_0$$ parameter value 

4. **Compute** the probability (p-value) of our sample producing such a statistic or one more extreme under $$H_0$$

5. **Reject of fail to reject $$H_0$$** based on the p-value and $$\alpha$$

6. State an overall **conclusion** to the test

#### Hypotheses

By convention, the **null hypothesis** states that the observed phenomena **simply occur by chance** and that the speculated agent has **no effect**. 

This null hypothesis is considered true until the evidence suggests otherwise: it will be either **nullified or not** by the test.

Let's consider a test where the null hypothesis states that $$\mu = \mu_0$$. There are three possible **alternate hypotheses**:

1. two-sided test: $$\mu \neq \mu_0$$
2. left-tailed test: $$\mu \lt \mu_0$$
3. right-tailed test: $$\mu \gt \mu_0$$

#### P-Value 

The [p-value](https://en.wikipedia.org/wiki/P-value) is the probability that our sample produces such a statistic or one more extreme under $$H_0$$.

If the p-value is too low, then it is unlikely that the $$H_0$$ probability distribution actually describe the population: we reject the null hypothesis.

$$P\leq\alpha$$:

+  the observed effect is **statistically significant**
+  the null hypothesis is ruled out, and the **alternative hypothesis** is **valid**
  
$$P\gt\alpha$$: 
  
+  the observed effect is **not statistically significant**
+ we **fail to reject** the null hypothesis
    
<br>
Let's consider a simple case: the one mean test. Our null hypothesis states that  $$\mu = \mu_0$$. Our sample of size $$n$$ has a mean $$\bar{y}$$ and a standard deviation of $$s$$.

In that case, the normalized sampling distribution of sample mean can be approximated by a t-distribution of n-1 degrees of freedom:

$$t^* = (\space \bar{y} - \mu_0 \space) / (\space s / \sqrt{n} \space)$$

We can measure the probability (p-value) that in this distribution, our sample mean takes its particular value $$\bar{y}$$. 

The p-value being smaller than $$\alpha$$ would mean that $$\bar{y}$$ under $$H_0$$ is in the blue areas in the graphs below, depending on the alternate hypothesis:

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/21a0a7a855f51f6426dfbf6115b872161490032937519"/>

#### Outcomes and Errors

There are four possible outcomes for our hypothesis testing, with two [**types of errors**](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors):

| Decision          | $$H_0$$ is True                      | $$H_0$$ is False                     |
|-------------------:|:---------------------------------:|:---------------------------------:|
| **Reject H0** | **Type I error**: False Positive   | Correct inference: True Positive |
| **Fail to reject H0** | Correct inference: True Negative | **Type II error**: False Negative |

The probabilities of making these two kinds of errors are related. If you decrease the probability of rejecting a true null hypothesis (Type I error), you increase the probability of accepting a false one (Type II error). If you want both to decrease, you have to increase the sample size.

<br>
The **Type I error** $$\alpha$$, or **significance level**, is the probability of incorrectly rejecting a true $$H_0$$. 

It indicates that a given condition is present when it actually is not, hence its name of **False Positive**.
  
<br>
The **Type II error** $$\beta$$ is the probability of incorrectly failing to reject a false $$H_0$$. 

It fails to detect that a given condition is present, hence its name of **False Negative**.

<br>
[**Power**](https://en.wikipedia.org/wiki/Statistical_power) is the probability of correctly rejecting a false $$H_0$$. It is equal to $$1 - \beta$$.


#### Power and Sample Size

There are two possible reasons for the failure to reject the null hypothesis:

+ the null hypothesis is **reasonable**, or
+ the sample size is **too small** to achieve a powerful test

To compute the sample size $$n$$ required to have a powerful test, you need:

+ $$\alpha$$: the probability of incorrectly rejecting a true $$H_0$$ (typically 5%)
+ $$\beta$$: the probability of correctly rejecting a false $$H_0$$ (typically 80%)
+ $$s$$: an estimation of the population' standard deviation
+ the minimum difference between the sample statistic and the hypothesized population value that is considered significant enough to reject $$H_0$$