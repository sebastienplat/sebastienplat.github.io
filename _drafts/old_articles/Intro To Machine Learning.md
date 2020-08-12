> This article is mostly based on the excellent Stanford course on [Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) by Andrew Ng. It also takes elements from this [blog](http://scott.fortmann-roe.com) for the variance-bias tradeoff and model prediction errors.

# INTRODUCTION

#### What is Machine Learning

> "Let's consider a task T with performance measure P. A computer program is said to learn from experience E if its performance at T, measured by P, improves with E."

Example: playing checkers
+ T = the task of playing checkers
+ P = the probability that the program will win the next game
+ E = the experience of playing many games of checkers


#### Supervised Learning

In supervised learning, we are given a data set for which the correct output is known. The algorithm will learn the best relationship between input parameters in order to predict the output of any set of data.

The user has to define the relevant input variables, or **features**, and the model he/she wants to apply to the problem at hand: the **hypothesis function**.

Supervised learning problems are categorized into **regression** and **classification** problems.

+ **regression**: we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
+ **classification**: we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.


#### Unsupervised Learning

Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results, i.e., there is no teacher to correct you. It’s not just about clustering. For example, associative memory is unsupervised learning.

A few examples:

+ **clustering**: Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number that are somehow similar or related by different variables, such as word frequency, sentence length, page count, and so on.
+ **Cocktail Party Algorithm**: find structure in messy data- like identifying individual voices and music from a mesh of sounds at a cocktail party




# SUPERVISED LEARNING PRINCIPLES

#### Goal

As mentioned in the Introduction, all supervised learning problems start with a data set for which the correct output is known.  The user has to define the relevant input variables, or **features**, and the model he/she wants to apply: the **hypothesis function**. 

This model should:

1. reliably link our input data to our output data
2. reliably predict the output of new data, ie. generalize well


#### Training, Validation, Test Set

Once the parameters have been chosen, we need to train our model. We also want to check if it generalizes well, so we usually split our data set in two parts:

+ **training set**: approx. 70% of our total
+ **test set**: the remaining 30%

Once our model is trained, we can apply it to our **test set** to measure how well it performs with **new examples**.

We could be tempted to fine tune our model in order to improve its performance, going back and forth between training and test set. But in doing so, we would be indirectly using the test set as an extension of the training set, which is not advised. 

> To get the most reliable performance measurement, the **test set** should be used **only once the model is finalized**, and remain unknown and unused during the entire conception process.

What we can do instead is **split our training set** and set aside a small subset for **validation**. Several techniques can be used *(more details [here](https://en.wikipedia.org/wiki/Cross-validation_(statistics&#41;))*.

A typical ratio would be:

+ **training set**: approx. 60% of our total
+ **validation set**: 20%
+ **test set**: 20%

_Note: a good practice is to randomize the sampling to prevent bias, especially if the data was sorted beforehand._

#### Bias vs Variance

There are two sources of error that prevent supervised learning algorithms from generalizing beyond their training set *([source](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff))*:

+ **bias**: errors from erroneous assumptions in the learning algorithm. An algorithm with high bias will miss the relevant relations between features and target outputs (underfitting).
+ **variance**: errors from sensitivity to small fluctuations in the training set. An algorithm with high variance will model the random noise in the training data rather than the intended outputs (overfitting).

Another way to look at it ([source](http://scott.fortmann-roe.com/docs/BiasVariance.html)): given several valid training sets, we could build several versions of our model. Due to randomness in the underlying data sets, the resulting models will have a range of predictions for a given input. 

+ **bias**: how far off in general these models' predictions are from the correct value.
+ **variance**: how much the predictions for a given point vary between different realizations of the model.

In the following image, we show different predictions for a given input. Each dot represents the same model, but trained with a different training set.

![Bias_Variance.jpg](https://sebastienplat.s3.amazonaws.com/a9a3a238b8b5a0bfe07d83b1f07c85bd1472143621831)


#### Identifying Bias & Variance

One way to identify bias & variance is to compare the predictions errors of the training set vs the validation set. 

The more complex the model, the closer it fits the training set. But a model that is **too complex** will include the noise in the training set and generalize poorly: its prediction error for the validation set will be much higher.

On the contrary, a model that is **too simple** will not predict either set accurately: both validation error and training error will be high.

The below figure shows the relationship between model complexity and training/validation errors. The optimal model has the lowest generalization error, and is marked by a dashed line.

![highvariance.png](https://sebastienplat.s3.amazonaws.com/d349c234f3a819be20b29ad950e629701472143635989)

The below table recaps how the compared errors behave for different models: 

| status | training error | validation error |
|:---------:|:-------------------:|:-----------------------:|
|**underfit - high bias**| HIGH | HIGH |
|**good fit**|LOW |LOW|
|**overfit - high variance**  | LOW| HIGH |


#### Dealing w/ Bias & Variance

+ get more training examples
+ try a smaller set of features
+ try getting additional features
+ adding polynomial features ($$x_1^2, x_2^2, x1x2$$, etc.)
+ increasing / decreasing the regularization parameter $$\lambda$$

See also:

+ [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
+ [Feature Selection](https://en.wikipedia.org/wiki/Feature_selection)
+ [Feature Extraction](https://en.wikipedia.org/wiki/Feature_extraction)
+ [Kernel Smoothing](https://en.wikipedia.org/wiki/Kernel_smoother)
+ [ROC Curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

More generally:

+ [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)
+ [Bias Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)




# MATHEMATICAL NOTIONS

#### Notations

| term | meaning |
|:-------:|:------------- |
|$$x$$| input data set |
|$$m$$| size of our data set, ie. the number of our training examples |
|$$n$$| number of features in our model |
|$$x_j^{(i)}$$| value of feature $$j$$ of the $$i^{th}$$ training example |
|$$y$$| output data set |

_Note: $$x$$ is a $$m*n$$ matrix and $$y$$ is a $$m*1$$ vector._


#### Hypothesis Function

$$h_\theta$$ can be expressed for each training example $$x^{(i)}$$:

$$
h_\theta (x^{(i)}) = f(\theta_0 + \theta_1 * x_1^{(i)} + \ldots + \theta_n * x_n^{(i)})
$$

Or as a matrix product:

$$h_\theta (x^{(i)}) = f(X^{(i)} \theta)$$, where
$$
X^{(i)} = 
\begin{bmatrix}
1 & x_1^{(i)} & \ldots & x_n^{(i)}
\end{bmatrix}
$$
and
$$
\theta = 
\begin{bmatrix}
\theta_0\\
\theta_1\\
\vdots\\
\theta_n
\end{bmatrix}
$$

$$\theta$$ is the list of all coefficients (called **parameters**) of our hypothesis function: one for each feature plus the bias $$\theta_0$$. It is a $$(n+1)*1$$ vector. 

$$\theta$$ is the same for all examples. Once it is accurate enough _(see below)_, we can use it to estimate the output of any new example.

_Note: f depends on the problem type: linear or logistic regression._



#### Cost Function

The **cost function** $$J(\theta)$$ measures the accuracy of our hypothesis function: the average discrepancy between $$h_\theta$$ results and the actual outputs.

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m g (h_\theta(x^{(i)}) - y^{(i)})$$

_Note: g depends on the problem type: linear or logistic regression._


#### Gradient Descent

With our hypothesis function and a way of measuring how accurate it is, all we need now is a way to **automatically improve** it. That's where gradient descent comes in.

We make **incremental steps** towards the minimum (or at least, a local minimum) of our cost function. The direction is found by calculating its **slope for each parameter** at its current point, ie. its derivative:

$$dir_j = \frac{\delta}{\delta \theta_j} J(\theta)$$

For each step, the new set of parameters is given by the formula:

$$
\theta_j := 
\theta_j - \alpha * dir_j = 
\theta_j - \alpha * \frac{\delta}{\delta \theta_j} J(\theta)
$$

$$\alpha$$ is the size of each incremental step. It is called the **learning rate**.

*Note: more sophisticated algorithms can be used, like 
[BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
or [Conjugate Gradient](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method), etc. But their implementation ([optim()](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html) in R) still requires $$J(\theta)$$ and its gradients. *

#### Optimizing Gradient Descent

$$\theta$$ descends quickly on small ranges and slowly on large ranges, so it will oscillate inefficiently down to the optimum when the variables are very uneven.

A good way to prevent this is to normalize input variables so they are all in the same range (typically a range of 1 centered in 0):

+ **scaling**: divide the values by the feature's range (ie. max - min), resulting in a new range of 1
+ **mean normalization**: substract the mean from all values, resulting in a new average of 0




# TO FOLLOW UP

#### General Points

+ learning curves
+ 70 training / 20 CV / 10 test
+ n > 10000 => random sampling
+ otherwise, check mean/median/variance
+ cross validation: k-fold, unless order is important (time series, etc.)
+ also see LOOCV
+ bootstrapping: samples of size n, drawn with repetition
+ ensembling of predictors


#### See Also

See also:

+ [Introduction to ML](http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html)
+ [Tutorial](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction)
+ [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) for NLP
+ [LDA paper](http://ai.stanford.edu/~ang/papers/jair03-lda.pdf) for NLP
