## Classification Error Rate

The Bias-Variance tradeoff also applies to Classification problems, where the **Error Rate** is defined as follows:

$$\frac{1}{n} \sum_{i=1}^n I(y_i \neq  \hat{y_i}).$$

### Bayes Error Rate

The Error Rate can be minimized by assigning each observation $x_0$ to its most likely class, given its predictor values $X_1, ..., X_p$: the **Bayes classifier**. It requires knowing the **Bayes decision boundary**, that indicates where the probability of each class is 50% for a given predictor value. 

The **Bayes error rate** is akin to the error term $\epsilon$ in regression problems: it represents the noise in the data.

<img src=https://d2vlcm61l7u1fs.cloudfront.net/media%2F06f%2F06f3438f-37f3-4c03-8553-32ca946b5397%2FphpUAm7qf.png width=600px>

### K-Nearest Neighbors

For real data, we do not know the conditional distribution of $Y$ given $X$, and so computing the Bayes classifier is impossible. 

We can approximate the **Bayes Decision Boundary** by using the **K-Nearest Neighbors** classifier. For each test observation $x_0$, we:
+ calculate the probability of each class as the fraction of the K-nearest training observations having that class.
+ apply the Bayes rule to these probabilities.

The K-Nearest Neighbors classifier becomes less flexible as K increases (i.e. can capture less complex relationships):
+ small K means low bias but high variance
+ high K means high bias but low variance

___

## Support Vector Machine

More information [here](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72).
