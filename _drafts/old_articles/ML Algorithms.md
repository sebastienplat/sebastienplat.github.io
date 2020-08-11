# CLASSIFICATION ALGORITHMS

#### Logistic Regression

Logistic regression assumes a Gaussian distribution 
for the numeric input variables and can model binary classiﬁcation problem


#### LDA

Linear Discriminant Analysis or LDA is a statistical technique 
for binary and multiclass classiﬁcation. It too assumes a Gaussian distribution
for the numerical input variables.

#### Naive Bayes

Naive Bayes calculates the probability of each class and the conditional 
probability of each class given each input value. These probabilities are 
estimated for new data and multiplied together, assuming that they are all 
independent (a simple or naive assumption). When working with real-valued data,
a Gaussian distribution is assumed to easily estimate the probabilities for
input variables using the Gaussian Probability Density Function.

#### KNN

The k-Nearest Neighbors algorithm (or KNN) uses a distance metric to ﬁnd 
the k most similar instances in the training data for a new instance 
and takes the mean outcome of the neighbors as the prediction

#### Decision Trees

Classiﬁcation and Regression Trees (CART or just decision trees) construct 
a binary tree from the training data. Split points are chosen greedily by 
evaluating each attribute and each value of each attribute in the training data
in order to minimize a cost function (like the Gini index). 

#### SVM

Support Vector Machines (or SVM) seek a line that best separates two classes. 
Those data instances that are closest to the line that best separates 
the classes are called support vectors and inﬂuence where the line is placed. 
SVM has been extended to support multiple classes. Of particular importance is
the use of different kernel functions via the kernel parameter. 
A powerful Radial Basis Function is used by default.

#### XGBOOST VS RANDOM FORESTS

A great introduction to Random Forests and Boosted Trees can be found [here](https://xgboost.readthedocs.io/en/latest/model.html).




# CLASSIFICATION METRICS

#### Accuracy

'accuracy' 

the ratio of correct predictions
Only use when there are an equal number of observations in each class and
all predictions and prediction errors are equally important.
the closer to 1, the better

#### Log Loss

'neg_log_loss'

used when predicting membership to a given class. 
The probability can be seen as a measure of conﬁdence for a prediction. 
Predictions that are correct or incorrect are rewarded or punished proportionally to the conﬁdence of the prediction:
The use of log on the error provides extreme punishments for being both confident and wrong. 
the closer to 0, the better

#### AUC

'roc_auc'

Area under ROC Curve (or AUC for short) is a performance metric for binary 
classiﬁcation problems. It represents a model’s ability to discriminate 
between positive and negative classes. 
An area of 1.0 represents a model that made all predictions perfectly. 
An area of 0.5 represents a model that is as good as random. 
ROC can be broken down into sensitivity and speciﬁcity. 
A binary classiﬁcation problem is really a trade-off between sensitivity and speciﬁcity:
+ Sensitivity is the true positive rate also called the recall. 
  It is the number of instances from the positive (ﬁrst) class that actually predicted correctly.
+ Speciﬁcity is also called the true negative rate. 
  Is the number of instances from the negative (second) class that were actually predicted correctly.

the closer to 1, the better




# REGRESSION ALGORITHMS

#### Linear Regression

Linear regression assumes that the input variables have a Gaussian distribution. 
It is also assumed that input variables are relevant to the output variable 
and that they are not highly correlated with each other (a problem called collinearity). 

#### Ridge Regression

Ridge regression is an extension of linear regression where the loss function
is modiﬁed to minimize the complexity of the model measured as the 
sum squared value of the coeﬃcient values (also called the L2-norm).

#### LASSO

The Least Absolute Shrinkage and Selection Operator (or LASSO for short) 
is a modiﬁcation of linear regression, like ridge regression, where 
the loss function is modiﬁed to minimize the complexity of the model 
measured as the sum absolute value of the coeﬃcient values (also called the L1-norm). 

#### ElasticNet

ElasticNet is a form of regularization regression that combines the properties
of both Ridge Regression and LASSO regression. It seeks to minimize 
the complexity of the regression model (magnitude and number of regression coeﬃcients) 
by penalizing the model using both the L2-norm (sum squared coeﬃcient values) 
and the L1-norm (sum absolute coeﬃcient values). 

#### KNN

The k-Nearest Neighbors algorithm (or KNN) locates the k most similar instances
in the training dataset for a new data instance. From the k neighbors, a mean 
or median output variable is taken as the prediction. Of note is 
the distance metric used (the metric argument). The Minkowski distance 
is used by default, which is a generalization of both the Euclidean distance 
(used when all inputs have the same scale) and Manhattan distance 
(for when the scales of the input variables diﬀer).

#### Decision Tress

Decision trees or the Classiﬁcation and Regression Trees (CART)
use the training data to select the best points to split the data in order 
to minimize a cost metric. The default cost metric for regression decision trees 
is the mean squared error, speciﬁed in the criterion parameter. 

#### SVM

Support Vector Machines (SVM) were developed for binary classiﬁcation. 
The technique has been extended for the prediction real-valued problems
called Support Vector Regression (SVR). Like the classiﬁcation example, 
SVR is built upon the LIBSVM library.




# REGRESSION METRICS

#### Mean Absolute Error

'neg_mean_absolute_error' 

The Mean Absolute Error (or MAE) is the sum of the absolute differences 
between predictions and actual values. It gives an idea of how wrong 
the predictions were (ie. magnitude of the error), but no idea of 
the direction (e.g. over or under predicting). 
the closer to 0, the better

#### Mean Squared Error

'neg_mean_squared_error'

The Mean Squared Error (or MSE) is much like the mean absolute error in that 
it provides a gross idea of the magnitude of error. Taking the square root of 
the mean squared error converts the units back to the original units of the output variable 
and can be meaningful for description and presentation. This is called the Root Mean Squared Error (or RMSE). 
the closer to 0, the better

#### R2

'r2'

The R2 (or R Squared) metric provides an indication of the goodness of ﬁt of
a set of predictions to the actual values. In statistical literature 
this measure is called the coefficient of determination. 
This is a value between 0 and 1 for no-ﬁt and perfect ﬁt respectively. 

the closer to 1, the better