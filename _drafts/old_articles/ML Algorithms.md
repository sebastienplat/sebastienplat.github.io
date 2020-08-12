
# CHOOSING AN Algorithms

<img src="http://scikit-learn.org/stable/_static/ml_map.png"/>

+ [decision trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
+ [plotting decision trees](http://scikit-learn.org/stable/modules/tree.html#classification)

See also:
+ [Summary by Type](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
+ [List of commonly used algorithms](https://towardsdatascience.com/supervised-learning-algorithms-explanaition-and-simple-code-4fbd1276f8aa).
+ [List of commonly used algorithms](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/).

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

See also:

+ Definition on [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
+ [ROC Analysis](http://mlwiki.org/index.php/ROC_Analysis)
+ [ROC curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
+ [ROC AUC score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
+ [AUC score](http://fastml.com/what-you-wanted-to-know-about-auc/)
+ Plot for [multi-class](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
+ Plot in [cross-validation](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)
+ [Youden's Index](https://en.wikipedia.org/wiki/Youden's_J_statistic)

#### SCIKIT LEARN

```
# import relevant packages
from sklearn.family import Model
from sklearn.model_selection import train_test_split

# instanciate model
my_model = Model(model_params)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=7)

# train model
model.fit(X_train, y_train)

# get probabilities for each label per observation
# test_size * number_of_labels
predictions = model.predict_proba(X_test)

# get predictions (learned label per observation)
# label with the highest probability
# test_size * 1
predictions = model.predict(X_test)
```

We can now evaluate our model by comparing our predictions to the correct values. 

The evaluation method depends on the problem & the ML algorithm.

#### SCIKIT-LEARN EXAMPLE

```
from sklearn import metrics

# predict on train set to set threshold
train_prob = model.predict_proba(X_train)

# calculate ROCAUC points
fpr, tpr, thresholds = metrics.roc_curve(y_train, train_prob[:, 1])
roc_auc = metrics.auc(fpr, tpr)

# get optimal threshold - Youden's index
optim_threshold = thresholds[np.nanargmax(tpr - fpr)]
print(optim_threshold)

# confusion matrix - default threshold
train_pred_default = model.predict(X_train)
cm = pd.DataFrame(metrics.confusion_matrix(y_train, train_pred_default), 
                  index=["actual 0", "actual 1"])
print(cm)

# confusion matrix - optimal threshold
train_pred_optim = [0 if x < optim_threshold else 1 for x in train_prob[:,1]]
cm = pd.DataFrame(metrics.confusion_matrix(y_train, train_pred_optim), 
                  index=["actual 0", "actual 1"])
print(cm)

# plot ROCAUC
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(fpr, tpr)
ax.set_title('ROC area = %0.3f' % roc_auc)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
```

#### Convert Categorical Data into INT

```
# convert categorical data into int
label_mapping = {}
cols = [list_of_cols]
for col in cols:
  df[col], label_mapping[col] = pd.factorize(df[col])
```

#### Choose and train model
```
# Choose and train model
from sklearn import ensemble
from sklearn.model_selection import train_test_split

# split X & y
y_col = col_name
X = df.drop([y_col], axis=1)
y = df[y_col].values

# split train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# train model
model = ensemble.GradientBoostingRegressor(max_depth=8, subsample=0.9)
model.fit(X_train, y_train)
```

#### Get features importance

```
# Get Feature Importance from the classifier
feature_importance = model.feature_importances_

# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())

# features importance as df
features_importance_df = pd.DataFrame(feature_importance, index=X.columns.values, columns=["importance"])
features_importance_df = features_importance_df.sort_values(by="importance", ascending=True)

features_importance_df.plot.barh()
```

#### Confusion matrix

```
from sklearn import metrics

# train confusion matrix & accuracy
train_prob = model.predict(X_train)
train_pred = [0 if x < 0.5 else 1 for x in train_prob]
print("\nAccuracy (Train): %.4g" % metrics.accuracy_score(y_train, train_pred))
print("Confusion Matrix (Train)")
cm = pd.DataFrame(metrics.confusion_matrix(y_train, train_pred), 
                  index=["actual 0", "actual 1"])
print(cm)

# test confusion matrix & accuracy
test_prob = model.predict(X_test)
test_pred = [0 if x < 0.5 else 1 for x in test_prob]
print("\nAccuracy (Test): %.4g" % metrics.accuracy_score(y_test, test_pred))
print("Confusion Matrix (Test)")
cm = pd.DataFrame(metrics.confusion_matrix(y_test, test_pred), 
                  index=["actual 0", "actual 1"])
print(cm)
```

#### Area Under ROC Curve

```
from sklearn import metrics

# predict on train set to set threshold
train_prob = model.predict(X_train)

# calculate ROCAUC points
fpr, tpr, thresholds = metrics.roc_curve(y_train, train_prob)
roc_auc = metrics.auc(fpr, tpr)

# plot ROCAUC
fig, ax = plt.subplots(figsize=(5,3))
ax.plot(fpr, tpr)
ax.set_title('ROC area = %0.3f' % roc_auc)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')

# confusion matrix - default threshold
train_pred_default = [0 if x < 0.5 else 1 for x in train_prob]
cm = pd.DataFrame(metrics.confusion_matrix(y_train, train_pred_default), 
                  index=["actual 0", "actual 1"])
print("Confusion Matrix (Train - default)")
print(cm)

# get optimal threshold - Youden's index
optim_threshold = thresholds[np.nanargmax(tpr - fpr)]

# confusion matrix - optimal threshold
train_pred_optim = [0 if x < optim_threshold else 1 for x in train_prob]
cm = pd.DataFrame(metrics.confusion_matrix(y_train, train_pred_optim), 
                  index=["actual 0", "actual 1"])
print("Confusion Matrix (Train - optimal)")
print(cm)
```




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




# K-MEANS

#### Algorithms

Pipeline

+ Standardize
+ Convert into categorical
+ PCA

Supervised Learning

+ ** K-Nearest Neighbors**: predict the class of an item based on the most frequent class of its K nearest neighbors


Unsupervised Learning

+ **K-Means Clustering**: Clustering items into K groups based on their distances (works best with standardized features) 

#### K-Means Clustering

+ [choosing K](http://stackoverflow.com/questions/1793532/how-do-i-determine-k-when-using-k-means-clustering)
+ [python code for choosing K](http://datascience.stackexchange.com/questions/6508/k-means-incoherent-behaviour-choosing-k-with-elbow-method-bic-variance-explain)
