## Definition

> As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially.

When the dimensionality increases (often with hundreds or thousands of dimensions), the volume of the space increases so fast that the available **data become sparse**. This sparsity is problematic for any method that requires **statistical significance**. It also prevents efficient clustering, as all data points become equidistant from one another.

In Machine Learning, an **enormous amount of training data** is required to ensure that there are **several samples** with each **combination of values**. A typical rule of thumb is that there should be **at least 5 training examples** for each dimension in the representation.

With a **fixed number of training samples**, the **power** of a classifier or regressor **first increases** as number of dimensions/features used is increased but then decreases. This is known as Hughes phenomenon or peaking phenomena.

A few methods exist to reduce the dimentionality of a dataset:
+ Principal Component Analysis
+ Linear Discriminant Analysis

A brief explanation of the difference between the two methods can be found [here](https://sebastianraschka.com/faq/docs/lda-vs-pca.html).

___

## Principal Component Analysis (PCA)

### Definition


**Principal Components Analysis (PCA)** creates a set of principal components that are:
+ rank-ordered by variance (the first component accounts for as much of the variability in the data as possible).
+ uncorrelated (to prevent [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity) issues).
+ low in number (we can throw away the lower ranked components as they contain little signal).

Each principal component is:
+ a **linear combination** of the individual features. 
+ **orthogonal** to the previous principal components.


### Limitations

There are a few main limitations to PCA:
+ We lose a lot of **interpretability**.
+ It is concerned only with the **(co)variance** within the predictor matrix $x$. 
+ It assumes that the most important variables are the ones with **the most variation**.

Dimension reduction by PCA can be harmful to predictions if $y$ only depends on predictors in $X$ that have a low (co)variance with other predictors.

More details on limitations [here](https://towardsdatascience.com/pca-is-not-feature-selection-3344fb764ae6).

> If the predictive relationship between the predictors and response is not connected to the predictors’ variability, then the derived PCs will not provide a suitable relationship with the response.

###  Mathematical Details

The PCA is based on the [**covariance matrix**](https://en.wikipedia.org/wiki/Covariance_matrix) of the data that has been transformed to be centered around the origin (by subtracting the mean of each variable). The eigenvalues and corresponding eigenvectors are calculated and normalized.

Zoom on these [terms](https://www.geeksforgeeks.org/principal-component-analysis-with-python/):
+ eigenvectors: new set of axes of the principal component space.
+ eigenvalues: quantity of variance that each eigenvector have.


_Note: PCA is sensitive to the relative scale of your features. Using [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) or [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) from `scikit-learn`:_
+ `MinMaxScaler` scales each observation to be a predefined range (typically 0 to 1).
+ `StandardScaler` calculates the standard score Z of each observation.

Complete examples can be found [here](https://www.geeksforgeeks.org/principal-component-analysis-with-python/) and [here](https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad).

___

## Linear discriminant analysis (LDA) for Classification

### Definition

[**Linear discriminant analysis (LDA)**](https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/) finds a linear combination of features that separates two or more classes of objects or events. 

It differs from ANOVA:
+ LDA uses **continuous features** and a **categorical outcome**.
+ ANOVA uses **categorical features** and a **continuous outcome**.


### Limitations

LDA assumptions are the same as for linear regression:
+ Normality.
+ Homogeneity of variance (homoscedasticity vs heteroscedasticity).
+ Independent observations.

It is very similar to logistic regression, but its more stringent sets of hypothesis means it is rarely used today.

### Mathematical Details

Discriminant analysis works by creating one or more linear combinations of predictors, creating a new **latent variable** for each discriminant function. LDA explicitly attempts to model the **difference between the classes** of data: it  uses information of classes to find new features in order to maximize **its separability**.


Two criteria are used by LDA to create a new axis:

+ Maximize the distance between means of the two classes.
+ Minimize the variation within each class.

### Generalization

+ **Quadratic Discriminant Analysis (QDA)**: Each class uses its own estimate of variance (or covariance when there are multiple input variables).
+ **Flexible Discriminant Analysis (FDA)**: Where non-linear combinations of inputs is used such as splines.
+ **Regularized Discriminant Analysis (RDA)**: Introduces regularization into the estimate of the variance (actually covariance), moderating the influence of different variables on LDA.

### Applications

+ **Face Recognition**: In the field of Computer Vision, face recognition is a very popular application in which each face is represented by a very large number of pixel values. Linear discriminant analysis (LDA) is used here to reduce the number of features to a more manageable number before the process of classification. Each of the new dimensions generated is a linear combination of pixel values, which form a template. The linear combinations obtained using Fisher’s linear discriminant are called Fisher faces.

+ **Medical**: In this field, Linear discriminant analysis (LDA) is used to classify the patient disease state as mild, moderate or severe based upon the patient various parameters and the medical treatment he is going through. This helps the doctors to intensify or reduce the pace of their treatment.

+ **Customer Identification**: Suppose we want to identify the type of customers which are most likely to buy a particular product in a shopping mall. By doing a simple question and answers survey, we can gather all the features of the customers. Here, Linear discriminant analysis will help us to identify and select the features which can describe the characteristics of the group of customers that are most likely to buy that particular product in the shopping mall.

