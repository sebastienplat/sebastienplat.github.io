## Correlation between variables

Correlation is the measure of dependance between two variables; It typically indicates their linear relationship, but more broadly measures how in sync they vary. This is expressed by their **covariance**. 

A more common measure is the **[Pearson product-moment correlation coefficient](https://en.wikipedia.org/wiki/Correlation_and_dependence#Pearson's_product-moment_coefficient)**, built on top of the covariance. It's akin to the standard variation vs the variance for bivariate data and represents how far the relationship is from the line of best fit.

The correlation coefficient divides the covariance by the product of the standard deviations. This normalizes the covariance into a unit-less variable whose values are between -1 and +1.

The line of best fit has a slope equal to the Pearson coefficient multiplied by SDy / SDx.

___

## Linear Regression

+ only incude variables that are correlated to the outcome.
+ check for collinearity.
