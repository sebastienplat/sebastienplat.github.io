# SCIPY ECOSYSTEM

#### Analysis VS Forecasting

**Time Series Analysis** aims to understand a dataset using seasonal patterns, trends, relation to external factors, etc. The Time Series is typically decomposed into 4 constituent parts:

+ **Level**: baseline value for the series if it were a straight line.
+ **Trend**: increase or decrease of the series over time.
+ **Seasonality**: repeating patterns or cycles over time.
+ **Noise**: variability in the observations that cannot be explained by the model.


**Time Series Forecasting** aims to predict future values of a dataset. This is often achieved at the expense of understanding the underlying causes behind the problem.


#### SciPy Ecosystem

[SciPy](https://www.scipy.org/) is an ecosystem of Python libraries for mathematics, science, and engineering. It includes:

+ [NumPy](http://www.numpy.org/) for efficient array operations
+ [Matplotlib](http://matplotlib.org/) for 2D plotting
+ [Pandas](http://pandas.pydata.org/) for high-performance, easy-to-use data structures and data analysis
+ [Statsmodel](http://www.statsmodels.org/) for statistical data exploration and modeling.
+ [Scikit-learn](http://scikit-learn.org/) for machine learning algorithms.


**Pandas** is well-suited for Time Series manipulations thanks to:

+ Powerful [time series](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) manipulations.
+ Explicit handling of date-time indexes in data and date-time ranges.
+ Transforms such as shifting, lagging, and filling.
+ Resampling methods such as up-sampling, down-sampling, and aggregation.


**Statsmodel** includes tools dedicated to [time series analysis](http://www.statsmodels.org/stable/tsa.html) that can also be used for forecasting:

+ Statistical tests for stationarity such as the Augmented Dickey-Fuller unit root test.
+ Time series analysis plots such as:
  + autocorrelation function (ACF)
  + partial autocorrelation function (PACF)
+ Linear time series models such as:
  + autoregression (AR)
  + moving average (MA)
  + autoregressive moving average (ARMA)
  + autoregressive integrated moving average (ARIMA)

**Scikit-learn** 

+ Data preparation tools, such as scaling and imputing data.
+ Machine learning algorithms used to model data and make predictions.
+ Resampling methods for estimating the performance of a model on unseen data, specifically the TimeSeriesSplit class.




# FORECASTING AS SUPERVISED LEARNING

In typical machine learning problems, all the observations used to make predictions are treated equally.  A time series adds an explicit order dependence between observations: it is a sequence of observations taken sequentially in time.

Time Series can be used to measure one or several variable at each time. They are called **Uni- and Multivariate Times Series**, respectively. 

The first step is to convert the Time Series into a classical Regression problem. It can be done by using prior time steps as additional input variables: the **sliding window** method.

#### Sliding Window

Observations made at prior times are called **lag times** or lags:

+ `t-n`: prior or lag time.
+ `t`:  current time and point of reference.
+ `t+n:` future or forecast time.

The number of previous time steps used as input variables is called the **window width** or size of the lag.

In the following example, we have a Univariate Time Series with $$i$$ features and a  window width of $$2$$: 

| Features | Lag Features | Output |
|:------------: |:-----------------: | :---------: |
| $$x_1 ^{(t)}, ..., x^{(t)}_i$$ | $$y^{(t-2)}, y^{(t-1)}$$ | $$y^{(t)}$$ |

_Note: the first two records have NULL values for their Lag Features so they can't be used._

The Sliding Window method can also be used for Multivariate Time Series. In this example with two variables:

| Features | Lag Features | Output |
|:------------: |:-----------------: | :---------: |
| $$x^{(t)}_1, ..., x^{(t)}_i$$ | $$y^{(t-2)}_{1}, y^{(t-2)}_{2}, y^{(t-1)}_{1}, y^{(t-1)}_{2}$$ | $$y^{(t)}_{1}, y^{(t)}_{2}$$ |


#### Multi-steps

Forecasts can be made for one or several steps ahead. They are called **One- and Multi-Steps** Forecasts, respectively.

In the following example, we have a Univariate Time Series with $$i$$ features, a  window width of $$1$$ and a two-steps forecast: 

| Features | Lag Features | Output |
|:------------: |:-----------------: | :---------: |
| $$x_1 ^{(t)}, ..., x^{(t)}_i$$ | $$y^{(t-1)}$$ | $$y^{(t)}, y^{(t+1)}$$ |

_Note: the first record has NULL values for its Lag Features so it can't be used. The last record has NULL values for its Output so it can't be used._




# DATA PREPARATION

Time series often requires cleaning, scaling, and even transformation to deal with frequency issues (too high or too low for the forecasting needs), outliers & missing values.


#### Feature Engineering

#### Resampling and Interpolation

#### Power Transforms

#### Moving Average Smoothing




# TEMPORAL STRUCTURE

#### White Noise

#### Random Walk

#### Decomposition

##### Trend

##### Seasonality

#### Stationarity




# FORECASTS MODELS

#### Box-Jenkins

#### Autoregression

#### Moving Averages

#### ARIMA

#### Autocorrelation

#### Confidence Intervals