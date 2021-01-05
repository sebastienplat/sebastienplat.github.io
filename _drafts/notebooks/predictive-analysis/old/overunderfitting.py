

"""
This script returns example plots of over- and underfitting models.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class OverUnderFit():
	"""
	Example class for Over-Underfitting.
	Builds dummy datasets with different available true functions:
	+ cos: high bias
	+
	+
	The only public method:
	1. trains a model on different subsamples of the training set
	   for different polynomial degrees
	2. calculates train/test MSE + test bias squared/variance
	3. display the results in nice plots
	"""






	# --------------- Init Functions ---------------#

	def __init__(self, n_samples=6000, true_func='cos'):

		self.true_func = true_func
		self.n_samples = n_samples
		self._init_data()




	def _get_true_values(self, X):
		"""
		Returns true values & true values + noise
		for the required X vector
		"""

		# set seed
		np.random.seed(X.shape[0])

		# build true y values
		if self.true_func == 'cos':
			y_true = np.sin(1.5 * np.pi * X)
		elif self.true_func == 'poly':
			y_true = X**5 + 0.6*X**3+X

		# add noise
		y_noise = np.random.randn(X.shape[0]) * 0.2
		y_rand = y_true + y_noise

		return y_true, y_noise, y_rand




	def _init_data(self):
		"""
		Init function; Creates X & y = cos(X) + noise for the n samples
		"""

		# set seed
		np.random.seed(0)

		# training set
		self.X_train = np.sort(np.random.rand(self.n_samples)) * 1.5
		_, _, self.y_train = self._get_true_values(self.X_train)

		# test set
		self.X_test = np.linspace(0, 1, 100) * 1.5
		self.y_test_true, self.y_test_noise, self.y_test = self._get_true_values(self.X_test)






	# --------------- Main Function ---------------#

	def overunderfitting_example(self, degrees=(1, 4, 15), test_size=200, n_iterations=50):
		"""
		Calculate train/test MSE + test bias/variance
		for all the degrees values.
		Plot results.
		"""

		# init arrays
		mse = np.zeros((len(degrees), 2))
		test_errors = np.zeros((len(degrees), 3))
		preds = np.zeros(( len(degrees), self.X_test.shape[0] ))

		# predict for each degree
		for (i, degree) in enumerate(degrees):

			train_MSE_avg, test_MSE_avg, test_bias_squared, test_variance, mean_preds = \
				self._calculate_errors(degree, test_size, n_iterations)

			mse[i, 0] = train_MSE_avg
			mse[i, 1] = test_MSE_avg

			test_errors[i, 0] = test_bias_squared
			test_errors[i, 1] = test_variance
			test_errors[i, 2] = np.var(self.y_test_noise)  # irreducible error term

			preds[i, :] = mean_preds

		# plot test set
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
		fig.suptitle('Test Predictions vs Real Values', fontsize=14)

		ax.plot(self.X_test, self.y_test_true)
		ax.plot(self.X_test, preds.transpose())
		ax.scatter(self.X_test, self.y_test)

		# plot results
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4))
		fig.suptitle('Mean Squared Error', fontsize=14)

		ax1_plot = ax1.plot(degrees, mse)
		ax1.set_title('Training vs Test MSE')
		ax1.legend(ax1_plot, ('Train', 'Test'))

		ax2_plot = ax2.plot(degrees, test_errors)
		ax2.set_title('Breakdown of Test MSE')
		ax2.legend(ax2_plot, ('Bias', 'Variance', 'Irreducible'))

		fig.subplots_adjust(top=0.8)






# --------------- Calculation Functions ---------------#

	def _calculate_errors(self, polynom_degree, test_size, n_iterations):
		"""
		1. Get predictions for n_iterations of the same model
		   with different seeds
		2. Calculate average train/test MSE
		3. Calculate average test bias/variance
		"""

		# init arrays
		train_MSE = np.zeros(n_iterations)
		test_MSE = np.zeros(n_iterations)
		test_preds = np.zeros((n_iterations, self.y_test.shape[0]))

		# get predictions for each iteration of the model
		for itr in range(n_iterations):

			# predictions
			y_train, y_train_preds, y_test_preds = self._predict(
				polynom_degree, test_size, random_state=itr)

			# train MSE
			train_MSE[itr] = np.mean((y_train_preds - y_train)**2)

			# test MSE
			test_MSE[itr] = np.mean((y_test_preds - self.y_test)**2)

			# test preds
			test_preds[itr, :] = y_test_preds

		# store test values
		train_MSE_avg = train_MSE.mean()
		test_MSE_avg = test_MSE.mean()

		# useful metrics
		mean_preds = test_preds.mean(axis=0)
		mean_squared_preds = (test_preds**2).mean(axis=0)

		# estimate test bias
		test_bias_squared = (mean_preds - self.y_test_true)**2	# bias for each x0
		test_bias_squared = test_bias_squared.mean()      		# avg bias

		# estimate test variance
		test_variance = mean_squared_preds - mean_preds**2 # variance for each x0
		test_variance = test_variance.mean()               # avg variance

		return train_MSE_avg, test_MSE_avg, test_bias_squared, test_variance, mean_preds






	# --------------- Prediction Functions ---------------#

	def _predict(self, polynom_degree, test_size, random_state=0):
		"""
		1. Subsample training set(using random_state and train_size)
		2. Add polynomial & interaction features to the training set
		   (that starts as a 1D vector of random values)
		3. Fits a linear regressor to the augmented training subset
		4. Returns predictions for the training and test sets
		"""

		# subset train set
		X_train, y_train = self._subset_train(test_size, random_state=random_state)

		# convert 1D vectors to 2D arrays
		X_train = X_train[:, np.newaxis]
		X_test = self.X_test[:, np.newaxis]

		# fit linear regression on polynomial pipeline
		pipeline = self._build_pipeline(polynom_degree)
		pipeline.fit(X_train, y_train)

		# get train/test predictions
		y_train_preds = pipeline.predict(X_train)
		y_test_preds = pipeline.predict(X_test)

		return y_train, y_train_preds, y_test_preds




	def _subset_train(self, test_size, random_state=0):
		"""
		Return a random subset of
		test_size observations from the train set
		"""

		_, X_train_subset, _, y_train_subset = train_test_split(
                 self.X_train,
			self.y_train,
			test_size=test_size,
			random_state=random_state
		)

		return X_train_subset, y_train_subset




	def _build_pipeline(self, polynom_degree):
		"""
		1. Adds polynomial and interaction features to X
		2. Predicts outcomes using the LinearRegression estimator
		3. Returns the combined pipeline
		"""

		polynomial_features = PolynomialFeatures(degree=polynom_degree, include_bias=False)
		linear_regression = LinearRegression()

		pipeline = Pipeline([
            ("polynomial_features", polynomial_features),
			("linear_regression", linear_regression)
		])

		return pipeline
