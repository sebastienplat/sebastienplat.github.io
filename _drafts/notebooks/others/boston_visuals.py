###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
#import warnings
#warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import numpy as np
import pandas as pd
import shap

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import NearestNeighbors





class ModelSelector(object):

	def __init__(self, df=None):
		"""."""
		
		self.scoring = 'r2'
		
		# initialize cv with default parameters
		self.set_cv()
		
		if df is None:
			self.build_boston_dataset()
	
	
	
	
	def build_boston_dataset(self):
		"""Returns clean X, y from the Boston dataset"""
		
		# load dataset
		Xraw, y = shap.datasets.boston()

		# build df
		df_raw = pd.concat((Xraw, pd.Series(y, name='MEDV')), axis=1)

		# only keep required columns; drop undesired rows
		df = (
			df_raw[['RM','LSTAT','PTRATIO','MEDV']]
			.loc[~(
				(df_raw['MEDV'] == 50) |
				(df_raw['RM'] == 8.78)
			)]
			.reset_index(drop=True)
		)
		
		# X and y (prices adjusted for inflation)
		self.X = df.drop(columns=['MEDV'])
		self.y = 21e3 * df['MEDV']
	
	
	
	
	def set_cv(self, n_splits = 10, test_size = 0.2, random_state = 0):
		self.cv = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = random_state)
	
	
	
	
	
	
	# ----- Visualization methods ----- #
		
	def plot_features_vs_outcome(self):
		"""Plots correlation between each feature and the outcome"""
		
		y = self.y
		
		plt.figure(figsize=(20, 5))
		
		for i, col in enumerate(self.X.columns):
			
			# initialize subplot			
			plt.subplot(1, 3, i+1)			
			plt.title(col)
			plt.xlabel(col)
			plt.ylabel('prices')
			
			# Create regression line
			x = self.X[col]
			plt.plot(x, y, 'o')
			plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
			
			# start all prices at zero; format prices
			plt.gca().set_ylim([0, None])
			plt.gca().yaxis.set_major_formatter(
			   tkr.FuncFormatter(lambda x, p: "{:,}k".format(int(x/1e3))))
	
	
	
	
	def ModelLearning(self, figsize=(10,7)):
		""" 
		Calculates the performance of several models with varying sizes of training data.
		The learning and testing scores for each model are then plotted. 
		"""
				
		# Generate the training set sizes increasing by 50 (increasingly bigger dataset)
		train_sizes = np.rint(np.linspace(1, self.X.shape[0]*0.8 - 1, 9)).astype(int)
		
		# train several models with varying depths
		max_depths = [1, 3, 6, 10]
		
		# create figure
		fig = plt.figure(figsize=figsize)
		fig.suptitle
		
		for k, max_depth in enumerate(max_depths):
		
			ax = fig.add_subplot(2, 2, k+1)
			regressor = DecisionTreeRegressor(max_depth = max_depth)
			train_sizes_list, train, test = self.__build_learning_curves(regressor, cv=self.cv, scoring = self.scoring, train_sizes=train_sizes)
			ModelSelector.__plot_curve(train_sizes_list, train, test, 'max_depth = {}'.format(max_depth), ax=ax)
			
			# get legend from the first subplot
			h, l = ax.get_legend_handles_labels()

		# Visual aesthetics
		fig.legend(h, l, loc='lower center', ncol=2)
		fig.suptitle('Learning Performances', fontsize = 16, y = 1.03)
		fig.tight_layout()
		fig.subplots_adjust(top=0.95,bottom=0.15)
	
	
	
	
	def ModelComplexity(self):
		""" 
		Calculates the performance of the model as model complexity increases.
		Plots the resulting train and test performance scores.
		"""
				
		# Vary the max_depth parameter from 1 to 10
		max_depths = np.arange(1,11)

		# Calculate the training and testing scores
		regressor = DecisionTreeRegressor()
		max_depth, train, test = self.__build_validation_curves(regressor, cv=self.cv, scoring=self.scoring, param_name='max_depth', param_range=max_depths)
		
		ModelSelector.__plot_curve(max_depth, train, test, 'Complexity Performance')
	
	
	
	
	
	
	# ----- Train Model ----- #
	
	def BestModelSearch(self, search_method, perform_search=True):
		"""
		Performs grid search over the 'max_depth' parameter for a 
		decision tree regressor trained on the input data [X, y].
		"""
		
		# create regressor
		regressor = DecisionTreeRegressor(random_state=0)
		
		# hyperparameters values grid
		max_depths = np.arange(1, 11)
		params = dict(max_depth = max_depths)
		
		# convert performance_metric function into a model scorer
		scoring_fnc = make_scorer(ModelSelector.__performance_metric)
		
		# perform GridSearchCV
		if search_method == 'Grid':
			searchCV = GridSearchCV(regressor, params, cv=self.cv, scoring=scoring_fnc)
		else:
			searchCV = RandomizedSearchCV(regressor, params, cv=self.cv, scoring=scoring_fnc)
		
		if perform_search:
			# Return the optimal model after fitting the data
			grid = searchCV.fit(self.X, self.y)
			return grid.best_estimator_
		else:
			return searchCV
	
	
	
	
	
	
	# ----- Validate Reasonableness ----- #
	
	def avg_NearestNeighbors_price(self, obs, num_neighbors=5):
		"""
		Returns Price Range of the num_neighbors 
		closest to the requested obs: mean +/- 1sd.
		"""
		
		# fit nearest neighbors
		neigh = NearestNeighbors(num_neighbors)
		neigh.fit(self.X)
		
		# get nearest neighbors for the required observation
		obs = np.array(obs).reshape(1, -1) # converts 1*n vector to n*1 vector
		_, indexes = neigh.kneighbors(obs) 
		
		# avg price
		neighbors_avg = self.y.loc[indexes[0]].mean()
		neighbors_std = self.y.loc[indexes[0]].std()
		return neighbors_avg - neighbors_std, neighbors_avg + neighbors_std
	
	
	
	
	def PredictTrials(self, obs_range, num_tests=10):
		""" 
		Returns prediction for the requested obs
		for 10 iteration of the same model,
		with a different seed every time.
		"""

		# Store the predicted prices
		prices = []

		for k in range(num_tests):
			
			# Split the data
			X_train, X_test, y_train, y_test = train_test_split(
				self.X, self.y, test_size = 0.2, random_state = k)
				
			# Fit the data
			searchCV = self.BestModelSearch(search_method='Grid', perform_search=False)
			grid = searchCV.fit(X_train, y_train)
			reg = grid.best_estimator_
			
			# Make a prediction
			pred = np.round(reg.predict(obs_range) / 100) * 100
			prices.append(pred.tolist())
			
		# reset cv to its default state
		self.set_cv()
		
		return prices
	
	
	
	
	
	
	# ----- private methods ----- #
	
	def __build_learning_curves(self, estimator, cv=None, scoring=None, train_sizes=np.linspace(.1, 1.0, 5), n_jobs=None):
		
		"""
		Trains an estimator for varying training size, as defined in the train_sizes array.
		Returns arrays of train/test scores for each training size.
		--
		
		estimator : object type that implements the "fit" and "predict" methods
			An object of that type which is cloned for each validation.
			 
		X : array-like, shape (n_samples, n_features)
			Training vector, where n_samples is the number of samples and
			n_features is the number of features.

		y : array-like, shape (n_samples) or (n_samples, n_features), optional
			Target relative to X for classification or regression;
			None for unsupervised learning.
			
		train_sizes : array-like, shape (n_ticks,), dtype float or int
			+ Relative or absolute numbers of training examples that will be used to
			  generate the learning curve. 
			+ If the dtype is float, it is regarded as a fraction of the maximum 
			  size of the training set (that is determined by the selected 
			  validation method), i.e. it has to be within (0, 1].
			+ Otherwise it is interpreted as absolute sizes of the training sets.
			  
			+ Note that for classification the number of samples usually have to
			  be big enough to contain at least one sample from each class.
			  (default: np.linspace(0.1, 1.0, 5))
			  
		cv : int, cross-validation generator or an iterable, optional
			Determines the cross-validation splitting strategy.
			Possible inputs for cv are:
			  - None, to use the default 3-fold cross-validation,
			  - integer, to specify the number of folds.
			  - :term:`CV splitter`,
			  - An iterable yielding (train, test) splits as arrays of indices.

			For integer/None inputs, if ``y`` is binary or multiclass,
			:class:`StratifiedKFold` used. If the estimator is not a classifier
			or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

			Refer :ref:`User Guide <cross_validation>` for the various
			cross-validators that can be used here.
		
		scoring : string, callable or None, optional, default: None
			A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y)
			
		n_jobs : int or None, optional (default=None)
			Number of jobs to run in parallel.
			``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
			``-1`` means using all processors. See :term:`Glossary <n_jobs>`
			for more details.
		"""
		
		train_sizes, train_scores, test_scores = learning_curve(
			estimator, self.X, self.y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
		
		train, test= ModelSelector.__get_plot_range(train_scores, test_scores)
		
		return train_sizes, train, test
	
	
	
	
	def __build_validation_curves(self, estimator, cv=None, scoring=None, param_name='max_depth', param_range=[1,3,6,10], n_jobs=None):
		"""
		Trains an estimator for varying values of a given param_name, as defined in the param_range array.
		Returns arrays of train/test scores for each param value.
		"""
	
		train_scores, test_scores = validation_curve(
			estimator, self.X, self.y, cv = cv, n_jobs=n_jobs, param_name = param_name, param_range = param_range, scoring = scoring)

		train, test= ModelSelector.__get_plot_range(train_scores, test_scores)
		
		return param_range, train, test
	
	
	
	
	
	
	# ----- private methods - static ----- #
	
	@staticmethod
	def __get_plot_range(train_scores, test_scores):
		"""
		Returns mean, mean-std and mean+std of
		performance scores for both train and test sets.
		"""
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std  = np.std(train_scores, axis=1)
		test_scores_mean  = np.mean(test_scores, axis=1)
		test_scores_std   = np.std(test_scores, axis=1)
			
		train = {
			'mean':  	train_scores_mean,
			'ci_min':  	train_scores_mean - train_scores_std,
			'ci_max':  	train_scores_mean + train_scores_std
		}
		
		test = {
			'mean':  	test_scores_mean,
			'ci_min':  	test_scores_mean - test_scores_std,
			'ci_max':  	test_scores_mean + test_scores_std
		}
		
		return train, test
		
		
		
		
	@staticmethod
	def __plot_curve(param_range, train, test, title, ylim=[-0.05, 1.05], ax=None):
		"""
		Plots performance curves of both train and test sets.
		--
		
		title : string
			Title for the chart.
			
		train/test: dictionary. For each set:
			+ mean
			+ ci_min = mean - std
			+ ci_max = mean + std
		
		ylim : tuple, shape (ymin, ymax), optional
			Defines minimum and maximum yvalues plotted.
		
		ax: figure ax. Optional. If None, the function will automatically create a figure.
		"""
		
		# create figure if needed
		if ax is None:
			fig, ax = plt.subplots(nrows=1, ncols=1)
			fig.suptitle(title)
			add_legend = True
		else:
			ax.set_title(title)
			add_legend = False
			
		# axis names & limits
		ax.set_xlabel("Training Size")
		ax.set_ylabel("Score")
		if ylim is not None:
			ax.set_ylim(*ylim)
		
		# plot learning curves
		ax.fill_between(param_range, train['ci_min'], train['ci_max'], alpha=0.1, color="r")
		ax.fill_between(param_range, test['ci_min'],  test['ci_max'],  alpha=0.1, color="g")
		
		ax.plot(param_range, train['mean'], 'o-', color="r", label="Training score")
		ax.plot(param_range, test['mean'],  'o-', color="g", label="Cross-validation score")
		
		# add legend
		if add_legend:
			ax.legend(loc="best")
		
		return ax
	
	
	
	
	@staticmethod
	def __performance_metric(y_true, y_predict):
		""" 
		Calculates and returns the performance score between 
		true and predicted values based on the metric chosen. 
		Note: kwarg is here only because make_scorer requires **kwargs
		for the function to convert. 
		"""
    
		score = r2_score(y_true, y_predict)
		return score
	







	
	
	
	



	

def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)
        
        # Fit the data
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))
	
	


