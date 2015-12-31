"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

#print('scikit-learn version is {}.'.format(sklearn.__version__))

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    hp_size_total = np.size(housing_prices)  #506
    hf_size_total = np.size(housing_features) #6578
    
    print housing_prices.shape
    print housing_features.shape
    hp_size = housing_prices.shape[0]
    hf_size = housing_features.shape[1]

    hp_max = np.max(housing_prices)    #50.0
    hp_min = np.min(housing_prices)    #5.0
    hp_mean = np.mean(housing_prices)    #22.532806324110677
    hp_median = np.median(housing_prices)    #21.199999999999999
    hp_std = np.std(housing_prices)    #9.1880115452782025

    print('Size of data (number of houses): {}'.format(hp_size))
    print('Number of features: {}'.format(hf_size))
    print('Minimum price: {}'.format(hp_min))
    print('Maximum price: {}'.format(hp_max))
    print('Mean price: {}'.format(hp_mean))
    print('Median price: {}'.format(hp_median))
    print('Standard Deviation: {}'.format(hp_std))

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?  506
    # Number of features?  13
    # Minimum price? 5.0
    # Maximum price? 50.0
    # Calculate mean price?  22.5328063241
    # Calculate median price? 21.2
    # Calculate standard deviation? 9.18801154528


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    X, y = shuffle(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

    print X_train.shape, X_test.shape

    return X_train, y_train, X_test, y_test

    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    #meanSquaredError = mean_squared_error(label, prediction)
    meanAbsoluteError = mean_absolute_error(label, prediction)

    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    # other errors are median_absolute_error and mean_absolute_error
    
    return meanAbsoluteError
    #return meanSquaredError


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    gridScorer = make_scorer(performance_metric, greater_is_better=False)

    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    crossvalidation = KFold(n=X.shape[0], n_folds=5, shuffle=True, random_state=1)
    grid = { 'splitter' : ['best', 'random'],
             'max_depth' : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20],
             #'max_features' : (3, 5, 7, 9, 11, 13, 'sqrt', 'log2', None)  default None was best
             #'min_samples_split' : (2, 3, 4, 5, 8, 10),  default 2 was best
             'min_samples_leaf' : (1, 2, 3, 4, 5)
            }

    reg = GridSearchCV(estimator=regressor, param_grid=grid, scoring=gridScorer,
        refit=True, n_jobs=1,  cv=crossvalidation)

    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    print reg.fit(X, y)

    print 'Best parameters: %s' % reg.best_params_ 
    print 'best score: %.3f' % abs( reg.best_score_) 

    
    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)

#In the case of the documentation page for GridSearchCV, it might be the case that the example is just a demonstration of syntax for use of the function, rather than a statement about 

def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()