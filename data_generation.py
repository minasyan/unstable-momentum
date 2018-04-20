from sklearn import linear_model, datasets
import numpy as np

# Sources:
#  [1] http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
#  [2] J. Friedman, “Multivariate adaptive regression splines”, The Annals of Statistics 19 (1), pages 1-67, 1991.
#  [3] L. Breiman, “Bagging predictors”, Machine Learning 24, pages 123-140, 1996.

'''
Vanilla Linear Data Generator with random noise for Convex Problem (Linear Regression). Used sklearn.datasets.make_
  regression, which generates a random linear Regression Problem [1]

Parameters:
  * n_samples:
      - number of samples
      - int, optional (default=10000)
  * n_features
      - number of features
      - int, optional (default=100)
  * n_informative
      - number of informative features
      - int, optional (default=15)
  * bias
      - bias of underlying linear model
      - float, optional (default=3.0)
  * noise
      - gaussian noise on output
      - float, optional (default=1.0)

Outputs:
  * X
      - input samples
      - array of shape : [n_samples, n_features]
  * y
      - output values
      - array of shape: [n_samples]
'''
def lin_gen(n_samples=10000, n_features=100, n_informative=15, bias=3.0, noise=1.0):
    #change n_samples and n_features for tarek's thing
    X, y = datasets.make_regression(n_samples, n_features, n_informative, bias, noise)
    print X, y
    return X, y


'''
Polynomial data generator for Convex Problem (Linear Regression). Used the function y = X + 0.7*X^2 +0.25*X^3 to
  generate outputs.

Parameters:
  * n_samples:
      - number of samples
      - int, optional (default=10000)
  * n_features
      - number of features
      - int, optional (default=100)
  * bias
      - bias of underlying linear model
      - float, optional (default=3.0)
  * noise
      - gaussian noise on output
      - float, optional (default=1.0)

Output
  * X:
      - input samples
      - array of shape : [n_samples, n_features]
  * y:
      - output values
      - array of shape: [n_samples]
'''
def pol_gen(n_samples = 10000, n_features = 100, bias = 3.0, noise = 1.0):
    #Generating random X and computing solution to output function
    X = np.random.rand(10000, 100)
    val = np.sum(X, 0.7*np.power(X, 2), 0.25*np.power(X, 3))

    #Adding noise to the output values
    rnd = np.random.RandomState(0)
    error = noise * rnd.randn(10000, 100)

    y = val + error

    print X, y
    return X, y

'''
Polynomial and sine transforms data generator for Convex Problem (Linear Regression) [2][3]. Used sklearn.datasets
  .make_friedman1, which generates data from Friedman 1 Regression Model, which computes the output y according to
  the formula y(X) = 10 * sin(pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
  + noise * N(0, 1). [1]

Parameters:
  * n_samples:
      - number of samples
      - int, optional (default=10000)
  * n_features:
      - number of features
      -  int, optional (default=100)
  * noise:
      - gaussian noise on output
      - float, optional (default=1.0)

Outputs:
  * X:
      - input samples
      - array of shape: [n_samples, n_features]
  * y:
      - output values
      - array of shape: [n_samples]

'''
def fried1_gen(n_samples=10000, n_features=100, noise=1.0):
    X, y = datasets.make_friedman1(n_samples, n_features, noise)

    print X, y
    return X, y
