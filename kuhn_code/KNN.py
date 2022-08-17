# %%
import warnings
from typing import Union

import numpy as np
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% other handy functions

# get the statistics of numpy array
def get_statistics(numpy_array: Union[np.ndarray, torch.Tensor], name=None):
    """
    get the statistics of numpy array
    :param numpy_array: numpy array
    :return: none
    It will print many statistics of numpy array.
    """
    # if it's tensor, then convert to numpy array
    print()
    if name is not None:
        print("################################################## %s" % name)
    else:
        print("##################################################")
    if isinstance(numpy_array, torch.Tensor):
        try:
            numpy_array = numpy_array.numpy()
        except:
            warnings.warn("Can't convert to numpy array. Maybe data is on GPU.", UserWarning)
            numpy_array = numpy_array.cpu().detach().numpy()
    statistics_dict = {
        "mean": np.mean(numpy_array),
        "std": np.std(numpy_array),
        "max": np.max(numpy_array),
        "min": np.min(numpy_array),
        "median": np.median(numpy_array),
        "variance": np.var(numpy_array)
    }
    for key in statistics_dict:
        print("%s: %.4f" % (key, statistics_dict[key]))
    print("##################################################")
    print()
    return statistics_dict


# %% data preprocessing


iris = datasets.load_iris()
feature = iris.data
standardizer = StandardScaler()
feature_standardized = standardizer.fit_transform(feature)
# %% KNN
nearest_neighbor = NearestNeighbors(n_neighbors=3).fit(feature_standardized)

new_observations = [1, 1, 1, 1]

distance, indices = nearest_neighbor.kneighbors([new_observations])
print("hello")
print("hello")

# %% KNN + GridSearchCV Using Pipeline

knn = KNeighborsClassifier(n_neighbors=5)
pipe = Pipeline([('standardizer', standardizer), ('knn', knn)])

search_space = [
    {'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    # {'knn__n_neighbors': list(np.arange(1, 10))},
    # {'knn__weights': ['uniform', 'distance']},
    # {'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
]

classifier = GridSearchCV(pipe, search_space, cv=5, n_jobs=1).fit(feature_standardized, iris.target)
best_param = classifier.best_estimator_.get_params()
print(classifier.best_estimator_.get_params()["knn__n_neighbors"])


