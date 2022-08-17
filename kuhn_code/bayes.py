# %%
import warnings
from typing import Union

import numpy as np
import torch
from sklearn import datasets
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
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

# %% Bayes

basic_bayes = GaussianNB()
bayes_classifier = GaussianNB().fit(feature_standardized, iris.target)
# bayes_classifier = GaussianNB(priors=[0.2, 0.4, 0.4]).fit(feature_standardized, iris.target)

classifier_sigmoid = CalibratedClassifierCV(basic_bayes, method="sigmoid")
classifier_sigmoid.fit(feature, iris.target)

one_observation = np.array([[5.0, 3.6, 1.3, 0.25]])
classifier_sigmoid_proba = classifier_sigmoid.predict_proba(one_observation)

print(bayes_classifier.predict(one_observation))
print(bayes_classifier.predict(one_observation))

vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

vector_c = vector_b @ vector_a
print(vector_c)
