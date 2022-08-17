from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
# use train_test_split to split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
print(X_train.shape)

# # get 0,2,3 columns of the data
# X_train_0 = X_train[:, 0]
# X_train_1 = X_train[:, 1]
# X_train_2 = X_train[:, 2]
# X_train_3 = X_train[:, 3]
# X_train = np.column_stack((X_train[:, 0], X_train[:, 2], X_train[:, 3]))
# # tacle x_test
# X_test_0 = X_test[:, 0]
# X_test_1 = X_test[:, 1]
# X_test_2 = X_test[:, 2]
# X_test_3 = X_test[:, 3]
# X_test = np.column_stack((X_test[:, 0], X_test[:, 2], X_test[:, 3]))

decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
randomforest2 = RandomForestClassifier(random_state=0, n_jobs=-1)

one_tree_from_forest = SelectFromModel(randomforest2, threshold=0.3)
features_importance = one_tree_from_forest.fit_transform(X_train, y_train)

decisiontree_model = decisiontree.fit(X_train, y_train)
randomforest_model = randomforest.fit(X_train, y_train)
randomforest_model2 = randomforest2.fit(X_train, y_train)

# evaluate the model
y_predict_decisiontree = decisiontree_model.predict(X_test)
y_predict_randomforest = randomforest_model.predict(X_test)
y_predict_randomforest2 = randomforest_model2.predict(X_test)

print(accuracy_score(y_test, y_predict_decisiontree))
print(accuracy_score(y_test, y_predict_randomforest))
print(accuracy_score(y_test, y_predict_randomforest2))
print(confusion_matrix(y_test, y_predict_decisiontree))
metrics.ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict_decisiontree))
importance = decisiontree_model.feature_importances_
importance_index = np.argsort(importance)[::-1]
iris_feature_names = iris.feature_names

print("original feature names:", iris_feature_names)
for i in importance_index:
    print(iris_feature_names[i], importance[i])

    # delete the repeated number in the list


def delete_repeat(list):
    new_list = []
    for i in list:
        if i not in new_list:
            new_list.append(i)
    return new_list


# get the statistics of the data in one list, the number of each number in the list
def get_list_statistics(list):
    statistics = {}
    for i in list:
        if i in statistics:
            statistics[i] += 1
        else:
            statistics[i] = 1
    return statistics


y_metadata = get_list_statistics(y_test)
print(y_metadata)


#%% random forest
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
# use train_test_split to split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
randomforest = RandomForestClassifier(
    random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1,
)
# get the statistics of the data in one list, the number of each number in the list
def get_list_statistics(list):
    statistics = {}
    for i in list:
        if i in statistics:
            statistics[i] += 1
        else:
            statistics[i] = 1
    return statistics

feature = X_train[40:, :]
target = y_train[40:]
get_list_statistics(list(target))
target = np.where((target == 0), 0, 1)
get_list_statistics(list(target))
randomforest.fit(feature, target)
randomforest.oob_score_
