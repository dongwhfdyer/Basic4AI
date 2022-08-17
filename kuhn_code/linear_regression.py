# from sklearn.datasets import load_boston
# from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


#
# boston = load_boston()
# feature = boston.data
# target = boston.target
#
# interaction = PolynomialFeatures(degree=13, include_bias=False, interaction_only=True)
# feature_interaction = interaction.fit_transform(feature)
#
# regression = LinearRegression()
#
# model = regression.fit(feature_interaction, target)
#
# # use model to inference
#
# print("hello")
# print("hello")

# x从-3 - 3均匀取值
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
# y是二次方程
y = 2 * x ** 3 + 0.5 * x ** 2 + x + 2 + np.random.normal(0, 3, size=100)

plt.scatter(x, y)
# plt.show()
##################################################

X_ = PolynomialFeatures(degree=3, ).fit_transform(X)

# 实例化线性模型
lr1 = LinearRegression()
lr2 = LinearRegression()

decisiontree = DecisionTreeRegressor()

lr1.fit(X_, y)
lr2.fit(X, y)
decisiontree.fit(X, y)
y_predict_l1 = lr1.predict(X_)
y_predict_decisiontree = decisiontree.predict(X)

plt.scatter(x, y)

plt.plot(np.sort(x), y_predict_l1[np.argsort(x)], label="linear regression")
plt.plot(np.sort(x), y_predict_decisiontree[np.argsort(x)], label="decisiontree")

oo = [[1], [2]]
op = decisiontree.predict(oo)
for dot_x, dot_y in zip(oo, op):
    plt.scatter(dot_x, dot_y, color="red")
print(op)
plt.show()

# %%
