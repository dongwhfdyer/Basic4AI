from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
feature = boston.data
target = boston.target

interaction = PolynomialFeatures(degree=13, include_bias=False, interaction_only=True)
feature_interaction = interaction.fit_transform(feature)

regression = LinearRegression()

model = regression.fit(feature_interaction, target)

# use model to inference

print("hello")
print("hello")
