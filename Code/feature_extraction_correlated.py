# Load libraries
import numpy as np
import pandas as pd

# Create feature matrix with two highly correlated features
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1],
                     [3, 1, 2],
                     [4, 2, 4],
                     [5, 3, 6],
                     [6, 4, 8],
                     [7, 5, 10],
                     [8, 6, 12],
                     [9, 7, 14],
                     [3, 8, 16],
                     [4, 9, 18],
                     [5, 10, 20],

                     ])

# Convert feature matrix into DataFrame
dataframe = pd.DataFrame(features)
# Create correlation matrix
corr_matrix = dataframe.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)

print("hello")
