# Load libraries
from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Load the data
digits = datasets.load_digits()
# Standardize feature matrix [1797, 64]
features = StandardScaler().fit_transform(digits.data)
# Make sparse matrix
features_sparse = csr_matrix(features)
# Create a TSVD
tsvd = TruncatedSVD(n_components=10)
# [1797, 10]
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
# Show results
print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])
