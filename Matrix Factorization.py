from sklearn.decomposition import NMF
import numpy as np


X = np.load('rating_matrix.npy')
print(X)
less_than_zero = X<0
X = X+less_than_zero
print(X)
model = NMF()
W = model.fit_transform(X)
print(W)
