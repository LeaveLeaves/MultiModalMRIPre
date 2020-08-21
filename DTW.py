import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import new_img_like, load_img, get_data
from sklearn.model_selection import KFold
from nilearn.image import index_img
import nilearn.decoding
from nilearn.input_data import NiftiMasker
import nilearn as nil
import nibabel as nib

from tslearn.metrics import cdist_dtw
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from tslearn.metrics import cdist_dtw   
from sklearn.neighbors import KNeighborsClassifier

class DTW_KNN(BaseEstimator, ClassifierMixin):
    """ DTW_KNN model in sklearn style (for pipeline transforms)
    """
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
    def dtw_noNan(data1, data2, process):
        entry = dtw(data1, data2)
        if math.isnan(entry):
            entry = dtw(data2, data1)
        print(process)
        return entry
    
    def dtw_matrix(dataset1, verbose = 0, compute_diagonal = False):
        matrix = np.zeros((len(dataset1), len(dataset1)))
        indices = np.triu_indices(len(dataset1), k=0 if compute_diagonal else 1, m=len(dataset1))
        matrix[indices] = Parallel(n_jobs = 10,
                               prefer="threads",
                               verbose = verbose)(
            delayed(dtw_noNan)(
                dataset1[i], dataset1[j],
            )
            for i in range(len(dataset1))
            for j in range(i if compute_diagonal else i + 1,
                          len(dataset1))
        )
        print("up")
        indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        print("train")
        return matrix    
    
    def dtw_2_matrix(data1, data2, verbose = 0, compute_diagonal = False):
        matrix = Parallel(n_jobs = 10,
                      prefer="processes",
                      verbose = verbose)(
            delayed(dtw_noNan)(
                data1[i], data2[j],[i, j]
            )
            for i in range(len(data1)) for j in range(len(data2))
        )
        print('test')
        return np.array(matrix).reshape((len(data1), -1))

    def fit(self, X, y = None):
        #add train data to object for later predictions:
        self.X_train = X
        #Compute DTW distance matrix of train data:
        x_train_dtw = dtw_matrix(X)
        #Fit KNN on train DTW matrix
        self.knn = KNeighborsClassifier(n_neighbors = self.n_neighbors, metric = 'precomputed')
        self.knn.fit(x_train_dtw, y)
        print('fit')
        
    def predict(self, X, y = None):
        x_test_dtw = dtw_2_matrix(X, self.X_train)
        return self.knn.predict(x_test_dtw)
