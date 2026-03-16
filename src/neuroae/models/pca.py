import numpy as np
from sklearn.decomposition import PCA as sklearn_PCA

from ..load_data import ADNIDataset

class PCA(sklearn_PCA):
    def __init__(self, dataset:ADNIDataset, n_components = None, *, copy = True, whiten = False, svd_solver = "auto", tol = 0, iterated_power = "auto", n_oversamples = 10, power_iteration_normalizer = "auto", random_state = None):
        if dataset.preserve_timepoints:
            n_components = n_components // dataset.timepoint_dim

        super().__init__(n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer, random_state=random_state)
        self.dataset = dataset
        # timepoint dimension
        self.T = None
        # region dimension
        self.R = None

        print(f"PCA: {n_components=}")

    def _pre_X(self, X):
        if self.dataset.timepoints_as_samples: # it is already in required shape
            return X
        if self.dataset.fc_input:   # it doesn't make sense to reshape as the input is not 2d
            return X
        B = X.shape[0]
        # making sure the data shape is (B x T, R) to reduce only the regions to latent space
        if self.dataset.flatten:
            # revert flattening
            X = X.reshape(B, self.dataset.original_shape[0], self.dataset.original_shape[1])
        if not self.dataset.transpose:
            # transpose if not already transposed
            X = X.transpose(0, 2, 1)
        if self.T is None:
            self.T = X.shape[1]
        if self.R is None:
            self.R = X.shape[-1]
        # from (B, T, R) to (B x T, R)
        X = X.reshape(-1, X.shape[-1])
        return X
    
    def _post_Z(self, Z):
        if self.dataset.timepoints_as_samples: # it is already in required shape
            return Z
        if self.dataset.fc_input:   # it doesn't make sense to reshape as the input is not 2d
            return Z
        L = self.n_components   # latent_dim
        # adjust the space back to reflect the output of the models
        # from (B x T, L) to (B, T, L)
        Z = Z.reshape(-1, self.T, L)
        B = Z.shape[0]
        if not self.dataset.transpose:
            Z = Z.transpose(0, 2, 1)    # to (B, L, T)
        if self.dataset.flatten:
            Z = Z.reshape(B, -1)
        return Z
    
    def _pre_Z(self, Z): # reverse post_Z for inverse transform
        if self.dataset.timepoints_as_samples: # it is already in required shape
            return Z
        if self.dataset.fc_input:   # it doesn't make sense to reshape as the input is not 2d
            return Z
        # Z shape comes in (B, T x L), where T x L order is unknown
        B = Z.shape[0]
        if self.dataset.flatten:
            if self.dataset.transpose: # reshape from (B, T x L) to (B, T, L)
                Z = Z.reshape(B, -1, self.n_components)
            else:                      # reshape from (B, L x T) to (B, L, T)
                Z = Z.reshape(B, self.n_components, -1)
        if not self.dataset.transpose:
            Z = Z.transpose(0, 2, 1)
        Z = Z.reshape(-1, self.n_components)
        # shape output should be (B x T, L)
        return Z
    
    def _post_X(self, X): # reverse pre_X for inverse transform
        if self.dataset.timepoints_as_samples: # it is already in required shape
            return X
        if self.dataset.fc_input:   # it doesn't make sense to reshape as the input is not 2d
            return X
        # X shape comes in (B x T, R)
        R = X.shape[-1]
        X = X.reshape(-1, self.T, R)
        B = X.shape[0]
        if not self.dataset.transpose:
            X = X.transpose(0, 2, 1)    # to (B, L, T)
        if self.dataset.flatten:
            X = X.reshape(B, -1)
        return X

    def fit(self, X, y = None):
        X = self._pre_X(X)
        return super().fit(X, y)
    
    def transform(self, X):
        X = self._pre_X(X)
        Z = super().transform(X)
        Z = self._post_Z(Z)
        return Z
    
    def inverse_transform(self, Z):
        Z = self._pre_Z(Z)
        X = super().inverse_transform(Z)
        X = self._post_X(X)
        return X
    

class PCA_multi:
    """ Have a separate PCA trained for every timepoint """
    def __init__(self, dataset, n_components, random_state=None):
        self.dataset = dataset
        # timepoint dimension
        self.T = dataset.timepoint_dim
        # region dimension
        self.R = None

        self.n_components = n_components // self.T

        self.pcas = [sklearn_PCA(self.n_components, random_state=random_state) for _ in range(self.T)]
        print(f"PCA {self.n_components=} {self.T=}")


    def _pre_X(self, X):
        B = X.shape[0]
        # making sure the data shape is (B x T, R) to reduce only the regions to latent space
        if self.dataset.flatten:
            # revert flattening
            X = X.reshape(B, self.dataset.original_shape[0], self.dataset.original_shape[1])
        if not self.dataset.transpose:
            # transpose if not already transposed
            X = X.transpose(0, 2, 1)
        if self.T is None:
            self.T = X.shape[1]
        if self.R is None:
            self.R = X.shape[-1]
        # from (B, T, R) to (T, B, R)
        X = X.transpose(1, 0, 2)
        return X
    
    def _post_Z(self, Z):
        # adjust the space back to reflect the output of the models
        # from (T, B, L) to (B, T, L)
        Z = Z.transpose(1, 0, 2)
        B = Z.shape[0]
        if not self.dataset.transpose:
            Z = Z.transpose(0, 2, 1)    # to (B, L, T)
        if self.dataset.flatten:
            Z = Z.reshape(B, -1)
        return Z

    def _pre_Z(self, Z): # reverse post_Z for inverse transform
        # Z shape comes in (B, T x L), where T x L order is unknown
        B = Z.shape[0]
        if self.dataset.flatten:
            if self.dataset.transpose: # reshape from (B, T x L) to (B, T, L)
                Z = Z.reshape(B, -1, self.n_components)
            else:                      # reshape from (B, L x T) to (B, L, T)
                Z = Z.reshape(B, self.n_components, -1)
        if not self.dataset.transpose: # match (B, T, L)
            Z = Z.transpose(0, 2, 1)
        # from (B, T, L) to (T, B, L)
        Z = Z.transpose(1, 0, 2)
        return Z

    def _post_X(self, X): # reverse pre_X for inverse transform
        # from (T, B, L) to (B, T, L)
        X = X.transpose(1, 0, 2)
        if not self.dataset.transpose:
            X = X.transpose(0, 2, 1)        # to (B, L, T)
        if self.dataset.flatten:
            X = X.reshape(X.shape[0], -1)   # to (B, L x T) or (B, T x L)
        return X

    def fit(self, X):
        X = self._pre_X(X)
        [self.pcas[t].fit(X[t,:,:]) for t in range(self.T)]
    
    def transform(self, X):
        X = self._pre_X(X)
        Z = [self.pcas[t].transform(X[t,:,:]) for t in range(self.T)]
        Z = np.stack(Z)
        Z = self._post_Z(Z)
        return Z
    
    def inverse_transform(self, Z):
        Z = self._pre_Z(Z)
        X = [self.pcas[t].inverse_transform(Z[t,:,:]) for t in range(self.T)]
        X = np.stack(X)
        X = self._post_X(X)
        return X
