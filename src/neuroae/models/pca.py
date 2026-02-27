from sklearn.decomposition import PCA as sklearn_PCA

class PCA(sklearn_PCA):
    def __init__(self, dataset, n_components = None, *, copy = True, whiten = False, svd_solver = "auto", tol = 0, iterated_power = "auto", n_oversamples = 10, power_iteration_normalizer = "auto", random_state = None):
        super().__init__(n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer, random_state=random_state)
        self.dataset = dataset
        # timepoint dimension
        self.T = None
        # region dimension
        self.R = None

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
        # from (B, T, R) to (B x T, R)
        X = X.reshape(-1, X.shape[-1])
        return X
    
    def _post_Z(self, Z):
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