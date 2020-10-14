class HMM_AR():
    def __init__(self, M, N):
        self.N = N
        self.M = M

    def fit(self, X_train, num_states, num_mix, window_size, overlap, intervals=3):
        # X_train: list of time signals. Each component corresponds to a HMM
        assert len(X_train)==self.N
        self.models = []

        for k in range(self.N):
            print('Training HMM',k, '...')
            ar = self.get_ar(X_train[k])
            x_fit = np.zeros((0,self.M))
            len_fit = []
            D = ar.shape[0]//intervals
            for d in range(intervals):
              x_fit = np.concatenate((x_fit,ar[d*D:(d+1)*D,:]), axis=0)
              len_fit.append(D)
            hmm = GMMHMM(n_components=num_states, n_mix=num_mix, n_iter=100, verbose=False, init_params='smcw', tol=1e-3)
            hmm.fit(x_fit, len_fit)
            self.models.append(hmm)
            del hmm

    def predict(self, X):
        # X: array of shape [num_samples, windowed_time_signal]
        # M: number of AR coefficients
        # ar: array of shape [num_samples, M]
        print('Extracting AR coefficients ...')
        ar = self.get_ar(X)
        num_samples = X.shape[0]
        predictions = np.zeros((num_samples, self.N))

        print('Calculating probabilities ...')
        for i in range(num_samples):
            for j in range(self.N):
                predictions[i,j] = self.models[j].score(ar[i,:].reshape(1,-1))
            predictions[i,:] = scipy.special.softmax(predictions[i,:])
        return predictions

    def get_ar(self, X):
        a = []
        for k in range(X.shape[0]):
            mod = AR(X[k,:])
            fitmod = mod.fit(self.M-1)
            a.append(fitmod.params)
        return np.array(a)
