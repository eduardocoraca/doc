import scipy.special
from statsmodels.tsa.ar_model import AR
import numpy as np

class hmm_ar():
    def __init__(self, hmm_list):
        self.hmm_list = hmm_list
        self.N_hmm = len(hmm_list)

    def predict(self, X, M):
        # X: array of shape [num_samples, windowed_time_signal]
        # M: number of AR coefficients
        # ar: array of shape [num_samples, M]
        ar = self.get_ar(X, M)

        num_samples = X.shape[0]
        predictions = np.zeros((num_samples, self.N_hmm))

        for i in range(num_samples):
            for j in range(self.N_hmm):
                predictions[i,j] = self.hmm_list[j].score(X[i,:].reshape(-1,1))
        return predictions

    def get_ar(self, X, M):
        a = []
        for k in range(X.shape[0]):
            mod = AR(X[k,:])
            fitmod = mod.fit(M-1)
            a.append(fitmod.params)
        return np.array(a)

        