import scipy.special
from statsmodels.tsa.ar_model import AR
import numpy as np
from hmmlearn.hmm import *

def split_signal(signal, window_size, overlap):
  # window_size: length of window
  # overlap: % of overlap
  signal = signal.reshape(-1)
  X = []
  o = round(overlap*window_size)
  N = len(signal)
  for k in range(0,N-window_size, window_size-o):
    X.append(signal[k:k+window_size])

  return (np.array(X))

class HMM_AR():
    def __init__(self, M, N):
        self.N = N
        self.M = M

    def fit(self, X_train, num_states, num_mix):
        # X_train: list of time signals. Each component corresponds to a HMM
        assert len(X_train)==self.N
        self.models = []
        for k in range(self.N):
            hmm = GMMHMM(n_components=num_states, n_mix=num_mix, n_iter=100, verbose=False, init_params='smcw', tol=1e-3)
            hmm.fit(X_train[k])
            self.models.append(hmm)

    def predict(self, X):
        # X: array of shape [num_samples, windowed_time_signal]
        # M: number of AR coefficients
        # ar: array of shape [num_samples, M]
        ar = self.get_ar(X, self.M)

        num_samples = X.shape[0]
        predictions = np.zeros((num_samples, self.N))

        for i in range(num_samples):
            for j in range(self.N):
                predictions[i,j] = self.models[j].score(X[i,:].reshape(-1,1))
        return predictions

    def get_ar(self, X):
        a = []
        for k in range(X.shape[0]):
            mod = AR(X[k,:])
            fitmod = mod.fit(M-1)
            a.append(fitmod.params)
        return np.array(a)

        
