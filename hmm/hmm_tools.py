import scipy.special
from statsmodels.tsa.ar_model import AR
import numpy as np

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

        