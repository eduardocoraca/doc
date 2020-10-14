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
