def extract_ar(x,M,N,k):
  # M: no. of AR coefficients
  # N: window size
  # x: signal
  # k: starting time index
  X = []
  y = []
  for n in range(N):
    y.append(x[k+n])
    row = []
    for m in range(1,M+1):
      row.append(x[k+n-m])
    X.append(row)
  X = np.array(X).reshape((N,M))
  y = np.array(y).reshape((N,1))

  return (np.matmul(scipy.linalg.inv(np.matmul(X.T,X)), X.T),y)