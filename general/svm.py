def train_svmr(x_train, x_test, y_train, y_test):
  # 1) Definindo a busca de hiperparâmetros
  C_range = np.logspace(2, 3, 3)
  gamma_range = np.logspace(-3, -1, 3)
  epsilon_range = np.logspace(-3, -1, 3)
  param_grid = dict(gamma=gamma_range, C=C_range, epsilon=epsilon_range)

  grid = sklearn.model_selection.GridSearchCV(estimator = sklearn.svm.SVR(kernel = 'rbf'),
                                            param_grid = param_grid,
                                            cv = 5,
                                            scoring = 'neg_root_mean_squared_error')

  # 2) Treinando o modelo
  grid.fit(X = x_train, y = y_train)    

  return grid
