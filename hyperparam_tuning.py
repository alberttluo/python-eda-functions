def find_best_params(cl, df, model, trial):
  if cl == 'logistic_regression':
    cff = LogisticRegression()
    parameters = [{'penalty':['l2', None]},
            {'C':[1, 10, 100, 1000]}, {'max_iter' : [i for i in range(100000, 500000, 10000)]}]
  elif cl == 'decision_tree':
    cff = DecisionTreeClassifier()
    parameters = [{'max_depth':[10, 100, 300, 500, 1000, 5000, 10000]}, {'min_samples_split':[10, 50, 200, 400, 1000]},
     {'min_samples_leaf':[10, 100, 300, 500, 1000, 5000, 10000]}, {'criterion':['gini', 'entropy', 'log_loss']},
      {'max_features':[10, 100, 300, 500, 1000, 5000, 10000]}, {'max_leaf_nodes': [10, 100, 300, 500, 1000, 5000, 10000]},
       {'min_impurity_decrease': [10, 100, 300, 500, 1000, 5000, 10000]}]
  else:
    raise ValueError("cl must be either 'logistic_regression' or 'decision_tree'.")

  if model == 'random_search':
    opt = RandomizedSearchCV(estimator=cff, scoring='f1', param_distributions=parameters).fit(x, y)
  elif model == 'grid_search':
    opt = GridSearchCV(estimator=cff, scoring='f1', param_grid=parameters).fit(x, y)
  else:
    raise ValueError("Model type must be either 'random_search' or 'grid_search'.")

  return(opt.best_estimator_.get_params())


out = find_best_params('logistic_regression', DF, 'grid_search')
print(out, "\n\n")

classifier = 'logistic_regression'

f1_list, f1_mean, f1_std = my_classifier(x, np.ravel(y), classifier, n_folds=10)
print("ITERATIONS: ", f1_list, "\n\n", "MEAN ACCURACY: ", f1_mean, "\n\n", "STANDARD DEVIATION: ", f1_std, "\n\n")
