def objRF(trial):
  #n_estimators; max_depth; min_samples_split;
  #min_samples_leaf; criterion; max_features
  paramRF = {
    'n_estimators' : trial.suggest_int('n_estimators', 2, 20),
    'max_depth' : (trial.suggest_int('max_depth', 1, 32)),
    'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 2, 20),
    'max_features' : trial.suggest_int('max_features', 3, 20)
  }

  mod = RandomForestClassifier(**paramRF)
  return cross_val_score(mod, DF, DF['ACTION'], 
           cv=10).mean()

def get_RF_params():
  study = optuna.create_study(direction="maximize")
  study.optimize(objRF, n_trials=10)
  return study.best_trial.params


def objSVM(trial):
  paramSVM = {
    'C' : trial.suggest_int('C', 1, 10),
    'kernel' : trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
  }

  mod = SVC(**paramSVM)
  return cross_val_score(mod, DF, DF['ACTION'], 
           cv=10).mean()

def get_SVM_params():
  study = optuna.create_study(direction="maximize")
  study.optimize(objSVM, n_trials=5)
  return study.best_trial.params

def objXGBoost(trial):
  #XGBoost: booster; num_features; eta; gamma; max_depth
  paramXG = {
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
      "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
      "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0)
  }

  mod = XGBClassifier(**paramXG)
  return cross_val_score(mod, DF, DF['ACTION'], 
           cv=10).mean()

def get_XGBoost_params():
  study = optuna.create_study(direction="maximize")
  study.optimize(objXGBoost, n_trials=5)
  return study.best_trial.params

#---------------------------

def classify(x, y, classifier, k_folds=10):
  if classifier == 'SVM':
    clf = SVC(**get_SVM_params()).fit(x, y)
  elif classifier == 'Random Forest':
    clf = RandomForestClassifier(**get_RF_params()).fit(x, y)
  elif classifier == 'XGBoost':
    clf = XGBClassifier(**get_XGBoost_params()).fit(x, y)

  f1_scores = cross_val_score(clf, x, y, cv=k_folds)
  mean_f1_score = np.mean(f1_scores)
  std_f1_score = np.std(f1_scores)
  ras = roc_auc_score(y, clf.predict_proba(x)[:, 1])

  return f1_scores, mean_f1_score, std_f1_score, ras
