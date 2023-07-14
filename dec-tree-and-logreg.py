# classifier function
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import numpy as np

def my_classifier(x, y, classifier, classifier_args=None, n_folds=10):
  if classifier == 'decision_tree':
    clf = DecisionTreeClassifier(criterion='gini')

  elif classifier == 'logistic_regression':
    clf =  classifier_args[1] if classifier_args else LogisticRegression(max_iter=10000)

  else:
    raise ValueError("Invalid classifier. Please choose 'decision_tree' or 'logistic_regression'.")

  f1_scores = cross_val_score(clf, x, y, cv=n_folds)
  mean_f1_score = np.mean(f1_scores)
  std_f1_score = np.std(f1_scores)

  return f1_scores, mean_f1_score, std_f1_score
