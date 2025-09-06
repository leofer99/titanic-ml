# main.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, matthews_corrcoef)

from pipeline import load_data, preprocess_data, train_models, evaluate_models, tune_hyperparameters

# get path
parent_dir = os.path.abspath((os.path.dirname(__file__)))

# Load
train_path = os.path.join(parent_dir, 'train.csv')
train_df, test_df = load_data(train_path)

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

# Train
models = train_models(X_train, y_train)

# Evaluate
results = evaluate_models(models, X_test, y_test)
print(results)

best_params = tune_hyperparameters(X_train, y_train)
print("\nBest Hyperparameters:")
print(best_params)

