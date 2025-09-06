# pipeline.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, matthews_corrcoef)


# --------------------- Data Loading ---------------------
def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    if test_path:
        test_df = pd.read_csv(test_path)
    else:
        train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)
    return train_df, test_df


# --------------------- Preprocessing ---------------------
def preprocess_data(train_df, test_df):
    # Handle missing values, map categorical features, engineer new ones
    # Example: Age, Fare, Embarked, Sex, Cabin/Deck, Titles, relatives
    # Drop irrelevant columns (PassengerId, Ticket, Name)
    # Return clean X_train, X_test, y_train, y_test
    data = [train_df, test_df]


    # Fill missing Embarked
    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S').map({"S": 0, "C": 1, "Q": 2})


    # Age imputation + binning
    for dataset in data:
        mean, std = train_df["Age"].mean(), test_df["Age"].std()
        n_missing = dataset["Age"].isnull().sum()
        rand_ages = np.random.randint(mean - std, mean + std, size=n_missing)
        dataset.loc[dataset["Age"].isnull(), "Age"] = rand_ages
        dataset["Age"] = dataset["Age"].astype(int)
        dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[dataset['Age'] > 66, 'Age'] = 7


    # Fare imputation + binning
    for dataset in data:
        dataset['Fare'] = dataset['Fare'].fillna(0).astype(int)
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
        dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5


    # Sex
    for dataset in data:
        dataset['Sex'] = dataset['Sex'].map({"male": 0, "female": 1})


    # Relatives & Not Alone
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset['not_alone'] = (dataset['relatives'] == 0).astype(int)


    # Deck from Cabin
    deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck_map).fillna(0).astype(int)


    # Title extraction
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in data:
        dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr',
        'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        dataset['Title'] = dataset['Title'].map(title_map).fillna(0).astype(int)


    # Drop unused features
    drop_cols = ['PassengerId', 'Cabin', 'Name', 'Ticket', 'Parch', 'not_alone']
    train_df = train_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)


    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']


    return X_train, X_test, y_train, y_test

# --------------------- Training ---------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# --------------------- Evaluation ---------------------
def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        })
    return pd.DataFrame(results)

def tune_hyperparameters(X_train, y_train):
    tuned = {}


    param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
    }
    logreg = GridSearchCV(LogisticRegression(max_iter=200), param_grid_logreg, cv=5, scoring='accuracy')
    logreg.fit(X_train, y_train)
    tuned['Logistic Regression'] = logreg.best_params_


    param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
    }
    rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
    rf.fit(X_train, y_train)
    tuned['Random Forest'] = rf.best_params_


    return tuned