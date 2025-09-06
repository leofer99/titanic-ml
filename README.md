# Titanic ML Pipeline

This project provides a **machine learning pipeline** for predicting passenger survival on the Titanic dataset. It performs data cleaning, feature engineering, exploratory analysis, model training, evaluation, and hyperparameter tuning.

---

## 📂 Project Structure

```
├── main.py              # Entry point for running the pipeline
├── pipeline.py          # Core functions (data loading, preprocessing, training, evaluation, tuning)
├── train.csv            # Titanic training dataset
├── test.csv             # Titanic test dataset (optional, otherwise train is split)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Features

* **Data Preprocessing**

  * Handles missing values (Age, Fare, Embarked, Cabin)
  * Encodes categorical features (Sex, Embarked, Titles)
  * Engineers new features (Relatives, Deck, Title)
  * Drops irrelevant features (PassengerId, Ticket, etc.)

* **Model Training**

  * Logistic Regression
  * Naive Bayes
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest

* **Model Evaluation**

  * Accuracy, Precision, Recall, F1-score, MCC
  * Confusion matrices
  * Cross-validation scores

* **Hyperparameter Tuning**

  * GridSearchCV for Logistic Regression and Random Forest

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd titanic-ml-pipeline
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n titanic python=3.11 -y
conda activate titanic
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the pipeline:

```bash
python main.py
```

Expected output:

* Metrics table with Accuracy, Precision, Recall, F1, MCC
* Best hyperparameters for Logistic Regression and Random Forest

---

## 📊 Example Output

```
              Model  Accuracy  Precision  Recall    F1    MCC
0  Logistic Regression    0.80       0.78     0.74   0.76   0.59
1           Naive Bayes    0.77       0.70     0.72   0.71   0.53
2                   KNN    0.79       0.75     0.76   0.75   0.57
3          Decision Tree    0.75       0.72     0.71   0.71   0.49
4         Random Forest    0.82       0.80     0.78   0.79   0.62

Best Hyperparameters:
{'Logistic Regression': {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'},
 'Random Forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}}
```

---

## 🧪 Datasets

The pipeline expects Titanic data in CSV format:

* `train.csv` – training data with survival labels
* `test.csv` – test data (optional). If not provided, the script will split `train.csv` automatically.

Download Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data).

