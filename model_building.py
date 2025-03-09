import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import missingno as msno
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency
from statsmodels.imputation.mice import MICEData
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import miceforest as mf
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, precision_recall_curve, auc # plot_roc_curve also it errored TODO
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
#TODO copy pasted packages for ease; remove unused when complete

### Read in data
# Train
X_path = Path(__file__).parent / "X_training_data.csv"
y_path = Path(__file__).parent / "y_training_data.csv"
X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path).squeeze() # Convert to 1D array
# Test
X_path = Path(__file__).parent / "X_testing_data.csv"
y_path = Path(__file__).parent / "y_testing_data.csv"
X_test = pd.read_csv(X_path)
y_test = pd.read_csv(y_path).squeeze() # Convert to 1D array

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

### EVALUATE DIFFERENT MODELS ##########################################################################################
# Bool to skip investigate
Show_inves = True

if Show_inves:
    # Initialise dict
    model_scores = {}

    def basic_train(model, X_train, y_train, identifier, dict):
        model.fit(X_train, y_train) # Fit
        y_pred = model.predict(X_train) # Predict on training
        f1_train = f1_score(y_train, y_pred) # F1 training score

        # Accuracy scores
        accuracy_train = accuracy_score(y_train, y_pred)

        # Cross validation with F!
        f1_val = cross_val_score(model, X_train, y_train, scoring='f1', cv=10)
        # Cross validation for accuracy
        accuracy_val = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)

        # Return scores
        dict[identifier] = [identifier, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]

    # Logistic Regression
    log_reg = LogisticRegression(penalty='l1',solver='liblinear')
    basic_train(log_reg,X_train,y_train,'Logistic Regression',model_scores)

    # SVM
    svc_clf = SVC()
    basic_train(svc_clf,X_train,y_train,'Support Vector Classifier',model_scores)

    # Random Forest
    rnd_clf = RandomForestClassifier(random_state=42)
    basic_train(rnd_clf,X_train,y_train,'RandomForestClassifier',model_scores)

    # AdaBoost
    dt_clf_ada = DecisionTreeClassifier()
    Ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42)
    basic_train(Ada_clf,X_train,y_train,"AdaBoost Classifier",model_scores)

    # GradientBoost
    gdb_clf = GradientBoostingClassifier(random_state=42,subsample=0.8)
    basic_train(gdb_clf,X_train,y_train,"GradientBoosting Classifier",model_scores)

    # XGBoost
    xgb_clf = XGBClassifier(verbosity=0)
    basic_train(xgb_clf,X_train,y_train,"XGBoost Classifier",model_scores)

    # KNN
    knn_clf = KNeighborsClassifier()
    basic_train(knn_clf, X_train, y_train, 'K-Nearest Neighbors Classifier', model_scores)

    # Make dataframe of model scores
    scores = pd.DataFrame.from_dict(model_scores, orient='index',
                                    columns=['Model', 'Train F1', 'Test F1', 'Test Accuracy',
                                             'Train Accuracy']).reset_index(drop=True).sort_values(by='Test F1', ascending=False)
    # View results
    print(scores.head(len(scores)))
    # Result:
    #                             Model  Train F1   Test F1  Test Accuracy  Train Accuracy
    # 2          RandomForestClassifier  1.000000  0.749382          1.000   0.740
    # 4     GradientBoosting Classifier  0.853565  0.749083          0.848   0.737
    # 1       Support Vector Classifier  0.854651  0.746034          0.850   0.739
    # 0             Logistic Regression  0.764591  0.745587          0.758   0.741
    # 5              XGBoost Classifier  1.000000  0.716764          1.000   0.713
    # 6  K-Nearest Neighbors Classifier  0.774194  0.646666          0.783   0.672
    # 3             AdaBoost Classifier  1.000000  0.639224          1.000   0.643

    # So here RF, GB, LR, and SVM perform the best on the test, with RF, XGB, and ADB on the train.
    # TODO Typically I would proceed with RF, GB, SVM, and LR. However, for the purpose of practicing
    #  XGBoost I will proceed with that model alongside some inital RF, GB, & SVM comparison

    # TODO Return to this model evaluation above after I have further refined the model development process

    # TODO: Question: Compared to two others that use XGBoost, my model is worse! I think this might mainly
    #  be due to the encoding which is the main thing I did differently. My approach is technically correct
    #  for nominal encoding, but performs worse. If you found models perform better with the 'wrong' encoding,
    #  would you use the wrong one? Related note: Could it be because there are some ordinal categories, so even
    #  though they (I think) should technically be nominal, could the model perform better by picking up on that
    #  real ordinality despite some values ("Don't know") not fitting into the order.

### HYPEROPT PARAMETER TUNING ##########################################################################################

# Hyperopt - initialise domain space (range of values to explore hyperparameters) TODO double check values/params further for each
XGBspace={
    'max_depth': hp.quniform("max_depth", 3, 15, 1),
    'gamma': hp.uniform ('gamma', 0,9),
    'reg_alpha' : hp.quniform('reg_alpha', 0,10),
    'reg_lambda' : hp.uniform('reg_lambda', 0,10),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.6,1),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 12, 1),
    'n_estimators': ('n_estimators', 100, 500, 50),
    'seed': 0,
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10),  # Adjust if classes are imbalanced
    }

RFspace = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
    'max_depth': hp.quniform('max_depth', 5, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.8]),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'class_weight': hp.choice('class_weight', [None, 'balanced'])
    }

GBspace = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.8]),
    'loss': hp.choice('loss', ['deviance', 'exponential'])
}

SVMspace = {
    'C': hp.loguniform('C', np.log(0.001), np.log(1000)),
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'gamma': hp.choice('gamma', ['scale', 'auto', hp.loguniform('gamma_custom', np.log(1e-5), np.log(1))]),
    'class_weight': hp.choice('class_weight', [None, 'balanced'])
}

# Models to investigate and their hyperparameter ranges
models = {
    XGBClassifier : XGBspace,
    RandomForestClassifier : RFspace,
    GradientBoostingClassifier : GBspace,
    SVC : SVMspace
    }
# TODO below is a work in progress and won't yet run
# Define objective space for hyperopt
def objective(space):

    clfmodel = model(
       n_estimators=space['n_estimators'],
       max_depth=int(space['max_depth']),
       gamma=space['gamma'],
       reg_alpha=int(space['reg_alpha']),
       min_child_weight=int(space['min_child_weight']),
       colsample_bytree=space['colsample_bytree'],
       objective = 'reg:squarederror',
       learning_rate = space['learning_rate'])

    evaluation = [(X_train, y_train), (X_test, y_test)]
    # Fit the model with training data
    clfmodel.fit(X_train, y_train,
           eval_set=evaluation, eval_metric="rmse", #todo class
           early_stopping_rounds=10, verbose=False)
    # Predict with test data
    pred = clfmodel.predict(X_test)
    # Return metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return {'loss': -accuracy, 'status': STATUS_OK} z# TODO

for model in models:
    # Choose optimisation algorithm for hyperopt (TPE)
    trials = Trials()
    best_hyperparams = fmin(fn = objective,
                            space = models[model],
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)

    # Print best hyperparameters TODO maybe remove
    print(f"The optimised hyperparameters for {model} are: ","\n")
    print(best_hyperparams)








