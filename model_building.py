import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, precision_recall_curve, auc # plot_roc_curve also it errored TODO
# TODO (above) - was i going to implement more features using the above? check my other code (diamonds dataset might have used them)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow

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
# Bool to skip investigation
Show_inves = True
if Show_inves:
    # Initialise dict
    model_scores = {}

    # Basic model training function to get some initial scores and decide which model to proceed with
    # IMPROVE add more model types
    def basic_train(model, X_train, y_train, identifier, dict):
        model.fit(X_train, y_train) # Fit
        y_pred = model.predict(X_train) # Predict on training
        f1_train = f1_score(y_train, y_pred) # F1 training score

        # Accuracy scores
        accuracy_train = accuracy_score(y_train, y_pred)

        # Cross validation with F1
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
    # TODO Return to this model evaluation above after I have further refined the model development process

    # TODO: Question: Compared to two others that use XGBoost, my model is worse! I think this might mainly
    #  be due to the encoding which is the main thing I did differently. My approach is technically correct
    #  for nominal encoding, but performs worse. If you found models perform better with the 'wrong' encoding,
    #  would you use the wrong one? Related note: Could it be because there are some ordinal categories, so even
    #  though they (I think) should technically be nominal, could the model perform better by picking up on that
    #  real ordinality despite some values ("Don't know") not fitting into the order.

### HYPEROPT PARAMETER TUNING ##########################################################################################
# For selected models, define a parameter params['type'] for the model name. Then evaluates parameters and calculates the cross-validated accuracy.

# Dictionary to store the best model accuracies
best_accuracies = {
    'svm': 0.0,
    'rf': 0.0,
    'logreg': 0.0,
    'xgb': 0.0,
    'gb': 0.0
}

#Objective function; which parameter configuation is used
def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        # Convert parameters that must be integers (hyperopt returns floats)
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    elif classifier_type == 'xgb':
        # Convert parameters that must be integers (hyperopt returns floats)
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])

        clf = XGBClassifier(**params)
    elif classifier_type == 'gb':
        # Convert parameters that must be integers (hyperopt returns floats)
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])

        clf = GradientBoostingClassifier(**params)
    else:
        return 0
    # Calculate model accuracy
    accuracy = cross_val_score(clf, X_train, y_train, cv=10).mean()

    #  Track the best accuracy per model type
    if accuracy > best_accuracies[classifier_type]:
        best_accuracies[classifier_type] = accuracy
        # Log the new best accuracy for this model type
        mlflow.log_metric(f"best_{classifier_type}_accuracy", accuracy)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# Define the search space over hyperparameters #TODO find more practical examples where these are defined; what is worth definining and what ranges?
search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('svm_kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy']),
        'n_estimators': hp.quniform('rf_n_estimators', 50, 500, 50),
        'max_depth': hp.quniform('rf_max_depth', 2, 10, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', 0.8]),
        'class_weight': hp.choice('rf_class_weight', [None, 'balanced'])
    },
    {
        'type': 'logreg',
        'C': hp.lognormal('lr_C', 0, 1.0),
        'solver': hp.choice('lr_solver', ['liblinear', 'lbfgs'])
    },
    {
        'type': 'xgb',
        'max_depth': hp.quniform("xgb_max_depth", 3, 15, 1),
        'gamma': hp.uniform ('xgb_gamma', 0,9),
        'reg_alpha' : hp.quniform('xgb_reg_alpha', 0,10,1),
        'reg_lambda' : hp.uniform('xgb_reg_lambda', 0,10),
        'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.6,1),
        'min_child_weight' : hp.quniform('xgb_min_child_weight', 0, 12, 1),
        'n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
        'seed': 0,
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'scale_pos_weight': hp.uniform('xgb_scale_pos_weight', 1, 10),  # Adjust if classes are imbalanced
    },
    {
        'type': 'gb',
        'n_estimators': hp.quniform('gb_n_estimators', 50, 500, 50),
        'max_depth': hp.quniform('gb_max_depth', 3, 15, 1),
        'min_samples_split': hp.quniform('gb_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('gb_min_samples_leaf', 1, 10, 1),
        'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.005), np.log(0.2)),
        'subsample': hp.uniform('gb_subsample', 0.6, 1.0),
        'max_features': hp.choice('gb_max_features', ['sqrt', 'log2', 0.8]),
        'loss': hp.choice('gb_loss', ['log_loss', 'exponential'])
    },

])

# Use fmin() to tune hyperparameters
print("Now tuning hyperparameters \n")
with mlflow.start_run():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100, #IMPROVE Change this to ~10 if testing further
        trials=Trials())

# Print the best accuracies for each model
print("\nHighest model accuracies on train data:")
best_accuracy_df = pd.DataFrame(list(best_accuracies.items()), columns=['Models', 'Highest accuracy'])
print(best_accuracy_df)

# Extract the best set of hyperparameters and print
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ##################################################################################################
# Train final model and fit to test data
with mlflow.start_run(): #TODO need to find examples of this being done - unsure on the final training/testing after hyperopt tuning
    # Extract classifier type and parameters
    classifier_type = best_config['type']
    best_params = {k: v for k, v in best_config.items() if k != 'type'}

    # Log the best hyperparameters
    mlflow.log_params(best_config)

    # Construct the best model
    if classifier_type == 'svm':
        best_model = SVC(**best_params)
    elif classifier_type == 'rf':
        best_model = RandomForestClassifier(**best_params)
    elif classifier_type == 'logreg':
        best_model = LogisticRegression(**best_params)
    elif classifier_type == 'xgb':
        best_model = XGBClassifier(**best_params)
    elif classifier_type == 'gb':
        best_model = GradientBoostingClassifier(**best_params)

    # Train on full training data
    best_model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(best_model, "best_model")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")


# TODO: Question: If I get different models (LR and GB currently) on different runs, what should I do? Pick one? Use the
#  most common of multiple attempts?

# TODO: Test with ordinal encoding for some of the categories I used nominal
#  Plot feature importance - could potentially remove unimportant features and retest
#  Generate a learning curve

# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement
#  Could also do an ensemble model approach for the final training, and stacking/voting
#  Examine errors to see if I can identify where I'm making mistakes