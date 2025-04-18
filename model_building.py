import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow

# TODO produce graphs of metrics like AUC, ROC precision recall curve, etc.

### Read in data
# Train
X_path = Path(__file__).parent / "X_training_data.csv"
y_path = Path(__file__).parent / "y_training_data.csv"
X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / "X_testing_data.csv"
y_path = Path(__file__).parent / "y_testing_data.csv"
X_test = pd.read_csv(X_path)
y_test = pd.read_csv(y_path).squeeze()  # Convert to 1D array

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

### EVALUATE DIFFERENT MODELS ##########################################################################################
# Define a dictionary mapping classifier types to their optimal feature selectors
feature_selectors_dict = { # TODO: AI-gen-ed for quick options. Do research into actual best selection ethods for each model - knn/[and another] had to be replaced with sklearn native
    'svm': RFECV(estimator=LinearSVC(dual="auto", penalty="l1", C=0.1, random_state=42),
                 step=1,
                 cv=StratifiedKFold(3),
                 scoring="f1"
                 ),
    'rf': SelectFromModel(RandomForestClassifier(n_estimators=100,
                                                 max_depth=5,
                                                 random_state=42),
                                                 threshold="median"
                          ),
    'logreg': SelectFromModel(LogisticRegression(penalty="elasticnet",
                                                    solver="saga",
                                                    l1_ratio=0.5,  # Mix of L1/L2
                                                    C=0.1,
                                                    random_state=42
                                                 )
                              ),
    'xgb': RFECV(estimator=XGBClassifier(n_estimators=100, max_depth=3),
                                        step=1,
                                        cv=StratifiedKFold(3),
                                        scoring="f1"
                                        ),
    'gb': SelectFromModel(GradientBoostingClassifier(random_state=42)),
    'knn': RFECV(estimator=RandomForestClassifier(random_state=42),
                 step=1, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                 scoring='f1', min_features_to_select=1),
    'ada': RFECV(estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
                    step=1,
                    cv=StratifiedKFold(3),
                    scoring="f1"
                )
    }
# Bool to skip investigation
Show_inves = True
if Show_inves:
    # Initialise dict
    model_scores = {}


    # Basic model training function to get some initial scores and decide which model to proceed with
    # IMPROVE add more model types?
    def basic_train(model, model_type, X_train, y_train, identifier, dict):
        # Create a pipeline with feature selection and classifier - ensures same CV folds/feature selection
        selector = feature_selectors_dict[model_type]
        pipe = Pipeline([
            ('feature_selector', selector),
            ('classifier', model)
        ])

        # 10-fold cross validation for F1 score and accuracy
        f1_val = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=42))
        accuracy_val = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10, shuffle=True, random_state=42))

        # Fit the pipeline on the training data
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_train)
        f1_train = f1_score(y_train, y_pred)
        accuracy_train = accuracy_score(y_train, y_pred)

        dict[identifier] = [identifier, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]

    # Logistic Regression
    log_reg = LogisticRegression(penalty='l1', solver='liblinear')
    basic_train(log_reg, 'logreg', X_train, y_train, 'Logistic Regression', model_scores)

    # SVM
    svc_clf = SVC()
    basic_train(svc_clf, 'svm', X_train, y_train, 'Support Vector Classifier', model_scores)

    # Random Forest
    rnd_clf = RandomForestClassifier(random_state=42)
    basic_train(rnd_clf, 'rf', X_train, y_train, 'RandomForestClassifier', model_scores)

    # AdaBoost
    dt_clf_ada = DecisionTreeClassifier()
    ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42)
    basic_train(ada_clf, 'ada', X_train, y_train, "AdaBoost Classifier", model_scores)

    # GradientBoosting
    gdb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
    basic_train(gdb_clf, 'gb', X_train, y_train, "GradientBoosting Classifier", model_scores)

    # XGBoost
    xgb_clf = XGBClassifier(verbosity=0)
    basic_train(xgb_clf, 'xgb', X_train, y_train, "XGBoost Classifier", model_scores)

    # KNN
    knn_clf = KNeighborsClassifier()
    basic_train(knn_clf, 'knn', X_train, y_train, 'K-Nearest Neighbors Classifier', model_scores)

    # Make dataframe of model scores and print results
    scores = pd.DataFrame.from_dict(model_scores, orient='index',
                                    columns=['Model', 'Train F1', 'Test F1', 'Train Accuracy',
                                             'Test Accuracy']).reset_index(drop=True).sort_values(by='Test F1',
                                                                                                   ascending=False)
    print(scores.head(len(scores)))
    # Example printed result:
    #                             Model  Train F1   Test F1  Train Accuracy  Test Accuracy
    # 0             Logistic Regression  0.774443  0.760099           0.767   0.756016
    # 5              XGBoost Classifier  0.772569  0.753629           0.738   0.738999
    # 4     GradientBoosting Classifier  0.813102  0.744690           0.806   0.732019
    # 1       Support Vector Classifier  0.835708  0.744277           0.827   0.734012
    # 2          RandomForestClassifier  1.000000  0.740152           1.000   0.734009
    # 3             AdaBoost Classifier  0.758242  0.725148           0.692   0.687013
    # 6  K-Nearest Neighbors Classifier  0.771160  0.663106           0.781   0.674036

    # So here LR, XGB, GB, and SVM perform the best on the test, with RF, SVM, and GB on the train.

    # TODO: Question: Compared to two others that use XGBoost, my model is worse (at least before feature selection)
    #  I think this might mainly be due to the encoding which is the main thing I did differently. My approach is
    #  technically correct for nominal encoding, but performs worse. If you found models perform better with the 'wrong'
    #  encoding, would you use the wrong one? Related note: Could it be because there are some ordinal categories, so even
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


# Objective function; which parameter configuation is used
def objective(params):
    classifier_type = params['type']
    del params['type']
    selector = feature_selectors_dict[classifier_type]

    # Build the classifier based on provided type and convert parameters that must be integers (hyperopt returns floats) if necessary
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    elif classifier_type == 'xgb':
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])
        clf = XGBClassifier(**params)
    elif classifier_type == 'gb':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = GradientBoostingClassifier(**params)
    else:
        return {'loss': 1, 'status': STATUS_OK}

    # Incorporate feature selection into the pipeline
    pipe = Pipeline([
        ('feature_selector', selector),
        ('classifier', clf)
    ])

    # Use 10-fold cross validation to compute the mean accuracy
    accuracy = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1').mean()  # Reduced to 5-fold for speed

    # Log the best accuracy for each model type if improved
    if accuracy > best_accuracies[classifier_type]:
        best_accuracies[classifier_type] = accuracy
        mlflow.log_metric(f"best_{classifier_type}_accuracy", accuracy)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}


# Define the search space over hyperparameters (for classifier only; feature selection is fixed) #TODO find more practical examples where these are defined; what is worth definining and what ranges?
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
        # 'solver': hp.choice('lr_solver', ['liblinear', 'lbfgs'])
        'solver': 'saga',  # Force solver for elasticnet in feature selection
        'penalty': 'elasticnet',
    },
    {
        'type': 'xgb',
        'max_depth': hp.quniform("xgb_max_depth", 3, 15, 1),
        'gamma': hp.uniform('xgb_gamma', 0, 9),
        'reg_alpha': hp.quniform('xgb_reg_alpha', 0, 10, 1),
        'reg_lambda': hp.uniform('xgb_reg_lambda', 0, 10),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1),
        'min_child_weight': hp.quniform('xgb_min_child_weight', 0, 12, 1),
        'n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
        'seed': 0,
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'scale_pos_weight': hp.uniform('xgb_scale_pos_weight', 1, 10)  # Adjust if classes are imbalanced
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

print("Now tuning hyperparameters \n")
with mlflow.start_run():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # IMPROVE Change this to ~10 if testing further // Switch back to 100 if not
        trials=Trials()
    )

# Print the best accuracies for each model type
print("\nHighest model accuracies on train data:")
best_accuracy_df = pd.DataFrame(list(best_accuracies.items()), columns=['Models', 'Highest accuracy'])
print(best_accuracy_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ###########################################################################################
# Train final model using the full training data
with mlflow.start_run():  # TODO need to find examples of this being done - unsure on the final training/testing after hyperopt tuning
    classifier_type = best_config['type']
    best_params = {k: v for k, v in best_config.items() if k != 'type'}
    selector = feature_selectors_dict[classifier_type]

    # Log the best hyperparameters
    mlflow.log_params(best_config)

    # Construct the classifier with the best parameters - converting to integers if needed
    if classifier_type == 'svm':
        classifier = SVC(**best_params)
    elif classifier_type == 'rf':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = RandomForestClassifier(**best_params)
    elif classifier_type == 'logreg':
        classifier = LogisticRegression(**best_params)
    elif classifier_type == 'xgb':
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        classifier = XGBClassifier(**best_params)
    elif classifier_type == 'gb':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = GradientBoostingClassifier(**best_params)

    # Create the final pipeline with feature selection and classifier # TODO not sure if i need pipeline here since I dont feed into cross_val_score?
    final_pipeline = Pipeline([
        ('feature_selector', selector),
        ('classifier', classifier)
    ])

    # Train on full training data
    final_pipeline.fit(X_train, y_train)

    # Print the selected features
    try: # TODO this was made was RFECV was used for all, so likely no longer works
        selected = final_pipeline.named_steps['feature_selector']
        selected_features = X_train.columns[selected.support_]
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features.tolist())
    except:
        print("Unable to print features - see note in code.")

    # Log the final pipeline model
    mlflow.sklearn.log_model(final_pipeline, "best_model")

    # Evaluate the final model on the test set
    y_pred = final_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")

# Example output:
# Test accuracy with best model (xgb): 0.7331
# Test F1 score with best model (xgb): 0.7599

#TODO: !!! logreg failed on test so investigate that

# TODO: Question: If I get different models (LR and GB currently) on different runs, what should I do? Pick one? Use the
#  most common of multiple attempts? Or set seed so it's always consistent

# TODO: Test with ordinal encoding for some of the categories I used nominal
#  Plot feature importance - could potentially remove unimportant features and retest
#  Generate a learning curve

# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement
#  Could also do an ensemble model approach for the final training, and stacking/voting
#  More elegant way to handle hyperopt returning floats?

# TODO split up the final model building and best model selection so I don't have to re-do the initial testing