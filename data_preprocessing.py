######### SETUP ########################################################################################################
# Import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import xgboost as xgb
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


#todo make sure all libraries are used above

# self_employed/family_history/treatment(prediction variable) looks good; has two categories (Yes or No). But we will need to remove NaN for self-employed todo
# work_interfere has missing values, and ordinal categories ['Often' 'Rarely' 'Never' 'Sometimes' nan] todo
# no_employees is also ordinal categories ['6-25' 'More than 1000' '26-100' '100-500' '1-5' '500-1000'] todo
# leave is ordinal categories (+ Don't know) ['Somewhat easy' "Don't know" 'Somewhat difficult' 'Very difficult' 'Very easy'] todo
# todo do we ignore gender? or it still counts. maybe test with and without

# TODO these are skipped in the example code. Figure out if they're replaced. Not splitting xy was useful for EDA tho. Prob coulda been categories for EDA tho
# # Extract feature and target arrays
# X, y = diamonds.drop('price', axis=1), diamonds[['price']]
#
# # Extract categories and convert to an integer (required for newer XGBoost versions)
# cats = X.select_dtypes(exclude=np.number).columns.tolist()
# for col in cats:
#     X[col] = X[col].astype('category')
#     X[col] = X[col].cat.codes  # Converts the category to its integer code

# Results: Vast majority of respondents report work interference. Those who have sought treatment skew more towards the higher frequency than those untreated
# TODO should we omit this? Idk it's an odd variable to predict having sought treatment anyway. But this group have already sought treatment so... it feels odd to me

#todo remove self employed work interfere missing values

# todo there's also a few ordinal categories which i think should be encoded numerically (see todos above)


#feature importance is probably the key here - we want to focus on what allows employees to be able to seek treatment, not whether they do todo


# Set pandas to display all columns
pd.set_option('display.max_columns', None)

### Read in data as in the data_exploration.py script
# Read in data
csv_path = Path(__file__).parent / "survey_data.csv"
df = pd.read_csv(csv_path)
# Convert columns to lower case for easier handling
df.columns = df.columns.str.lower()
# Drop unnecessary columns
df = df.drop(['timestamp', 'country', 'state', 'comments'], axis=1)
# Remove values below 18 and above 80
df = df[(df['age'] >= 18) & (df['age'] <= 80)]
# Reduce gender options down to fewer categories
df.replace({'gender' : ['M', 'male', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man', 'msle', 'Mail', 'cis male',
                      'Malr', 'Cis Man']}, 'Male', inplace = True)
df.replace({'gender' : ['F', 'female', 'f', 'Woman', 'Cis Female', 'Femake', 'woman', 'Female ', 'cis-female/femme','Female (cis)', 'femail']}, 'Female',
                     inplace = True)
df.replace({'gender' : ['queer/she/they', 'Trans-female', 'something kinda male?', 'non-binary', 'Nah', 'All', 'Enby', 'fluid', 'Genderqueer',
                      'Androgyne', 'Agender', 'Guy (-ish) ^_^', 'male leaning androgynous', 'Trans woman', 'Neuter', 'Female (trans)', 'queer',
                      'A little about you', 'p', 'ostensibly male, unsure what that really means']}, 'Other', inplace = True)

# Split the train and test data TODO could do xy method here now
train, test = train_test_split(df, test_size=0.2, stratify=df['treatment'], random_state=42)

### INVESTIGATE MISSING VALUES #########################################################################################
# Set a bool to display graphs or missingness exploration
Show_graphs = False
Show_M_inves = False
### Investigate missing values in work_interfere and self_employed
missing_cols = ["work_interfere", "self_employed"]
# Visualize missingness patterns
if Show_graphs:
    msno.matrix(train)
    plt.show()
    # Result: No visible pattern but could be linked to other variables. For work_interfere we know that the question requires
    # the presence of a mental health condition so is MNAR.
    # TODO Question: is the above correct? And would I just say self_employed is MCAR too based on prior knowledge?
    # For self_employed, the missing frequency is very low so simple imputation methods like the mode could
    # be used, but as an exercise in missing values we will attempt to determine missingness type and impute it
    # For work_interfere we will also look at missingness type but impute with methods suitable for MNAR

# List categorical and continuous columns in order to split between chi squared and regression tests
categorical_cols = ['gender', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                    'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
                    'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
                    'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']
continuous_cols = ['age']  # List of continuous variables

# Function to explore if MCAR/MAR
def test_missing_mechanisms(df, target_col, categorical_vars, continuous_vars):
    """Run formal tests for MCAR/MAR."""
    # Remove target_col from the variable lists
    categorical_vars = [col for col in categorical_vars if col != target_col]
    continuous_vars = [col for col in continuous_vars if col != target_col]

    # Create missingness indicator
    df = df.copy()
    df[f'{target_col}_missing'] = df[target_col].isnull().astype(int)

    # Skipping Little's test as it only shows if the whole dataset is MCAR. And we know work_interferes is MNAR

    # Run Chi-square Tests for categorical variables to check if there is a relationship between variables
    if Show_M_inves:
        print(f"\nChi-Square Tests for {target_col} MAR test:") #todo "Categorical variables with sparse categories (e.g., obs_consequence) will produce unreliable χ² results". Check for frequencies
    significant_cols = []
    for col in categorical_vars:
        try:
            contingency_table = pd.crosstab(df[f'{target_col}_missing'], df[col])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            if p < 0.05:
                if Show_M_inves:
                    print(f"{col}: χ²={chi2:.2f}, p={p:.4f} *")
                significant_cols.append(col)
            else:
                if Show_M_inves:
                    print(f"{col}: χ²={chi2:.2f}, p={p:.4f}")
        except:
            print(f"{col}: WARNING: Test failed.")

    # Run Logistic Regression test for continuous variables to check if there is a relationship
    if Show_M_inves:
        print(f"\nLogistic Regression for {target_col} MAR test:")
    for col in continuous_vars:
        try:
            temp_df = df[[col, f'{target_col}_missing']].dropna()
            X = sm.add_constant(temp_df[col])
            y = temp_df[f'{target_col}_missing']

            model = sm.Logit(y, X).fit(disp=0)
            p_value = model.pvalues[col]
            if p < 0.05:
                if Show_M_inves:
                    print(f"{col}: Coef={model.params[col]:.2f}, p={p_value:.4f} *")
                significant_cols.append(col)
            else:
                if Show_M_inves:
                    print(f"{col}: Coef={model.params[col]:.2f}, p={p_value:.4f}")
        except:
            print(f"{col}: Regression failed")

    print(f"{target_col} significant columns: {significant_cols}")

    return significant_cols

# Store significant columns for each category
sig_cols_dict = {}
for i in missing_cols:
    # Show breakdown of categories, including missing data
    if Show_M_inves:
        print(f"\n{i} Values:\n", train[i].value_counts(dropna=False))
    # Call function to test for missingness type
    sig_cols = test_missing_mechanisms(train, target_col=i, categorical_vars=categorical_cols, continuous_vars=continuous_cols)
    sig_cols_dict[i] = sig_cols

    # Result:
    # work_interfere significant columns: ['gender', 'family_history', 'treatment', 'benefits', 'care_options', 'wellness_program', 'seek_help',
    #           'leave', 'mental_health_consequence', 'phys_health_consequence', 'supervisor', 'mental_health_interview',
    #           'obs_consequence', 'age']

    # We already know work_interfere is MNAR, but we also see a relationship between lots of other variables so
    # it could also be MAR. It would be expected that our unobserved variable of not having a mental health condition also has
    # a link to some of these variables, like family_history.
    # The relationships can also help us interpret the trends in these variables, e.g. those without a mental health condition
    # might be less likely to notice consequences others experienced (would need to check this with EDA)
    # For our imputation, we shall proceed with MNAR-compatible methods
        #TODO Question: is this interpretation correct? Does MNAR status (known from prior survey knowledge) supercede MAR status?

    # self_employed significant columns: ['anonymity', 'coworkers', 'mental_health_interview']

    # For self_employed, we see fewer relationships (although we do have very few data points so any links are spurious
    # or perhaps other relationships were not picked up on). We can conclude the data is MAR due to these relationships.
        # TODO Question: is this interpretation correct? Does it just suggest it? Is it invalid due to few data points?


### IMPUTE MISSING VALUES ##############################################################################################
### Handling MNAR values
    # Note: This is only done for work_interfere so for other data sets needs to be adapted as a function

# Store a dict of ordinal/one-hot-encoding categories and their categories (in order for ordinal)
ordinal_cats = {"work_interfere" : ['Never', 'Rarely', 'Sometimes', 'Often'],
                "no_employees" : ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
                "mental_health_consequence" : ['No', 'Maybe', 'Yes'],
                "phys_health_consequence" : ['No', 'Maybe', 'Yes'],
                "coworkers" : ['No', 'Some of them', 'Yes'],
                "supervisor" : ['No', 'Some of them', 'Yes'],
                "mental_health_interview" : ['No', 'Maybe', 'Yes'],
                "phys_health_interview" : ['No', 'Maybe', 'Yes'],
                }
# Note: for one_hot encoding the column values aren't actually needed TODO currently don't do one_hot encoding here
one_hot_cats = {"self_employed" : ['Yes', 'No'],
                "gender" : ['Male', 'Female', 'Other'],
                "family_history": ['Yes', 'No'],
                "treatment": ['Yes', 'No'],
                "remote_work" : ['Yes', 'No'],
                "tech_company" : ['Yes', 'No'],
                "benefits" : ['Yes', "Don't know", 'No'],
                "care_options" : ['Not sure', 'No', 'Yes'],
                "wellness_program" : ['No', "Don't know", 'Yes'],
                "seek_help" : ['Yes', "Don't know", 'No'],
                "anonymity" : ['Yes', "Don't know", 'No'],
                "leave" : ["Don't know", 'Somewhat easy', 'Somewhat difficult', 'Very difficult', 'Very easy'],
                "mental_vs_physical" : ['Yes', "Don't know", 'No'],
                "obs_consequence" : ['Yes', 'No']
                }
#TODO Question: If you have ordinal categories and a 'don't know', what's the best way to handle it

# TODO Question: I think the fact we can potentially infer mental health conditions from the work_interfere column is interesting;
# TODO           is it valid to add that as a column and use as a feature or bad practice?

### Create data set to with categorical encodings before imputing, and to prepare for model development
# Ordinal categories - do this before imputation
for cat, codes in ordinal_cats.items():
    train[cat] = pd.Categorical(train[cat], categories=ordinal_cats[cat], ordered=True).codes
    # Replace -1 with np.nan if needed (sometimes factorize returns -1, but pd.Categorical.cat.codes returns -1 for NaN) TODO
    train[cat] = train[cat].replace(-1, np.nan)

# Create a dataset to store intermediate columns for missingness handling TODO tweaked this in code
train_missing = train.copy()

# MAR Imputation (Iterative) - Use MICE with Random Forest (for categorical data) # TODO Can i do KNN here?
def impute_mar(target_col, input_df):
    print([sig_cols_dict[target_col]])
    print("****", sig_cols_dict)
    imputer_mar = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=42)
    input_df[f'{target_col}_code_imputed'] = imputer_mar.fit_transform(input_df[sig_cols_dict[target_col]]) #TODO check this works corrrectly (now doesn't bc nominal cats!!)


    # Convert imputed column back to categories to investigate todo for final data I don't need this but good for evaluating at end
    input_df[f'{target_col}_mar'] = pd.Categorical(
        pd.Series(np.round(input_df[f'{target_col}_code_imputed'].to_numpy()).astype(int)),
        categories=range(len(ordinal_cats[target_col])), ordered=True).rename_categories(ordinal_cats[target_col]) #TODO check this works


# todo certain parts need to be indented as Show_inves and else have the streamlined process

# TODO: Qeustion: The imputation above doesn't work because of nominal categories. What's the best way to handle this?
# e.g. should I do ordinal values for the binary categories and not use the rest for imputation?
# Impute missing values assuming MAR
    # Note: work_interfere is included here as we see some correlation, but we can also impute assuming it's an MNAR
for col in missing_cols:
    print(col)
    impute_mar(col, train_missing)
    print("To do: Fix MAR imputation")

# (Alternative) MNAR Imputation with new category as the missingness is informative; no mental health condition
train_missing['work_interfere_mnar'] = train_missing['work_interfere'].fillna('No Condition Disclosed')
#TODO Question: Is it bad practice to handle MNAR with adding a new category in the same column? Should it be separate?
#TODO Also, other example code uses 'Never' because most participants don't seek treatment. But personally I don't think
# we can draw that conclusion - am I wrong?
#TODO Questoin: Should I instead be implementing the more advanced MNAR methods, or just assume MAR since we have a relationship?

# Compare distributions TODO need to go back and fix MAR imputation
print("\nImputation Comparison:")
# print("MAR Imputed Distribution:")
# print(train_missing['work_interfere_mar'].value_counts(normalize=True))
print("\nMNAR Imputed Distribution:")
print(train_missing['work_interfere_mnar'].value_counts(normalize=True))

# Compare actual values
print("\n\n")
print(f"Original Values:\n", train_missing['work_interfere'].value_counts(dropna=False))
# print(f"MAR Values:\n", train_missing['work_interfere_mar'].value_counts(dropna=False))
print(f"MNAR Values:\n", train_missing['work_interfere_mnar'].value_counts(dropna=False))

# Results: TODO make a conclusion on MAR vs MNAR when it's fixed

# TODO: For now we proceed with the MNAR results and do a simple mode imputation for self_employed

# TODO remove this once I fix MAR...
# Impute self_employed with the modal value and add to _mode column
se_mode = train_missing['self_employed'].mode().values[0]
train_missing['self_employed_mode'] = train_missing['self_employed']
train_missing.fillna({'self_employed_mode': se_mode}, inplace=True)

# Update the training data frame with the imputed values
train['work_interfere'] = train_missing['work_interfere_mnar']
train['self_employed'] = train_missing['self_employed_mode']

print(train.head(1))

exit()
# Nominal categories (do one-hot encoding) todo this doesn't work with iterative imputer... find out where to put it
for cat in one_hot_cats.keys():
    train = pd.get_dummies(train, columns=[cat], dtype=int)





# TODO all the missingess/imputation code needs careful testing before deploying with a research model, e.g. check categories are
# TODO converted back correctly, imputations make sense, chi squared is working correctly, etc

#TODO: problem: iterativeimpute isn't working so my MAR code is currently commented out and I do an alternative method

# TODO need to repeat encoding on the test data in the other file. and maybe missing data or whatever else



