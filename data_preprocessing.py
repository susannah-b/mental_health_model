######### SETUP ########################################################################################################
# Import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import missingno as msno
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import miceforest as mf
from sklearn.preprocessing import LabelEncoder

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

# Split the train and test data
train, test = train_test_split(df, test_size=0.2, stratify=df['treatment'], random_state=42)

# Split train data into X and y
X_train = train.drop('treatment',axis=1)
y_train = train['treatment'].copy()

### INVESTIGATE MISSING VALUES #########################################################################################
# Set a bool to display graphs or missingness exploration
Show_graphs = True
Show_M_inves = True
### Investigate missing values in work_interfere and self_employed
missing_cols = ["work_interfere", "self_employed"]
# Visualize missingness patterns
if Show_graphs:
    msno.matrix(X_train)
    plt.show()
    # Result: No visible pattern but could be linked to other variables. For work_interfere we know that the question requires
    # the presence of a mental health condition so is MNAR.
    # TODO Question: is the above correct? And would I just say self_employed is MCAR too based on prior knowledge?
    # For self_employed, the missing frequency is very low so simple imputation methods like the mode could
    # be used, but as an exercise in missing values we will attempt to determine missingness type and impute it
    # For work_interfere we will also look at missingness type but impute with methods suitable for MNAR

# List categorical and continuous columns in order to split between chi squared and regression tests
    # Note: excludes 'treatment' as the predictor variable
categorical_cols = ['gender', 'self_employed', 'family_history', 'work_interfere',
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
            print(f"{col}: WARNING: Chi-squared test failed.")

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
            print(f"{col}: WARNING: Regression failed")

    print(f"{target_col} significant columns: {significant_cols}")

    return significant_cols

# Store significant columns for each category
sig_cols_dict = {}
for i in missing_cols:
    # Show breakdown of categories, including missing data
    if Show_M_inves:
        print(f"\n{i} Values:\n", X_train[i].value_counts(dropna=False))
    # Call function to test for missingness type
    sig_cols = test_missing_mechanisms(X_train, target_col=i, categorical_vars=categorical_cols, continuous_vars=continuous_cols)
    sig_cols_dict[i] = sig_cols

    # Result:
    # work_interfere significant columns: ['gender', 'family_history', 'benefits', 'care_options', 'wellness_program', 'seek_help',
    #           'leave', 'mental_health_consequence', 'phys_health_consequence', 'supervisor', 'mental_health_interview',
    #           'obs_consequence', 'age']

    # We already know work_interfere is MNAR, but we also see a relationship between lots of other variables so
    # it could also be MAR. It would be expected that our unobserved variable of not having a mental health condition also has
    # a link to some of these variables, like family_history.
    # The relationships can also help us interpret the trends in these variables, e.g. those without a mental health condition
    # might be less likely to notice consequences others experienced (would need to check this with EDA)
    # For our imputation, we shall proceed with both MAR and MNAR-compatible methods and compare
        #TODO Question: is this interpretation correct? Does MNAR status (known from prior survey knowledge) supercede MAR status?

    # self_employed significant columns: ['anonymity', 'coworkers', 'mental_health_interview']

    # For self_employed, we see fewer relationships (although we do have very few data points so any links are spurious
    # or perhaps other relationships were not picked up on). We can conclude the data is MAR due to these relationships.
        # TODO Question: is this interpretation correct? Does it just suggest it? Is it invalid due to few data points?


### IMPUTE MISSING VALUES ##############################################################################################
# Store a dict of ordinal/one-hot-encoding categories and their categories (in order for ordinal)
    # Minus 'treatment' column; only in X data
ordinal_cats = {"work_interfere" : ['Never', 'Rarely', 'Sometimes', 'Often'],
                "no_employees" : ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
                "mental_health_consequence" : ['No', 'Maybe', 'Yes'],
                "phys_health_consequence" : ['No', 'Maybe', 'Yes'],
                "coworkers" : ['No', 'Some of them', 'Yes'],
                "supervisor" : ['No', 'Some of them', 'Yes'],
                "mental_health_interview" : ['No', 'Maybe', 'Yes'],
                "phys_health_interview" : ['No', 'Maybe', 'Yes'],
                }
# Note: for one_hot encoding the column values aren't actually needed, but leaving in for possible future use and reference
nominal_cats = {"self_employed" : ['Yes', 'No'],
                "gender" : ['Male', 'Female', 'Other'],
                "family_history": ['Yes', 'No'],
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
# TODO Question: If you have ordinal categories and a 'don't know', what's the best way to handle it

# TODO Question: I think the fact we can potentially infer mental health conditions from the work_interfere column is interesting;
# TODO           is it valid to add that as a column and use as a feature or bad practice? (perhaps here it's mainly an ethics concern)

### Encode ordinal categories and convert nominal categories to pd categories for imputation
# Ordinal categories
for cat, codes in ordinal_cats.items():
    X_train[cat] = pd.Categorical(X_train[cat], categories=ordinal_cats[cat], ordered=True).codes
    # Replace -1 with np.nan if needed (sometimes factorize returns -1, but pd.Categorical.cat.codes returns -1 for NaN)
    X_train[cat] = X_train[cat].replace(-1, np.nan)

# Nominal categories
for cat in nominal_cats.keys():
    X_train[cat] = pd.Categorical(X_train[cat], ordered=False)

# Create a dataset to store intermediate columns for missingness handling
train_missing = X_train.copy()

# Reset index for miceforest use
train_missing = train_missing.reset_index(drop=True)

### MAR Imputation for complete dataset with MICE
# Initialize kernel (handles categoricals natively)
kernel = mf.ImputationKernel(
    data=train_missing, num_datasets=20, random_state=42)

# Run MICE with 10 iterations
kernel.mice(iterations=10)
kernel.plot_feature_importance(dataset=0)
kernel.plot_imputed_distributions()

# Extract completed data
train_missing = kernel.complete_data()

# todo certain parts need to be indented as Show_inves and else have the streamlined process

# (Alternative) MNAR Imputation with new category as the missingness is informative; no mental health condition
train_missing['work_interfere_mnar'] = X_train['work_interfere'].reset_index(drop=True).fillna('No Condition Disclosed')
#TODO Question: Is it bad practice to handle MNAR with adding a new category in the same column? Should it be separate?
#TODO Also, other example code uses 'Never' because most participants don't seek treatment. But personally I don't think
# we can draw that conclusion - am I wrong?
# TODO Question: Should I instead be implementing the more advanced MNAR methods, or just assume MAR since we have a relationship?

# Compare distributions before/MAR/MNAR
if Show_M_inves:
    print(f"\n\nOriginal Values:\n", X_train['work_interfere'].value_counts(dropna=False))
    print(f"MAR Values:\n", train_missing['work_interfere'].value_counts(dropna=False))
    print(f"MNAR Values:\n", train_missing['work_interfere_mnar'].value_counts(dropna=False))

### Plot work_interfere to compare TODO tweak, e.g. show percentages, fix FutureWarning, make colours consistent for cats
if Show_graphs:
    # Create mapping dictionary for numeric codes to labels
    category_map = {
        0.0: 'Never',
        1.0: 'Rarely',
        2.0: 'Sometimes',
        3.0: 'Often',
        'No Condition Disclosed': 'No Condition Disclosed'
    }

    # Create a temporary DataFrame for plotting
    plot_df = pd.DataFrame({
        "Original": X_train['work_interfere'],
        "MAR Imputed": train_missing['work_interfere'],
        "MNAR Imputed": train_missing['work_interfere_mnar']
    })

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    plt.suptitle("Work Interfere Distribution Comparison", fontsize=16, y=1.05)

    # Define order for first two plots
    main_order = ['Never', 'Rarely', 'Sometimes', 'Often']
    mnar_order = main_order + ['No Condition Disclosed']

    # Plot Original
    sns.countplot(
        x=plot_df['Original'].map(category_map),
        order=main_order,
        ax=axes[0],
        palette='viridis'
    )
    axes[0].set_title("Original Data (with Missing)", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot MAR Imputed
    sns.countplot(
        x=plot_df['MAR Imputed'].map(category_map),
        order=main_order,
        ax=axes[1],
        palette='viridis'
    )
    axes[1].set_title("MAR Imputed", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis='x', rotation=45)

    # Plot MNAR Imputed
    sns.countplot(
        x=plot_df['MNAR Imputed'].map(category_map),
        order=mnar_order,
        ax=axes[2],
        palette='viridis'
    )
    axes[2].set_title("MNAR Imputed", fontsize=12)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis='x', rotation=45)

    # Adjust layout and add y-label
    fig.text(0.08, 0.5, 'Count', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.show()

# Results: We can see a slightly different distribution for the MAR imputed values. This is consistent with what we
# expect to see with the 0/1 being higher after imputation - as those that didn't answer are likely to be those without
# mental health conditions we would expect work interference to be low. But we do see an increase in the higher values too
# (to a lesser extent) which makes sense as those undiagnosed/unwilling to disclose will also see some work interference.
# Based on these results, I think the best choice is to move forward with the MICE imputed values for work_interfere.

# Let's drop the MNAR column to just use the MICE imputed values
train_missing = train_missing.drop(['work_interfere_mnar'], axis=1)

# Nominal variables: One-hot encode for modeling #TODO check this is what I want!
train_condensed = train_missing.copy() # Make a copy of the dataset before one-hot encoding
train_missing = pd.get_dummies(train_missing,columns=nominal_cats.keys(), dtype=int)

# Scale X_train data
std_scaler = StandardScaler()
train_missing_scaled = std_scaler.fit_transform(train_missing)
train_missing = pd.DataFrame(train_missing_scaled, index=train_missing.index, columns=train_missing.columns)

### Encode y_train
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_encoded= label_encoder.transform(y_train)
# Convert back to df
y_train = pd.DataFrame(y_encoded, index=y_train.index, columns=["treatment"])
# Reset index to match other datasets

y_train = y_train.reset_index(drop=True) # Use drop to remove the old index TODO check this is consistent with all the others
# Note: 'treatment' has almost equal values for both the categories. We do not have to perform undersampling or oversampling

# Write to csv for use in model_building.py
train_missing.to_csv("X_training_data.csv", sep=",", index=False) # X_train data
y_train.to_csv("y_training_data.csv", sep=",", index=False) # y_train data


### CORRELATION MATRIX #################################################################################################
if Show_graphs:
    # Compute the correlation matrix
    train_missing['treatment'] = y_train['treatment']
    # print(train_missing['treatment'].head())
    corr = train_missing.corr()
    # print(list(train_missing.columns))
    # TODO this heatmap doesn't show the treatment column for some reason - remove prints when fixed

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='vlag', vmax=.3, center=0, square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.show()

    # Result: Something important to note for one-hot encoded data... it's useless between split columns from the same nominal
    # category!

    ### If we remove nominal categories:
    # Filter out nominal columns before computing correlations
    train_no_nominal = train_condensed.drop(columns=nominal_cats, errors='ignore')  # 'errors=ignore' skips non-existent columns
    # Add treatment column back to see correlation
    train_no_nominal['treatment'] = y_train['treatment']
    # Compute correlation matrix on the filtered data
    corr = train_no_nominal.corr()

    # Generate mask for upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap='vlag', vmax=.3, center=0, square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.show()
    print("full", train_missing['treatment'].isna().sum())
    print("NONOMINAL", train_no_nominal['treatment'].isna().sum())
    # Result: Can see some pos/neg correlations, but none with treatment


### TEST DATA PREPROCESSING ############################################################################################
# Now we need to perform the same processing steps as our train data on our test data

# Split test data into X and y
X_test = test.drop('treatment',axis=1)
y_test = test['treatment'].copy()

### Handle categories - ordinal and nominal
for cat, codes in ordinal_cats.items():
# Ordinal categories
    X_test[cat] = pd.Categorical(X_test[cat], categories=ordinal_cats[cat], ordered=True).codes
    # Replace -1 with np.nan if needed (sometimes factorize returns -1, but pd.Categorical.cat.codes returns -1 for NaN)
    X_test[cat] = X_test[cat].replace(-1, np.nan)
# Nominal categories
for cat in nominal_cats.keys():
    X_test[cat] = pd.Categorical(X_test[cat], ordered=False)

# Create a dataset to store intermediate columns for missingness handling
test_missing = X_test.copy()

# Reset index for miceforest use
test_missing = test_missing.reset_index(drop=True)

### MAR Imputation for complete dataset with MICE
# Initialize kernel (handles categoricals natively)
kernel = mf.ImputationKernel(
    data=test_missing, num_datasets=20, random_state=42)

# Run MICE with 10 iterations
kernel.mice(iterations=10)
kernel.plot_feature_importance(dataset=0)
kernel.plot_imputed_distributions()

# Extract completed data
test_missing = kernel.complete_data()

# Skipping MNAR handling code but would need to reimplement if using MNAR instead

# Nominal variables: One-hot encode for modeling
test_condensed = test_missing.copy() # Make a copy of the dataset before one-hot encoding
test_missing = pd.get_dummies(test_missing,columns=nominal_cats.keys(), dtype=int)

# Scale X_test data
std_scaler = StandardScaler()
test_missing_scaled = std_scaler.fit_transform(test_missing)
test_missing = pd.DataFrame(test_missing_scaled, index=test_missing.index, columns=test_missing.columns)

### Encode y_test
label_encoder = LabelEncoder()
label_encoder.fit(y_test)
y_encoded= label_encoder.transform(y_test)
# Convert back to df
y_test = pd.DataFrame(y_encoded, index=y_test.index, columns=["treatment"])
# Reset index to match other datasets

y_test = y_test.reset_index(drop=True) # Use drop to remove the old index
# Note: 'treatment' has almost equal values for both the categories. We do not have to perform undersampling or oversampling

# Write to csv for use in model_building.py
test_missing.to_csv("X_testing_data.csv", sep=",", index=False) # X_test data
y_test.to_csv("y_testing_data.csv", sep=",", index=False) # y_test data





# TODO all the missingess/imputation code needs more careful testing before deploying with a research model, e.g. check categories are
#  converted back correctly, imputations make sense, chi squared is working correctly, etc

# TODO: Check data looks correct. Might require a toy data set with a few values and work through whole process checking correct values are maintained
     # go through and check steps - e.g. the test data processing at the end was done quickly
# TODO: reset_index has caused some problems! for final datasets and when combining them, check the indexes.
