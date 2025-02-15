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
import plotly.express as px
#todo make sure all libraries are used above

# Ignore warnings
warnings.filterwarnings("ignore")

# Read in data
csv_path = Path(__file__).parent / "survey_data.csv"
df = pd.read_csv(csv_path)

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

### DATA PREPARATION ##########################################################################################
Print_Prep = False # Can switch this off for later analysis to avoid cluttering the terminal

if Print_Prep:
    print("Data shape:", df.shape, "\n")
    print(df.head(), "\n")
    print(df.info(), "\n")
    print(df.isnull().sum(), "\n")
    # We see that state, self_employed, work_interfere, and comments all have null values
    # Age is the only numerical column
    # We want to ignore timestamp (when they filled in the survey)
    # Comments we can ignore; there are fairly few entries and are difficult to categorise into meaningful dat

    # Convert columns to lower case for easier handling
    df.columns = df.columns.str.lower()

    # Examining individual columns
    print(df['country'].value_counts())
    # This is heavily weighted towards the US and many countries have 1/few entries. We will drop this column to avoid misleading results
    # Subsequently we will drop state, as this is US-oriented

    # Drop unnecessary columns
    df = df.drop(['timestamp', 'country', 'state', 'comments'], axis = 1)
    # Check our data again
    print(df.describe(include='all'))

    ### Checking each of the columns (df.describe results and the unique values prints) - comment at the start gives the results, followed by unique value print for most to check categories
    # age: max and min values look very off, we need to handle these
    # gender: 49 different categories! We need to simplify these
    remaining_variables = ['age', 'gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work',
                           'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                           'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                           'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

    for variable in remaining_variables:
        print(variable, df[variable].unique())
        # self_employed/family_history/treatment(prediction variable) looks good; has two categories (Yes or No). But we will need to remove NaN for self-employed todo
        # work_interfere has missing values, and ordinal categories ['Often' 'Rarely' 'Never' 'Sometimes' nan] todo
        # no_employees is also ordinal categories ['6-25' 'More than 1000' '26-100' '100-500' '1-5' '500-1000'] todo
        # remote_work/tech_company is fine
        # benefits is fine ['Yes' "Don't know" 'No']
        # care_options is fine ['Not sure' 'No' 'Yes']
        # wellness_program is fine ['No' "Don't know" 'Yes']
        # seek_help is fine ['Yes' "Don't know" 'No']
        # anonymity is fine ['Yes' "Don't know" 'No']
        # leave is ordinal categories (+ Don't know) ['Somewhat easy' "Don't know" 'Somewhat difficult' 'Very difficult' 'Very easy'] todo
        # mental_health_consequence is fine ['No' 'Maybe' 'Yes']
        # phys_health_consequence is fine ['No' 'Yes' 'Maybe']
        # coworkers is fine ['Some of them' 'No' 'Yes']
        # supervisor is fine ['Yes' 'No' 'Some of them']
        # mental_health_interview is fine ['No' 'Yes' 'Maybe']
        # phys_health_interview is fine ['Maybe' 'No' 'Yes']
        # mental_vs_physical is fine ['Yes' "Don't know" 'No']
        # obs_consequence is Y/N no missing values

else: # Do the bits of code we want to keep for future analysis only; no EDA
    # Convert columns to lower case for easier handling
    df.columns = df.columns.str.lower()
    # Drop unnecessary columns
    df = df.drop(['timestamp', 'country', 'state', 'comments'], axis = 1)

# DATA CLEANING ########################################################################################################
# Following the results above, we need to clean up certain columns

### 1. Age - Some values are outside of what we expect for tech workers (too young) or humans (too old)!
if Print_Prep:
    # Check for NAs - none found
    print(df['age'].isna().sum())

    # See head and tail of data - we can see some negative and very high values
    df_sorted = df.sort_values(by='age', ascending=True)
    print(df_sorted['age'].head(8))
    print(df_sorted['age'].tail())

    # Histogram of ages - looks very off, due to the few very high ages
    plt.figure()
    plt.hist(df['age'], bins=12, edgecolor='black')
    plt.title('Ages - before cleaning')

# Remove values below 18 and above 80
df = df[(df['age'] >= 18) & (df['age'] <= 80)]

if Print_Prep: # Show the results of removing ages
    # Histogram of ages after removal of odd values
    plt.figure()
    plt.hist(df['age'], bins=12, edgecolor='black')
    plt.title('Ages - after cleaning') # Now looks much better

    # Show plots
    plt.show()

### 2. Gender - We have 49 different categories that we need to simplify
    # ['Female' 'M' 'Male' 'male' 'female' 'm' 'Male-ish' 'maile' 'Trans-female' 'Cis Female' 'F' 'something kinda male?' 'Cis Male' 'Woman' 'f' 'Mal'
    # 'Male (CIS)' 'queer/she/they' 'non-binary' 'Femake' 'woman' 'Make' 'Nah' 'All' 'Enby' 'fluid' 'Genderqueer' 'Female ' 'Androgyne' 'Agender' 'cis-female/femme'
    # 'Guy (-ish) ^_^' 'male leaning androgynous' 'Male ' 'Man' 'Trans woman' 'msle' 'Neuter' 'Female (trans)' 'queer' 'Female (cis)' 'Mail' 'cis male' 'A little about you'
    # 'Malr' 'p' 'femail' 'Cis Man' 'ostensibly male, unsure what that really means']

# Reduce these down to fewer categories
df['gender'].replace(['M', 'male', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man', 'msle', 'Mail', 'cis male',
                      'Malr', 'Cis Man'], 'Male', inplace = True)
df['gender'].replace(['F', 'female', 'f', 'Woman', 'Cis Female', 'Femake', 'woman', 'Female ', 'cis-female/femme','Female (cis)', 'femail'], 'Female',
                     inplace = True)
df['gender'].replace(['queer/she/they', 'Trans-female', 'something kinda male?', 'non-binary', 'Nah', 'All', 'Enby', 'fluid', 'Genderqueer',
                      'Androgyne', 'Agender', 'Guy (-ish) ^_^', 'male leaning androgynous', 'Trans woman', 'Neuter', 'Female (trans)', 'queer',
                      'A little about you', 'p', 'ostensibly male, unsure what that really means'], 'Other', inplace = True)

# Check results (Now corrected, but note the heavy male sampling bias so beware of drawing false conclusions)
if Print_Prep:
    print(df['gender'].value_counts())

### EXPLORATORY DATA ANALYSIS ##########################################################################################
Print_EDA = False # Can switch this off for later analysis to avoid cluttering the terminal

# TODO these are skipped in the example code. Figure out why! Think parts are done later but yea
# # Extract feature and target arrays
# X, y = diamonds.drop('price', axis=1), diamonds[['price']]
#
# # Extract categories and convert to an integer (required for newer XGBoost versions)
# cats = X.select_dtypes(exclude=np.number).columns.tolist()
# for col in cats:
#     X[col] = X[col].astype('category')
#     X[col] = X[col].cat.codes  # Converts the category to its integer code

# First split the model so we don't perform EDA on the test data
# Split the data into train/test
train, test = train_test_split(df, test_size=0.2, stratify=df['treatment'], random_state=42)
# todo: I did it a bit differently where I split X and Y as well as train and test. Make sure all following code applies to all sets where necessary. eg data prep which isn't done yet!
# todo: now changed bc I want to colour by predictor

# Set colour palette for graphs coloured by treatment column
colour_palette = {"Yes": "#ff0a47", "No": "#00aaff"} # todo improve colours
if Print_EDA:
    print(f'Train data dimensions: {train.shape}')
    print(f'Test data dimensions: {test.shape}')

    ### Explore each column
        # ['age', 'gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work', 'tech_company',
        # 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave','mental_health_consequence', 'phys_health_consequence',
        # 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

    # 1. Start with our target variable: treatment
    #todo check all these graph styles are consistent
    sns.set_style("white")
    treatment_percent = round(train['treatment'].value_counts(normalize = True).reset_index(name = 'percentage'),3)
    ax = sns.barplot(x = 'treatment', y = 'percentage', data = treatment_percent, palette=colour_palette, alpha=0.7)
    # Annotate bars with the percentages (determines from height rounded to 1 dp)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() #todo changed the labels so chekc
        ax.annotate(f'{height:.1%}', (x + p.get_width() / 2, height + 0.7), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Sought treatment", fontsize=12)
    ax.set_ylabel("Percentage", fontsize=12)
    plt.title('Treatment percentage', fontsize=14, fontweight='bold')
    plt.show()
    # Result: nearly 50/50 sought vs didn't seek treatment
        # Important that we find all the true positives as we want to be able to help everyone - consider this when building the model
        # No class imbalance so don't need to resample TODO did i do this somewhere? I thought I did but can't remember

    # 2. Age column
    # Plot basics
    sns.set_style("white")
    num_bins = 18
    ax = sns.histplot(x = 'age', data = train, bins = num_bins, hue='treatment', palette=colour_palette, alpha=0.7)

    # Plot settings
    ax.set_xlabel("Age (years)", fontsize=12)
    plt.title('Age of participants - overlapping distributions', fontsize=14, fontweight='bold')
    plt.show()

    # Result: Similar distributions between treatment classes so not very useful as a predictor

    # 3. Gender
    # Calculate percentage data
    gender_treatment_percent = train.groupby(['gender', 'treatment']).size().reset_index(name='count')
    gender_treatment_percent['percentage'] = (gender_treatment_percent['count'] / gender_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='gender', data=gender_treatment_percent, hue='treatment',
                      palette=colour_palette, multiple="dodge", alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.7), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Gender of participants", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Gender', fontsize=14, fontweight='bold')
    plt.show()

    # Results: There is a notable class imbalance, so we need to be wary of saying that men are more prone to mental illness.
        # TODO not really sure what to do with this. Is it a feature?

    # 4. Self-employed
    # Calculate percentage data
    self_employed_treatment_percent = train.groupby(['self_employed', 'treatment']).size().reset_index(name='count')
    self_employed_treatment_percent['percentage'] = (self_employed_treatment_percent['count'] / self_employed_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='self_employed', data=self_employed_treatment_percent, hue='treatment',
                      palette=colour_palette, multiple="dodge", alpha=0.6, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.6), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Self-employment status", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Self-employment', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Vast majority are not self-employed. We see a similar treatment ratio in each category; not very useful for model building.
             # There is imbalance of the categories but due to similar treatment distribution it might not affect the model at all

    # 5. Family History
    # Calculate percentage data
    family_history_treatment_percent = train.groupby(['family_history', 'treatment']).size().reset_index(name='count')
    family_history_treatment_percent['percentage'] = (family_history_treatment_percent['count'] / family_history_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='family_history', data=family_history_treatment_percent, hue='treatment',
                      palette=colour_palette, multiple="dodge", alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.7), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Family History", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Is there a family history of mental illness?', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Those with a family history are more likely to seek treatment - making f_h an important feature

    # 6. Does your mental health condition (if present) interfere with your work?
    # Order categories by frequency #todo perhaps could combine this with... one hot encoding? which i might do later
    category_order = ['Never', 'Rarely', 'Sometimes', 'Often']
    train['work_interfere'] = pd.Categorical(train['work_interfere'], categories=category_order, ordered=True)

    # Calculate percentage data
    work_interfere_treatment_percent = train.groupby(['work_interfere', 'treatment']).size().reset_index(name='count')
    work_interfere_treatment_percent['percentage'] = (work_interfere_treatment_percent['count'] / work_interfere_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='work_interfere', data=work_interfere_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.7), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Work interference", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('If you have a mental health condition, does it interfere with your work?\n Overlapping distribution', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Vast majority of respondents report work interference. Those who have sought treatment skew more towards the higher frequency than those untreated
        # TODO should we omit this? Idk it's an odd variable to predict having sought treatment anyway. But this group have already sought treatment so... it feels odd to me

    # 7. Number of employees at the company
    # Order categories by frequency #todo perhaps could combine this with... one hot encoding? which i might do later
    category_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
    train['no_employees'] = pd.Categorical(train['no_employees'], categories=category_order, ordered=True)

    # Calculate percentage data
    no_employees_treatment_percent = train.groupby(['no_employees', 'treatment']).size().reset_index(name='count')
    no_employees_treatment_percent['percentage'] = (no_employees_treatment_percent['count'] / no_employees_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='no_employees', data=no_employees_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Number of employees", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Number of employees at the company\n Overlapping distribution', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Similar distribution for treatment yes/no

    # 8. Do you work remotely >50% of the time?
    # Calculate percentage data
    remote_work_treatment_percent = train.groupby(['remote_work', 'treatment']).size().reset_index(name='count')
    remote_work_treatment_percent['percentage'] = (remote_work_treatment_percent['count'] / remote_work_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='remote_work', data=remote_work_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Removed annotations due to overlap - could be re-added and adjusted

    # Plot settings
    ax.set_xlabel("(Majority) Remote worker", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Do oyu work remotely more than 50% of the time?', fontsize=14, fontweight='bold')
    plt.show()

    # Result: Very similar ratios between treatment classes.

    # 9. Do you work at a tech company?
    # Calculate percentage data
    tech_company_treatment_percent = train.groupby(['tech_company', 'treatment']).size().reset_index(name='count')
    tech_company_treatment_percent['percentage'] = (tech_company_treatment_percent['count'] / tech_company_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='tech_company', data=tech_company_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Tech company", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Do you work at a tech company?', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Very similar ratios between treatment classes.


    # 10. Does your employer provide mental health benefits?
    # Calculate percentage data
    benefits_treatment_percent = train.groupby(['benefits', 'treatment']).size().reset_index(name='count')
    benefits_treatment_percent['percentage'] = (benefits_treatment_percent['count'] / benefits_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='benefits', data=benefits_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')
#todo this and others i think say overlapping distribution but either aren't or shouldn't be! so check them all at the end
    # Plot settings
    ax.set_xlabel("Benefits", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Does your employer offer benefits?\n Overlapping distribution', fontsize=14, fontweight='bold')
    plt.show()

    # Results: 'Yes' has far more employees seeking treatment, 'No' about even, and don't know has the lowest ratio of those seeking treatment
        # Perhaps those who don't yet know about MH facilities provided by their employer are least likely to also seek independent care

    # 11. Do you know the options for mental health care your employer provides?
    # Calculate percentage data
    care_options_treatment_percent = train.groupby(['care_options', 'treatment']).size().reset_index(name='count')
    care_options_treatment_percent['percentage'] = (care_options_treatment_percent['count'] / care_options_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='care_options', data=care_options_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Care options", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Do you know about the mental health care options your employer provides?\n Overlapping distribution', fontsize=14, fontweight='bold')
    plt.show()

    # Results: 'Yes' were most likely to seek treatment with the majority doing so, and vice versa for 'No'. 'Not sure' mostly haven't sought
        # out treatment, although with a more even ratio than 'No'
        # (Also true for previous) Those seeking treatment might be more likely to investigate health care options - a good metric of inidivudals who should be offered care.
        # Alternatively, being aware of these options may increase tendency to use them
        # Another insight is that for both this and the benefits, it seems like awareness of options could be increased by the companies

    # 12. Has your employer ever discussed mental health as part of an employee wellness program?
    # Calculate percentage data
    wellness_program_treatment_percent = train.groupby(['wellness_program', 'treatment']).size().reset_index(name='count')
    wellness_program_treatment_percent['percentage'] = (wellness_program_treatment_percent['count'] / wellness_program_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='wellness_program', data=wellness_program_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage', multiple="dodge")

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Wellness program", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Has your employer ever discussed mental health as part of an employee wellness program?', fontsize=14, fontweight='bold')
    plt.show()

    # Results: Fairly even treatment ratio among groups, with 'Yes' being slightly more likely to seek treatment (although would need to actually statistically test).
        # 'Don't know' had the highest not-treated ratio. Possibly due to those who do not feel adverse affects on their mental health and subsequently didn't remember if it was offered
        # If so, this could be another artefact of grouping those who haven't been able to seek treatment with those who don't feel the need; 'treatment' here should not be taken
        # as a proxy for having mental illness as it does not include all the participants who do have adversely affected mental health

    # 13.  Does your employer provide resources to learn more about mental health issues and how to seek help?
    # Calculate percentage data
    seek_help_treatment_percent = train.groupby(['seek_help', 'treatment']).size().reset_index(name='count')
    seek_help_treatment_percent['percentage'] = (seek_help_treatment_percent['count'] / seek_help_treatment_percent['count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='seek_help', data=seek_help_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage', multiple="dodge")

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Resources to seek help", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Does your employer provide resources to learn more about mental health issues and how to seek help?', fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Similar pattern to previous; yes being more likely to seek help, and don't know the least, For 'No' it could also be that
        # employees are unaware. = Need to increase awareness

    # 14. Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
    # Calculate percentage data
    anonymity_treatment_percent = train.groupby(['anonymity', 'treatment']).size().reset_index(name='count')
    anonymity_treatment_percent['percentage'] = (anonymity_treatment_percent['count'] / anonymity_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='anonymity', data=anonymity_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage', multiple="dodge")

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Anonymity", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Similar trend to previous, showing anonymity is important to being able to seek help. Big proportion are unsure which needs addressing

    # 15. How easy is it for you to take medical leave for a mental health condition?
    # Order categories by frequency
    category_order = [ 'Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"]
    train['leave'] = pd.Categorical(train['leave'], categories=category_order, ordered=True)

    # Calculate percentage data
    leave_treatment_percent = train.groupby(['leave', 'treatment']).size().reset_index(name='count')
    leave_treatment_percent['percentage'] = (leave_treatment_percent['count'] / leave_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='leave', data=leave_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Removed annotations due to overlap - could be re-added and adjusted

    # Plot settings
    ax.set_xlabel("Leave", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.xticks()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('How easy is it for you to take medical leave for a mental health condition?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Again, don't know had the largest proportion of non-treated. The higher difficulties also had a higher proportion of
        # those seeking treatment; allowing more MH leave could reduce emergence of MH issues

    # 16. Do you think that discussing a mental health issue with your employer would have negative consequences?
    # Order categories by frequency
    category_order = [ 'No', 'Maybe', 'Yes']
    train['mental_health_consequence'] = pd.Categorical(train['mental_health_consequence'], categories=category_order, ordered=True)

    # Calculate percentage data
    mental_health_consequence_treatment_percent = train.groupby(['mental_health_consequence', 'treatment']).size().reset_index(name='count')
    mental_health_consequence_treatment_percent['percentage'] = (mental_health_consequence_treatment_percent['count'] / mental_health_consequence_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='mental_health_consequence', data=mental_health_consequence_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Consequences", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.xticks()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Do you think that discussing a mental health issue with your employer would have negative consequences?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Again, we see the highest proportion of those seeking treatment in 'Yes' and least in 'No'. Suggesting a less
         # accepting attitude towards MH issues exacerbating the issue

    # 17. Do you think that discussing a physical health issue with your employer would have negative consequences?
    # Order categories by frequency
    category_order = [ 'No', 'Maybe', 'Yes']
    train['phys_health_consequence'] = pd.Categorical(train['phys_health_consequence'], categories=category_order, ordered=True)

    # Calculate percentage data
    phys_health_consequence_treatment_percent = train.groupby(['phys_health_consequence', 'treatment']).size().reset_index(name='count')
    phys_health_consequence_treatment_percent['percentage'] = (phys_health_consequence_treatment_percent['count'] / phys_health_consequence_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='phys_health_consequence', data=phys_health_consequence_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Consequences", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.xticks()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.title('Do you think that discussing a physical health issue with your employer would have negative consequences?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Very similar distrubtion for treatment classes. A greater majority than with mental health think this wouldn't be a problem

    # 17. Would you be willing to discuss a mental health issue with your coworkers?
    # Order categories by frequency
    category_order = [ 'No', 'Some of them', 'Yes']
    train['coworkers'] = pd.Categorical(train['coworkers'], categories=category_order, ordered=True)

    # Calculate percentage data
    coworkers_treatment_percent = train.groupby(['coworkers', 'treatment']).size().reset_index(name='count')
    coworkers_treatment_percent['percentage'] = (coworkers_treatment_percent['count'] / coworkers_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='coworkers', data=coworkers_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Willing to discuss with coworkers", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Would you be willing to discuss a mental health issue with your coworkers?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Those who sought treatment are more likely to discuss with coworkers; possibly indicative of being more open to discussion/willing to
        # seek help, or a more welcoming work environment. Vast majority fall in some/yes

    # 18. Would you be willing to discuss a mental health issue with your direct supervisor(s)?
    # Order categories by frequency
    category_order = [ 'No', 'Some of them', 'Yes']
    train['supervisor'] = pd.Categorical(train['supervisor'], categories=category_order, ordered=True)

    # Calculate percentage data
    supervisor_treatment_percent = train.groupby(['supervisor', 'treatment']).size().reset_index(name='count')
    supervisor_treatment_percent['percentage'] = (supervisor_treatment_percent['count'] / supervisor_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='supervisor', data=supervisor_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Willing to discuss with supervisor(s)", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Would you be willing to discuss a mental health issue with your direct supervisor(s)?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Proprotion of Yes/Some is fewer than coworkers. Trend is reversed with those saying 'No' more likely to seek treatment

    # 19. Would you bring up a mental health issue with a potential employer in an interview?
    # Order categories by frequency
    category_order = ['No', 'Maybe', 'Yes']
    train['mental_health_interview'] = pd.Categorical(train['mental_health_interview'], categories=category_order, ordered=True)

    # Calculate percentage data
    mental_health_interview_treatment_percent = train.groupby(['mental_health_interview', 'treatment']).size().reset_index(name='count')
    mental_health_interview_treatment_percent['percentage'] = (mental_health_interview_treatment_percent['count'] / mental_health_interview_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='mental_health_interview', data=mental_health_interview_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Removed annotations due to overlap - could be re-added and adjusted

    # Plot settings
    ax.set_xlabel("Mention mental health in interview", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Would you bring up a mental health issue with a potential employer in an interview?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Vast majority say No

    # 20. Would you bring up a physical health issue with a potential employer in an interview?
    # Order categories by frequency
    category_order = ['No', 'Maybe', 'Yes']
    train['phys_health_interview'] = pd.Categorical(train['phys_health_interview'], categories=category_order, ordered=True)

    # Calculate percentage data
    phys_health_interview_treatment_percent = train.groupby(['phys_health_interview', 'treatment']).size().reset_index(name='count')
    phys_health_interview_treatment_percent['percentage'] = (phys_health_interview_treatment_percent['count'] / phys_health_interview_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='phys_health_interview', data=phys_health_interview_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage')

    # Removed annotations due to overlap - could be re-added and adjusted

    # Plot settings
    ax.set_xlabel("Mention physical health in interview", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Would you bring up a physical health issue with a potential employer in an interview?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: More in maybe than yes with pyhsical health; indicative of perceived cultural difference

    # 21. Do you feel that your employer takes mental health as seriously as physical health?
    # Order categories by frequency
    category_order = ['No', 'Maybe', 'Yes']
    train['mental_vs_physical'] = pd.Categorical(train['mental_vs_physical'], categories=category_order, ordered=True)

    # Calculate percentage data
    mental_vs_physical_treatment_percent = train.groupby(['mental_vs_physical', 'treatment']).size().reset_index(name='count')
    mental_vs_physical_treatment_percent['percentage'] = (mental_vs_physical_treatment_percent['count'] / mental_vs_physical_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='mental_vs_physical', data=mental_vs_physical_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage', multiple="dodge")

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Takes mental health seriously", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Do you feel that your employer takes mental health as seriously as physical health?',
              fontsize=14,
              fontweight='bold')
    plt.show()

    # Results: Same proportion of treated and untreated for yes, but those that say no are more likely to be those seeking treatment


    # 22. Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
    # Calculate percentage data
    obs_consequence_treatment_percent = train.groupby(['obs_consequence', 'treatment']).size().reset_index(name='count')
    obs_consequence_treatment_percent['percentage'] = (obs_consequence_treatment_percent['count'] / obs_consequence_treatment_percent[
        'count'].sum()) * 100

    # Plot the histogram
    sns.set_style("white")
    ax = sns.histplot(x='obs_consequence', data=obs_consequence_treatment_percent, hue='treatment',
                      palette=colour_palette, alpha=0.7, weights='percentage', multiple="dodge")

    # Annotate bars with the percentages (determined from bar height)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + p.get_width() / 2, height + 0.2), ha='center', fontweight='bold')

    # Plot settings
    ax.set_xlabel("Observed consequences", fontsize=12)
    ax.set_ylabel("Percentage of study group", fontsize=12)
    plt.title('Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?',
              fontsize=14,
              fontweight='bold')
    plt.show()

# Results: Those who seek treatment are more likely to say yes, but vast majority is no






# continue with eda, checking tutorials. haven't done the conclusion part for this yet




#todo remove self employed work interfere missing values

# todo there's also a few ordinal categories which i think should be encoded numerically (see todos above)


#feature importance is probably the key here - we want to focus on what allows employees to be able to seek treatment, not whether they do

