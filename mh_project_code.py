
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load mental health dataset
mhdat = pd.read_csv("survey.csv")

# Preview dataset
mhdat

# Count NaN values in each column to determine which variables to drop from analysis
print(mhdat.isna().sum())

# Variables for analyses
analys_vars = [
    "family_history",
    "treatment",
    "remote_work",
    "tech_company",
    "leave",
    "obs_consequence",
    "mh_cond",
    "wellness_program",
    "mental_health_interview",
    "care_options",
    "benefits",
    "seek_help",
    
]

# Remove unnecessary columns
columns_to_remove = ['Timestamp', 'state', 'comments']
mhdat = mhdat.drop(columns=columns_to_remove)

# Assess gender variable
print(mhdat['gender'].unique())

# Cleaning gender variable
women_list = ['F', 'female', 'Woman', 'woman', 'Female', 'Femake',
            'femail', 'f', 'cis-female/femme', 'Cis Female']
                
men_list = ['M', 'm', 'male', 'Mail', 'maile', 'Mal', 'male',
            'Guy (-ish) ^_^', 'cis male', 'Cis Man', 'Cis Male',
             'Male ', 'Male (CIS)', 'male leaning androgynous',
             'Male-ish', 'Malr', 'Man', 'msle']


# Function to fix gender variable
def getGender(gender_type):
    gender_type = gender_type.strip().lower()
    if gender_type in women_list:
        return 'Female'
    elif gender_type in men_list:
        return 'Male'
    else:
        return 'unknown'

mhdat['Gender'] = mhdat['Gender'].apply(getGender)
mhdat = mhdat[mhdat['Gender'] != 'unknown']


# Function to fix age variable
def drop_invalid_age(df):
    df = df[(df['Age'] >= 18) & (df['Age'] <= 85)]
    df['Age'] = df['Age'].astype(int)
    return df

mhdat = drop_invalid_age(mhdat)


# Regroup company sizes to above or below 500 employees
def relabel_no_employees(df, column):
    mapping = {
        '1-5': 'below 500',
        '6-25': 'below 500',
        '26-100': 'below 500',
        '100-500': 'below 500',
        '500-1000': 'above 500',
        'More than 1000': 'above 500'
    }

    df[column] = df[column].map(mapping)

relabel_no_employees(mhdat, 'no_employees')


# Convert these vars to binary for analyses
def convert_yes_no_to_binary(df, columns):
    for col in columns:
        df[col] = df[col].str.strip().str.lower().map({'yes': 1, 'no': 0})
        
convert_yes_no_to_binary(mhdat, ['family_history', 'treatment', 'remote_work', 'tech_company', 'benefits',
                                 'care_options', 'wellness_program', 'seek_help', 'anonymity',
                                 'mental_health_consequence', 'phys_health_consequence', 'mental_health_interview',
                                 'phys_health_interview', 'mental_vs_physical', 'obs_consequence'])


# Relabel how easy it is to take leave into numeric
def relabel_leave(df, column):
    mapping = {
        'Very easy': '0',
        'Somewhat easy': '1',
        'Somewhat difficult': '2',
        'Very difficult': '3',
        "Don't know": "Don't know",
    }

    df[column] = df[column].map(mapping)

relabel_leave(mhdat, 'leave')


# convert work interference to dummy var that tells yes or no if subject has experienced mental health issues
def convert_work_interfere_to_binary(df):
    df['mh_cond'] = df['work_interfere'].apply(lambda x: 1 if pd.notna(x) else 0)

convert_work_interfere_to_binary(mhdat)



# Extra cleaning steps
#mhdat['mh_cond'].value_counts()
#mhdat.head(40)
#mhdat['no_employees'].value_counts()
#print(mhdat.dtypes)
# unique_fh = mhdat['family_history'].unique()
# print(unique_fh)
# unique_leave = mhdat['leave']
# print(unique_leave)
# unique_cow = mhdat['coworkers'].unique()
# print(unique_cow)
#pd.Categorical(mhdat.Gender, categories = ['Female', 'Male'])


# Display data types to identify non-numeric types
print(mhdat.dtypes)

# Convert non-numeric types to numeric where applicable
for column in analys_vars:
    if mhdat[column].dtype == 'object' or mhdat[column].dtype == 'bool':
        mhdat[column] = pd.to_numeric(mhdat[column], errors='coerce')

# Check for NaN values that might have been introduced or were already present
print(mhdat[analys_vars].isnull().sum())

# Fill NaN values
mhdat[analys_vars] = mhdat[analys_vars].fillna(mhdat[analys_vars].median())

# Confirm changes
print(mhdat[analys_vars].info())

# Create a bar plot for the 'no_employees' variable
plt.figure(figsize=(10, 6))
sns.countplot(data=mhdat, x='Gender', palette='pink')

# Add labels and title
plt.xlabel('Gender of Respondent')
plt.ylabel('Count')
plt.title('Distribution of Respondent Gender')

# Show plot
plt.show()

# Define the independent variables (exclude 'obs_consequence' and include others)
X_obs = mhdat[[var for var in analys_vars if var != 'obs_consequence' and var != 'mh_cond']]
y_obs = mhdat['obs_consequence']  # Dependent variable

# Add a constant to the model (intercept)
X_obs_const = sm.add_constant(X_obs)

# Fit the logistic regression model for 'obs_consequence'
model_obs = sm.Logit(y_obs, X_obs_const)
result_obs = model_obs.fit()
print("Regression Analysis for 'obs_consequence' as Dependent Variable:")
print(result_obs.summary())

# Define the independent variables (exclude 'obs_consequence' and include others)
X_obs = mhdat[[var for var in analys_vars if var != 'obs_consequence' and var != 'mh_cond']]
y_obs = mhdat['mh_cond']  # Dependent variable

# Add a constant to the model (intercept)
X_obs_const = sm.add_constant(X_obs)

# Fit the logistic regression model for 'obs_consequence'
model_obs = sm.Logit(y_obs, X_obs_const)
result_obs = model_obs.fit()
print("Regression Analysis for 'mh_cond' as Dependent Variable:")
print(result_obs.summary())

from scipy.stats import chi2_contingency

# Example for 'Gender' and 'mh_cond'
contingency_table = pd.crosstab(mhdat['obs_consequence'], mhdat['mh_cond'])
chi2, p, dof, ex = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}, p-value: {p}")

# Descriptive stats: mh_cond
mhdat['mh_cond'].describe()
# Percentage of respondants who have and havent had mental health conditions
mh_cond_perc= mhdat['mh_cond'].value_counts(normalize=True) * 100
print(mh_cond_perc)
# Calculate the counts and percentages
mh_counts = mhdat['mh_cond'].value_counts()
mh_cond_perc = mhdat['mh_cond'].value_counts(normalize=True) * 100

# Create plot
plt.figure(figsize=(8, 6))
sns.countplot(x='mh_cond', data=mhdat, palette='viridis')

# Update the x-axis labels from 0/1 to No/Yes
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.title('Distribution of Tech Workers reporting Mental Health Condition(s)')
plt.xlabel('Mental Health Condition')
plt.ylabel('Count')
plt.show()

# Create a bar plot of no_employees using seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='no_employees', data=mhdat, palette='viridis', order=sorted(mhdat['no_employees'].unique()))
plt.title('Distribution of Company Size')
plt.xlabel('Number of Employees')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Distribution of respondents by country
plt.figure(figsize=(12, 8))
sns.countplot(y='Country', data=mhdat, order=mhdat['Country'].value_counts().index, palette='coolwarm_r')
plt.title('Distribution of Respondents by Country')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# Plot of Age
plt.figure(figsize=(10, 6))
sns.histplot(mhdat['Age'], bins=10, kde=False, color='purple')
plt.title('Distribution of Respondents by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Calculate counts and percentages for each gender
gender_counts = mhdat['Gender'].value_counts()
gender_percentages = mhdat['Gender'].value_counts(normalize=True) * 100

# Create a bar plot showing gender counts
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='YlOrBr')
plt.title('Distribution of Respondents by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Include 'obs_consequence' and the independent variables in the subset
subset_vars = analys_vars + [dependent_var]
subset_data = mhdat[subset_vars]

# Create a pairplot
sns.pairplot(subset_data, diag_kind='kde', hue=dependent_var, palette='viridis')
plt.show()

# Compute the correlation matrix
correlation_matrix = mhdat[analys_vars + ['obs_consequence']].corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Encode categorical data
le = LabelEncoder()
mhdat_encoded = mhdat.apply(le.fit_transform)

# Fit K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(mhdat_encoded)

# Define the independent variables (exclude 'obs_consequence' and include others)
X_obs = mhdat[[var for var in analys_vars if var != 'obs_consequence' and var != 'mh_cond']]
y_obs = mhdat['obs_consequence']  # Dependent variable

# Add a constant to the model (intercept)
X_obs_const = sm.add_constant(X_obs)

# Fit the logistic regression model for 'obs_consequence'
model_obs = sm.Logit(y_obs, X_obs_const)
result_obs = model_obs.fit()
print("Regression Analysis for 'obs_consequence' as Dependent Variable:")
print(result_obs.summary())

# Assuming `coefficients` is a dictionary or array of logistic regression coefficients
coefficients = {
    'family_history': 0.4309,
    'leave': 0.3783,
    'wellness_program': 0.1869,
    # Add other coefficients here
}

# Calculate odds ratios
odds_ratios = {key: np.exp(value) for key, value in coefficients.items()}

print("Odds Ratios:")
for key, value in odds_ratios.items():
    print(f"{key}: {value:.2f}")

