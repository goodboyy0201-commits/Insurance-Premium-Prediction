# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Scikit-learn for preprocessing, model selection, and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Machine Learning Model
import lightgbm as lgb

# General settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

print("Libraries imported successfully!")

# =============================================================================
# 2. LOAD AND EXPLORE DATA
# =============================================================================
print("\nLoading data...")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

# Store IDs and target for later
train_ids = train_df['id']
test_ids = test_df['id']
# The target variable is 'Premium Amount'
train_target = train_df['Premium Amount']

# Drop unnecessary columns for training
train_df = train_df.drop(['id', 'Premium Amount'], axis=1)
test_df = test_df.drop('id', axis=1)

print("Data loaded. Train shape:", train_df.shape, "Test shape:", test_df.shape)

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\nStarting Exploratory Data Analysis (EDA)...")

# --- IMPORTANT: Analyze the target variable distribution ---
plt.figure(figsize=(14, 5))

# Plot original distribution
plt.subplot(1, 2, 1)
sns.histplot(train_target, kde=True, bins=50)
plt.title('Distribution of Original Premium Amount (Skewed)')

# Plot log-transformed distribution
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_target), kde=True, bins=50)
plt.title('Distribution of Log-Transformed Premium Amount (More Normal)')
plt.xlabel('Log(1 + Premium Amount)')

plt.show()
# The log-transformed target looks much more like a normal distribution, which is ideal for many models.

# =============================================================================
# 4. DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
print("\nStarting data preprocessing...")

# --- Crucial Step: Log-transform the target variable ---
y_log = np.log1p(train_target)
print("Target variable transformed using log1p.")

# Combine train and test for easy preprocessing
full_df = pd.concat([train_df, test_df], axis=0)

# --- Handle Categorical Features using One-Hot Encoding ---
# get_dummies will automatically find and convert object-type columns
print("Applying One-Hot Encoding...")
full_df = pd.get_dummies(full_df, drop_first=True)

# Separate back into train and test sets
X = full_df[:len(train_df)]
X_test = full_df[len(train_df):]

print("Preprocessing complete. New shape of X:", X.shape)

# =============================================================================
# 5. MODEL TRAINING AND VALIDATION
# =============================================================================
print("\nTraining a LightGBM Regressor model...")

# Split the data to create a validation set
# We use the log-transformed target for training
X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Initialize the LightGBM Regressor
lgbm_reg = lgb.LGBMRegressor(random_state=42)

# Train the model
lgbm_reg.fit(X_train, y_train_log)

# Make predictions on the validation set (predictions will be on the log scale)
val_preds_log = lgbm_reg.predict(X_val)

# --- Evaluate the model using RMSLE ---
# To calculate RMSLE, we need to convert predictions back to the original scale
val_preds_original = np.expm1(val_preds_log)

# We also need the original, untransformed validation target values
y_val_original = np.expm1(y_val_log)

# Ensure no negative predictions (important for RMSLE)
val_preds_original[val_preds_original < 0] = 0

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val_original, val_preds_original))
print(f"\nValidation RMSLE: {rmsle}")

# =============================================================================
# 6. CREATE SUBMISSION FILE
# =============================================================================
print("\nCreating submission file...")

# Retrain the model on the FULL training dataset
lgbm_reg.fit(X, y_log)

# Predict on the test data (predictions are on the log scale)
test_preds_log = lgbm_reg.predict(X_test)

# --- IMPORTANT: Convert predictions back to the original scale ---
final_predictions = np.expm1(test_preds_log)

# Ensure no negative predictions
final_predictions[final_predictions < 0] = 0

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_ids, 'Premium Amount': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Top 5 rows of submission file:")
print(submission_df.head())
