"""
Dragon Real Estate - Price Predictor
A machine learning model to predict real estate prices using the Boston Housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Load the dataset
print("Loading dataset...")
housing = pd.read_csv("house_data.csv")

print("\nDataset Info:")
print(housing.info())

print("\nDataset Statistics:")
print(housing.describe())

print("\nCHAS Distribution:")
print(housing['CHAS'].value_counts())

# Train-Test Splitting with Stratified Sampling
print("\n" + "="*50)
print("Splitting data into train and test sets...")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(f"Training set size: {len(strat_train_set)}")
print(f"Test set size: {len(strat_test_set)}")

# Correlation Analysis
print("\n" + "="*50)
print("Analyzing correlations with target variable (MEDV)...")
corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))

# Feature Engineering - Creating new feature
print("\n" + "="*50)
print("Creating new feature: TAXRM (TAX/RM ratio)...")
housing["TAXRM"] = housing["TAX"] / housing["RM"]
corr_matrix = housing.corr()
print("\nUpdated correlations:")
print(corr_matrix['MEDV'].sort_values(ascending=False))

# Prepare training data
print("\n" + "="*50)
print("Preparing training data...")
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Create preprocessing pipeline
print("Creating preprocessing pipeline...")
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Transform the data
housing_num_tr = my_pipeline.fit_transform(housing)
print("Data preprocessing complete!")

# Train the model
print("\n" + "="*50)
print("Training Random Forest Regressor...")
model = RandomForestRegressor(random_state=42)
model.fit(housing_num_tr, housing_labels)
print("Model training complete!")

# Evaluate on training data
print("\n" + "="*50)
print("Evaluating model on training data...")
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(f"Training RMSE: {rmse:.2f}")

# Cross-validation
print("\n" + "="*50)
print("Performing 10-fold cross-validation...")
scores = cross_val_score(model, housing_num_tr, housing_labels, 
                        scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print(f"Cross-validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean():.2f}")
print(f"Standard deviation: {rmse_scores.std():.2f}")

# Save the model
print("\n" + "="*50)
print("Saving model to Dragon.joblib...")
dump(model, 'Dragon.joblib')
print("Model saved successfully!")

# Test on test data
print("\n" + "="*50)
print("Testing model on test set...")
x_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(f"Test RMSE: {final_rmse:.2f}")
print(f"Test MSE: {final_mse:.2f}")

# Show sample predictions
print("\n" + "="*50)
print("Sample predictions vs actual values:")
print("Predicted | Actual")
print("-" * 20)
for pred, actual in list(zip(final_predictions[:10], y_test[:10])):
    print(f"{pred:8.2f} | {actual:6.2f}")

print("\n" + "="*50)
print("Model training and evaluation complete!")
print("="*50)
