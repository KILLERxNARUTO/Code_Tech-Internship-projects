# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# Function to load dataset from a given CSV path
def load_data(path):
    df = pd.read_csv(path)
    return df

# Function to split the dataframe into features (X) and target (y)
def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])  # Drop the target column to get features
    y = df[target_column]  # Isolate the target column
    return X, y

# Function to build preprocessing pipeline for numeric and categorical columns
def build_preprocessing_pipeline(X):
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline for numeric data: impute missing values with mean and scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical data: impute with most frequent and one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine numeric and categorical pipelines into a single preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

# Function to save the processed datasets to disk
def save_data(X_train, X_test, y_train, y_test, output_dir="processed_data"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save feature arrays in .npy format
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)

    # Save target series in .csv format
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Data saved in folder: '{output_dir}'")

# Main function to run the entire preprocessing pipeline
def run_pipeline(csv_path, target_column):
    # Load the raw dataset
    df = load_data(csv_path)

    # Split data into features and target
    X, y = split_features_target(df, target_column)

    # Build and apply preprocessing
    preprocessor = build_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    print("Data preprocessing complete.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Save the processed datasets
    save_data(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

# Entry point: specify input file and target column, then run pipeline
if __name__ == "__main__":
    csv_path = "sample_data (1).csv"  # Path to your input CSV file
    target_column = "SalePrice"       # Name of the column to predict
    X_train, X_test, y_train, y_test = run_pipeline(csv_path, target_column)
