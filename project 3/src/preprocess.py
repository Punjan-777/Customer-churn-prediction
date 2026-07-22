import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.numerical_features = []
        self.categorical_features = []
        self.target = 'Churn'
        
    def clean_data(self):
        """Clean and preprocess the data"""
        # Remove customerID column (not useful for prediction)
        if 'customerID' in self.data.columns:
            self.data.drop('customerID', axis=1, inplace=True)
        
        # Convert TotalCharges to numeric (it might be string)
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with median
        self.data['TotalCharges'].fillna(self.data['TotalCharges'].median(), inplace=True)
        
        # Encode target variable
        self.data[self.target] = self.data[self.target].map({'Yes': 1, 'No': 0})
        
        print("Data cleaning completed successfully!")
        return self.data
    
    def identify_features(self):
        """Identify numerical and categorical features"""
        self.numerical_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numerical_features.remove(self.target)
        
        self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumerical features: {self.numerical_features}")
        print(f"Categorical features: {self.categorical_features}")
        
        return self.numerical_features, self.categorical_features
    
    def create_pipeline(self):
        """Create preprocessing pipeline"""
        # Create preprocessing for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('encoder', LabelEncoder())
        ])
        
        # Combine into column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def prepare_features(self):
        """Prepare features for training"""
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        # Encode categorical features manually
        for col in self.categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale numerical features
        scaler = StandardScaler()
        X[self.numerical_features] = scaler.fit_transform(X[self.numerical_features])
        
        return X, y