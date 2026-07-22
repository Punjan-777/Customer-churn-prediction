import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        self.data = pd.read_csv(self.file_path)
        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def get_basic_info(self):
        """Get basic information about the dataset"""
        print("\n=== Dataset Info ===")
        print(f"Shape: {self.data.shape}")
        print("\n=== Column Names ===")
        print(self.data.columns.tolist())
        print("\n=== Data Types ===")
        print(self.data.dtypes)
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
        print("\n=== Statistical Summary ===")
        print(self.data.describe())
        
    def get_target_distribution(self):
        """Check target variable distribution"""
        churn_counts = self.data['Churn'].value_counts()
        churn_percentage = (self.data['Churn'].value_counts(normalize=True) * 100)
        print("\n=== Churn Distribution ===")
        print(f"Churn Counts:\n{churn_counts}")
        print(f"\nChurn Percentage:\n{churn_percentage}")
        return churn_counts, churn_percentage