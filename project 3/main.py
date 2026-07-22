import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.model_training import ModelTrainer
from src.utils import *
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("="*60)
    
    # Step 1: Load Data
    print("\n1. Loading Data...")
    loader = DataLoader('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data = loader.load_data()
    loader.get_basic_info()
    loader.get_target_distribution()
    
    # Step 2: Preprocess Data
    print("\n2. Preprocessing Data...")
    preprocessor = DataPreprocessor(data)
    clean_data = preprocessor.clean_data()
    numerical_features, categorical_features = preprocessor.identify_features()
    
    # Show some visualizations
    print("\n3. Exploratory Data Analysis...")
    plot_churn_distribution(clean_data)
    plot_tenure_vs_churn(clean_data)
    plot_contract_type_analysis(clean_data)
    
    # Step 3: Prepare features for training
    print("\n4. Preparing Features for Training...")
    X, y = preprocessor.prepare_features()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Step 4: Train Models
    print("\n5. Training Models...")
    trainer = ModelTrainer(X, y)
    trainer.split_data()
    trainer.initialize_models()
    trainer.train_models()
    trainer.evaluate_all_models()
    trainer.find_best_model()
    
    # Step 5: Feature Importance
    print("\n6. Analyzing Feature Importance...")
    if hasattr(trainer.best_model, 'feature_importances_'):
        feature_importance = get_feature_importance(trainer.best_model, X.columns.tolist())
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Step 6: Save Model
    print("\n7. Saving Model...")
    trainer.save_model('models/churn_model.pkl')
    trainer.save_metrics('models/metrics.txt')
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()