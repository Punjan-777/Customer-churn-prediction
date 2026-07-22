import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def initialize_models(self):
        """Initialize different models"""
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
    def train_models(self):
        """Train all models"""
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics, y_pred
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for name, model in self.models.items():
            metrics, y_pred = self.evaluate_model(model, self.X_test, self.y_test)
            self.results[name] = metrics
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"  Confusion Matrix:\n{cm}")
            
    def find_best_model(self):
        """Find the best performing model"""
        best_score = 0
        for name, metrics in self.results.items():
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Accuracy: {best_score:.4f}")
        print(f"{'='*50}")
        
    def hyperparameter_tuning(self, model_name='XGBoost'):
        """Perform hyperparameter tuning on selected model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        print(f"\nHyperparameter Tuning for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)
        else:
            print("Tuning only available for XGBoost and Random Forest")
            return
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"{model_name} (Tuned)"
        
        # Evaluate tuned model
        metrics, _ = self.evaluate_model(self.best_model, self.X_test, self.y_test)
        print(f"Tuned model accuracy: {metrics['accuracy']:.4f}")
        
    def save_model(self, filepath='models/churn_model.pkl'):
        """Save the best model to disk"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"\nModel saved to {filepath}")
        
    def save_metrics(self, filepath='models/metrics.txt'):
        """Save evaluation metrics to file"""
        with open(filepath, 'w') as f:
            f.write("MODEL EVALUATION METRICS\n")
            f.write("="*50 + "\n\n")
            for name, metrics in self.results.items():
                f.write(f"{name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        print(f"Metrics saved to {filepath}")