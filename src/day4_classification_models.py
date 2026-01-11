"""Day 4: Classification Models Training and Evaluation (IMPROVED)
Build and evaluate multiple classification models with proper train/val/test split.
Includes Logistic Regression, Random Forest, XGBoost, and SVM with hyperparameter tuning.
"""
import numpy as np
import pandas as pd
import warnings
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

warnings.filterwarnings('ignore')

# Create output directories
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('visualizations').mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True)

class ClassificationModelsImproved:
    """Classification models trainer with proper train/val/test split and hyperparameter tuning"""
    
    def __init__(self, data_path='output/data_engineered.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.cv_results = {}
        
    def load_data(self):
        """Load engineered data"""
        print("Loading engineered data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        return self.df
    
    def prepare_data(self):
        """Prepare data with proper train/val/test split (60/15/25)"""
        print("\n" + "="*70)
        print("PREPARING DATA WITH TRAIN/VAL/TEST SPLIT")
        print("="*70)
        
        # Separate features and target
        X = self.df.drop(['Good_Investment', 'Future_Price_5Y'], axis=1, errors='ignore')
        
        # Drop original categorical columns
        categorical_cols = ['State', 'City', 'Property_Type', 'Furnished_Status', 
                          'Owner_Type', 'Availability_Status', 'Facing', 'Security']
        X = X.drop(columns=[col for col in categorical_cols if col in X.columns], errors='ignore')
        y = self.df['Good_Investment']
        
        print(f"Features: {X.shape[1]}, Target classes: {y.nunique()}")
        print(f"\nClass distribution:\n{y.value_counts()}")
        
        # IMPROVED: Proper train/val/test split (60% train, 15% val, 25% test)
        # Step 1: Split into train (60%) and temp (40%)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        # Step 2: Split temp into val (15%) and test (25%)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp
        )
        
        total = len(X)
        print(f"\n{'Dataset':<15} {'Samples':<15} {'Percentage':<15}")
        print("-" * 45)
        print(f"{'Train':<15} {self.X_train.shape[0]:<15} {self.X_train.shape[0]/total*100:>6.1f}%")
        print(f"{'Validation':<15} {self.X_val.shape[0]:<15} {self.X_val.shape[0]/total*100:>6.1f}%")
        print(f"{'Test':<15} {self.X_test.shape[0]:<15} {self.X_test.shape[0]/total*100:>6.1f}%")
        
        # Feature scaling (fit ONLY on training data to prevent leakage)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Features scaled (fit only on training data)")
        
    def train_logistic_regression(self):
        """Train Logistic Regression with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*70)
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10],
            'max_iter': [500, 1000],
            'solver': ['lbfgs', 'liblinear']
        }
        
        base_model = LogisticRegression(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print(f"GridSearchCV with {len(param_grid['C']) * len(param_grid['max_iter']) * len(param_grid['solver'])} combinations...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.models['Logistic Regression'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [5, 10]
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print(f"GridSearchCV with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} combinations...")
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING XGBOOST")
        print("="*70)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 7],
            'learning_rate': [0.01, 0.1]
        }
        
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=1)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print(f"GridSearchCV with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} combinations...")
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
    def train_svm(self):
        """Train SVM with hyperparameter tuning"""
        print("\n" + "="*70)
        print("TRAINING SUPPORT VECTOR MACHINE")
        print("="*70)
        
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        
        base_model = SVC(kernel='rbf', probability=True, random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        
        print(f"GridSearchCV with {len(param_grid['C']) * len(param_grid['gamma'])} combinations...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.models['SVM'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
    def evaluate_on_set(self, model, X, y, set_name):
        """Evaluate model on a specific dataset"""
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
            y_pred_proba = model.decision_function(X)
        
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'F1-Score': f1_score(y, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y, y_pred_proba),
        }
        
        return metrics, y_pred, y_pred_proba
        
    def evaluate_models(self):
        """Evaluate all models on train/val/test sets"""
        print("\n" + "="*70)
        print("EVALUATING MODELS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 50)
            
            # Use scaled data for LR and SVM, normal data for tree-based
            use_scaled = name in ['Logistic Regression', 'SVM']
            
            if use_scaled:
                X_train, X_val, X_test = self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
            else:
                X_train, X_val, X_test = self.X_train, self.X_val, self.X_test
            
            # Evaluate on all three sets
            train_metrics, train_pred, train_proba = self.evaluate_on_set(
                model, X_train, self.y_train, "Train"
            )
            val_metrics, val_pred, val_proba = self.evaluate_on_set(
                model, X_val, self.y_val, "Validation"
            )
            test_metrics, test_pred, test_proba = self.evaluate_on_set(
                model, X_test, self.y_test, "Test"
            )
            
            # Store results
            self.results[name] = {
                'Train': {**train_metrics, 'y_pred': train_pred, 'y_pred_proba': train_proba},
                'Val': {**val_metrics, 'y_pred': val_pred, 'y_pred_proba': val_proba},
                'Test': {**test_metrics, 'y_pred': test_pred, 'y_pred_proba': test_proba}
            }
            
            # Print metrics
            for set_name, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
                print(f"{set_name:>10} -> Acc: {metrics['Accuracy']:.4f} | "
                      f"Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | "
                      f"F1: {metrics['F1-Score']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
    def save_models(self):
        """Save trained models and scaler"""
        print("\n" + "="*70)
        print("SAVING MODELS AND SCALER")
        print("="*70)
        
        for name, model in self.models.items():
            filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filename}")
        
        # IMPROVED: Save scaler for production use
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: models/scaler.pkl")
        
    def save_results(self):
        """Save evaluation results"""
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        # Create results dataframe
        results_summary = []
        for model_name, sets in self.results.items():
            for set_name, metrics in sets.items():
                metrics_only = {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_pred_proba']}
                results_summary.append({
                    'Model': model_name,
                    'Dataset': set_name,
                    **metrics_only
                })
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('results/classification_metrics.csv', index=False)
        print(f"✓ Saved: results/classification_metrics.csv")
        
        print("\nMetrics Summary:")
        print(results_df.to_string(index=False))
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for test set"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, name in enumerate(self.models.keys()):
            test_pred = self.results[name]['Test']['y_pred']
            cm = confusion_matrix(self.y_test, test_pred)
            sns.heatmap
