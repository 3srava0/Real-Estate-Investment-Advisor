"""Day 4: Classification Models Training and Evaluation

Build and evaluate multiple classification models for predicting Good_Investment target.
Includes Logistic Regression, Random Forest, XGBoost, and SVM.
"""

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Create output directories
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('visualizations').mkdir(exist_ok=True)
Path('output').mkdir(exist_ok=True)

class ClassificationModels:
    """Classification models trainer and evaluator"""
    
    def __init__(self, data_path='output/data_engineered.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load engineered data"""
        print("Loading engineered data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        return self.df
    
    def prepare_data(self):
        """Prepare data for classification"""
        print("\nPreparing data...")
        
        # Separate features and target
        X = self.df.drop(['Good_Investment', 'Future_Price_5Y'], axis=1, errors='ignore')
        y = self.df['Good_Investment']
        
        print(f"Features: {X.shape[1]}, Target classes: {y.nunique()}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        # Train-test split (stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression"""
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = model
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def train_random_forest(self):
        """Train Random Forest"""
        print("\nTraining Random Forest...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        self.feature_importance_rf = model.feature_importances_
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def train_xgboost(self):
        """Train XGBoost"""
        print("\nTraining XGBoost...")
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        self.feature_importance_xgb = model.feature_importances_
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def train_svm(self):
        """Train SVM"""
        print("\nTraining SVM...")
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = model
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def evaluate_models(self):
        """Evaluate all models"""
        print("\nEvaluating models...")
        
        for name, model in self.models.items():
            # Get predictions
            if name in ['Logistic Regression', 'SVM']:
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        for name, model in self.models.items():
            filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {filename}")
            
    def save_results(self):
        """Save evaluation results"""
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('results/classification_metrics.csv')
        print(f"Saved: results/classification_metrics.csv")
        print(f"\nMetrics Summary:\n{results_df}")
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = self.results[name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')
            
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Saved: visualizations/confusion_matrices.png")
        plt.close()
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Classification Models')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
        print("Saved: visualizations/roc_curves.png")
        plt.close()
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Random Forest
        feature_names = self.X_train.columns
        importances_rf = self.feature_importance_rf
        idx_rf = np.argsort(importances_rf)[-10:]
        axes[0].barh(range(len(idx_rf)), importances_rf[idx_rf])
        axes[0].set_yticks(range(len(idx_rf)))
        axes[0].set_yticklabels([feature_names[i] for i in idx_rf])
        axes[0].set_title('Random Forest - Top 10 Features')
        axes[0].set_xlabel('Importance')
        
        # XGBoost
        importances_xgb = self.feature_importance_xgb
        idx_xgb = np.argsort(importances_xgb)[-10:]
        axes[1].barh(range(len(idx_xgb)), importances_xgb[idx_xgb])
        axes[1].set_yticks(range(len(idx_xgb)))
        axes[1].set_yticklabels([feature_names[i] for i in idx_xgb])
        axes[1].set_title('XGBoost - Top 10 Features')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: visualizations/feature_importance.png")
        plt.close()
        
    def run(self):
        """Run full pipeline"""
        print("="*50)
        print("DAY 4: CLASSIFICATION MODELS")
        print("="*50)
        
        self.load_data()
        self.prepare_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        self.evaluate_models()
        self.save_models()
        self.save_results()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_feature_importance()
        
        print("\n" + "="*50)
        print("Classification models training completed!")
        print("="*50)

if __name__ == '__main__':
    trainer = ClassificationModels()
    trainer.run()
