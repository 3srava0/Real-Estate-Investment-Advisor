"""Day 5: Regression Models Training and Evaluation

Build and evaluate multiple regression models for predicting Future_Price_5Y target.
Includes Linear Regression, Ridge, Lasso, XGBoost, and Random Forest regressors.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('visualizations').mkdir(exist_ok=True)

class RegressionModels:
    def __init__(self, data_path='output/data_engineered.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        print("Loading engineered data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
    
    def prepare_data(self):
        print("\nPreparing data...")
        X = self.df.drop(['Good_Investment', 'Future_Price_5Y'], axis=1, errors='ignore')
        y = self.df['Future_Price_5Y']
        print(f"Features: {X.shape[1]}, Target Mean: {y.mean():.2f}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_linear_regression(self):
        print("\nTraining Linear Regression...")
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = model
        
    def train_ridge_lasso(self):
        print("\nTraining Ridge & Lasso...")
        ridge = Ridge(alpha=10.0)
        ridge.fit(self.X_train_scaled, self.y_train)
        self.models['Ridge'] = ridge
        
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train_scaled, self.y_train)
        self.models['Lasso'] = lasso
        
    def train_xgboost(self):
        print("\nTraining XGBoost...")
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        self.feature_importance_xgb = model.feature_importances_
        
    def train_random_forest(self):
        print("\nTraining Random Forest...")
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        self.feature_importance_rf = model.feature_importances_
        
    def evaluate_models(self):
        print("\nEvaluating models...")
        for name, model in self.models.items():
            if 'Regression' in name or name in ['Ridge', 'Lasso']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'y_pred': y_pred}
            print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")
            
    def save_models(self):
        for name, model in self.models.items():
            Path(f'models/{name.lower()}_model.pkl').write_bytes(pickle.dumps(model))
            
    def save_results(self):
        pd.DataFrame(self.results).T.to_csv('results/regression_metrics.csv')
        
    def plot_all(self):
        # Actual vs Predicted
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for idx, (name, _) in enumerate(self.models.items()):
            if idx < 6:
                y_pred = self.results[name]['y_pred']
                axes[idx].scatter(self.y_test, y_pred, alpha=0.5)
                axes[idx].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
                axes[idx].set_title(f'{name}')
        plt.tight_layout()
        plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run(self):
        print("="*50 + "\nDAY 5: REGRESSION MODELS\n" + "="*50)
        self.load_data()
        self.prepare_data()
        self.train_linear_regression()
        self.train_ridge_lasso()
        self.train_xgboost()
        self.train_random_forest()
        self.evaluate_models()
        self.save_models()
        self.save_results()
        self.plot_all()
        print("\n" + "="*50 + "\nRegression models completed!\n" + "="*50)

if __name__ == '__main__':
    RegressionModels().run()
