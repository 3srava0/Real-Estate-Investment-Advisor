# DAY 5: Regression Models Training & Evaluation

## Objective
Build, train, and evaluate multiple regression models to predict the continuous "Future_Price_5Y" target variable. Implement hyperparameter tuning and cross-validation for regression tasks.

## Duration: 8 hours (1 day)

## Timeline
- **Hour 1-2**: Data preparation and feature scaling (Train-test split)
- **Hour 3-4**: Implement Linear Regression and Ridge/Lasso models
- **Hour 5-6**: Implement Gradient Boosting (XGBoost) and Random Forest regression
- **Hour 7-8**: Model comparison, evaluation metrics, and hyperparameter tuning

## Tasks

### 1. Data Preparation (1 hour)
- Load cleaned and engineered data from Day 2 output
- Apply feature scaling (StandardScaler or MinMaxScaler)
- Create random train-test split (80-20)
- Handle any outliers in target variable if necessary

### 2. Model Implementation (5 hours)

#### Linear Regression
- Simple linear regression model
- Fit with all features
- Calculate baseline R² score

#### Ridge & Lasso Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Hyperparameters: alpha values from 0.001 to 100
- Cross-validation for optimal alpha selection

#### Gradient Boosting (XGBoost)
- XGBoost regressor
- Hyperparameters: learning_rate, max_depth, n_estimators
- K-fold cross-validation (k=5)

#### Random Forest Regression
- Random Forest regressor
- Hyperparameters: n_estimators, max_depth, min_samples_split
- K-fold cross-validation (k=5)

### 3. Model Evaluation (2 hours)
- Calculate evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Mean Absolute Percentage Error (MAPE)
- Feature importance analysis (for tree-based models)
- Residual analysis and visualization
- Create comparison DataFrame
- Visualizations:
  - Actual vs Predicted scatter plots
  - Residual plots for all models
  - Feature importance plots
  - Error distribution histograms

## Output Files
- `models/linear_regression_model.pkl`
- `models/ridge_regression_model.pkl`
- `models/lasso_regression_model.pkl`
- `models/xgboost_regressor_model.pkl`
- `models/random_forest_regressor_model.pkl`
- `results/regression_metrics.csv`
- `visualizations/actual_vs_predicted.png`
- `visualizations/residual_plots.png`
- `visualizations/feature_importance_regression.png`
- `visualizations/error_distribution.png`

## Success Criteria
- All 5 regression models trained and evaluated
- Cross-validation scores logged for each model
- R² scores > 0.70 for at least 2 models
- RMSE documented for each model
- Feature importance identified for tree-based models
- Residual analysis completed and visualized

## Script: src/regression_models.py
```
run: python src/regression_models.py
```

## Deliverable
- Trained regression models saved
- Regression metrics report (MAE, RMSE, R²)
- Actual vs Predicted visualizations
- Residual analysis and plots
- Feature importance analysis

## Notes
- Use k-fold cross-validation for robust evaluation
- Save model artifacts for Day 6-7 integration
- Document hyperparameter selections for reproducibility
- Compare tree-based vs linear models performance
