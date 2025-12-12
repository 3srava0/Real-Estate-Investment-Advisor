# DAY 4: Classification Models Training & Evaluation

## Objective
Build, train, and evaluate multiple classification models to predict the binary "Good_Investment" target variable. Implement hyperparameter tuning and cross-validation.

## Duration: 8 hours (1 day)

## Timeline
- **Hour 1-2**: Data preparation and feature scaling (Stratified train-test split)
- **Hour 3-4**: Implement Logistic Regression and Random Forest classifiers
- **Hour 5-6**: Implement Gradient Boosting and SVM classifiers
- **Hour 7-8**: Model comparison, evaluation metrics, and hyperparameter tuning

## Tasks

### 1. Data Preparation (1 hour)
- Load cleaned data from Day 1 output
- Apply feature scaling (StandardScaler or MinMaxScaler)
- Create stratified train-test split (80-20)
- Handle any imbalanced classes (SMOTE or class weights)

### 2. Model Implementation (5 hours)

#### Logistic Regression
- Basic logistic regression model
- Hyperparameters: C, max_iter, solver
- Training with stratified k-fold CV (k=5)

#### Random Forest
- Random Forest classifier
- Hyperparameters: n_estimators, max_depth, min_samples_split
- Training with stratified k-fold CV (k=5)

#### Gradient Boosting
- XGBoost or LightGBM classifier
- Hyperparameters: learning_rate, max_depth, n_estimators
- Training with stratified k-fold CV (k=5)

#### SVM (Support Vector Machine)
- SVM with RBF kernel
- Hyperparameters: C, gamma, kernel
- Training with stratified k-fold CV (k=5)

### 3. Model Evaluation (2 hours)
- Calculate evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix
- Feature importance analysis (for tree-based models)
- Create comparison DataFrame
- Visualize:
  - Confusion matrices for all models
  - ROC curves for all models
  - Feature importance plots

## Output Files
- `models/logistic_regression_model.pkl`
- `models/random_forest_model.pkl`
- `models/xgboost_model.pkl`
- `models/svm_model.pkl`
- `results/classification_metrics.csv`
- `visualizations/confusion_matrices.png`
- `visualizations/roc_curves.png`
- `visualizations/feature_importance.png`

## Success Criteria
- All 4 classification models trained and evaluated
- Cross-validation scores logged for each model
- ROC-AUC scores > 0.85 for at least 2 models
- Feature importance identified for tree-based models
- Comprehensive comparison report generated

## Script: src/classification_models.py
```
run: python src/classification_models.py
```

## Deliverable
- Trained classification models saved
- Evaluation metrics report
- Visualizations comparing all 4 models
- Feature importance analysis

## Notes
- Use stratified k-fold to handle class imbalance
- Log all hyperparameters used for future reference
- Save model artifacts for Day 5-7 integration
