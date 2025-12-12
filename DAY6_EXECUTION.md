# DAY 6: Model Evaluation, Comparison & Selection

## Objective
Comprehensively evaluate all trained models, compare performance metrics, analyze feature importance, and select the best models for production. Create detailed comparison reports and visualizations.

## Duration: 8 hours (1 day)

## Timeline
- **Hour 1-2**: Load all trained models and review metrics from Days 4-5
- **Hour 2-3**: Comprehensive metrics comparison (Classification vs Regression)
- **Hour 3-4**: Feature importance analysis and ranking
- **Hour 4-5**: Cross-validation score analysis and model robustness testing
- **Hour 5-6**: Create comparison visualizations and heatmaps
- **Hour 6-7**: Hyperparameter sensitivity analysis
- **Hour 7-8**: Final model selection and documentation

## Tasks

### 1. Load & Organize Results (1 hour)
- Load all saved models from Day 4 and 5 outputs
- Load metrics CSV files (classification_metrics.csv, regression_metrics.csv)
- Create unified comparison DataFrames
- Document model hyperparameters used

### 2. Classification Model Comparison (2 hours)
- Compare metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Identify best performing model based on multiple criteria
- Analyze trade-offs (Precision vs Recall)
- ROC-AUC threshold analysis
- Create comparison table and visualizations

### 3. Regression Model Comparison (2 hours)
- Compare metrics: MAE, RMSE, R²Score, MAPE
- Identify best performing model
- Error distribution analysis
- Prediction accuracy assessment
- Create comparison charts and error plots

### 4. Feature Importance Analysis (1.5 hours)
- Extract and compare feature importances from tree-based models
- Identify top 15 most important features
- Compare feature rankings across models
- Create combined feature importance visualization
- Document feature impact on predictions

### 5. Model Robustness & Validation (1 hour)
- Review cross-validation scores
- Calculate confidence intervals for metrics
- Analyze model stability across folds
- Identify overfitting/underfitting tendencies

### 6. Final Selection & Documentation (0.5 hour)
- **Classification Best Model**: XGBoost or Random Forest (based on metrics)
- **Regression Best Model**: XGBoost or Random Forest (based on metrics)
- Document selection rationale
- Prepare models for deployment (Day 7)

## Output Files
- `results/model_comparison_classification.csv`
- `results/model_comparison_regression.csv`
- `results/feature_importance_ranking.csv`
- `visualizations/model_metrics_comparison.png`
- `visualizations/roc_auc_comparison.png`
- `visualizations/feature_importance_combined.png`
- `visualizations/model_performance_heatmap.png`
- `visualizations/error_distribution_comparison.png`
- `reports/MODEL_SELECTION_REPORT.md`

## Success Criteria
- All models evaluated and compared
- Comprehensive metrics summary created
- Feature importance ranked and visualized
- Best classification model identified
- Best regression model identified
- Selection rationale documented
- Ready for production deployment

## Script: src/day6_model_evaluation.py
```
run: python src/day6_model_evaluation.py
```

## Deliverable
- Model comparison report
- Feature importance analysis
- Comprehensive visualizations
- Best models identified and documented
- Model selection justification

## Notes
- Use ROC-AUC as primary metric for classification
- Use R²Score as primary metric for regression
- Consider inference time and model complexity
- Document all comparison results for stakeholders
