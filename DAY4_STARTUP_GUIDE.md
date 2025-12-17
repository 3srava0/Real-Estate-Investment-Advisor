# DAY 4 STARTUP GUIDE: Classification Models Training

**Status:** ✅ Ready to Start  
**Date:** December 17, 2025  
**Previous Progress:** Days 1-3 Complete (43% overall)

---

## QUICK SUMMARY

Day 4 focuses on **building and training 4 classification models** to predict the `Good_Investment` binary target variable. Most of the scaffolding is already in place - you primarily need to **run the Python script** and verify the outputs.

---

## WHAT'S ALREADY DONE (Verified)

### ✅ Documentation Files Created
1. **`DAY4_EXECUTION.md`** (5 days ago) 
   - Comprehensive 8-hour execution guide
   - Contains all tasks, timeline, and success criteria
   - References to 4 algorithms and evaluation metrics

### ✅ Python Script Created  
2. **`src/day4_classification_models.py`** (267 lines, 10.4 KB)
   - Complete ClassificationModels class
   - All 4 models implemented:
     - Logistic Regression (with StandardScaler)
     - Random Forest Classifier
     - XGBoost Classifier
     - SVM (Support Vector Machine)
   - Evaluation methods with all metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Visualization methods for:
     - Confusion matrices
     - ROC curves
     - Feature importance plots
   - Model saving and results export

---

## WHAT YOU MUST DO (Tasks for You)

### 🎯 PRIMARY TASK: Execute the Classification Models Script

**Command to Run:**
```bash
python src/day4_classification_models.py
```

**Expected Output:**
When you run this script, it will:

1. **Load Data**
   - Load `output/data_engineered.csv` from Day 2
   - Display dataset shape (should be ~250k rows)

2. **Prepare Data**
   - Split data into train-test (80-20) with stratification
   - Scale features using StandardScaler
   - Display class distribution

3. **Train 4 Models** (Should see these messages)
   ```
   Training Logistic Regression...
   CV ROC-AUC: [score] (+/- [std])
   
   Training Random Forest...
   CV ROC-AUC: [score] (+/- [std])
   
   Training XGBoost...
   CV ROC-AUC: [score] (+/- [std])
   
   Training SVM...
   CV ROC-AUC: [score] (+/- [std])
   ```

4. **Evaluate Models**
   - Display metrics for each model:
     - Accuracy, Precision, Recall, F1-Score, ROC-AUC

5. **Generate Outputs**
   - Save trained models to `models/` folder:
     - `logistic_regression_model.pkl`
     - `random_forest_model.pkl`
     - `xgboost_model.pkl`
     - `svm_model.pkl`
   - Save results to `results/classification_metrics.csv`
   - Create visualizations in `visualizations/`:
     - `confusion_matrices.png` (2x2 grid)
     - `roc_curves.png` (all 4 models overlaid)
     - `feature_importance.png` (Random Forest + XGBoost)

---

## PREREQUISITES TO CHECK

Before running the script, verify these are in place:

### ✅ Required Data File
- [ ] `output/data_engineered.csv` exists (from Day 2-3)
  - Size should be ~100-150 MB
  - Must contain:
    - `Good_Investment` column (target variable)
    - `Future_Price_5Y` column (will be dropped)
    - All feature columns

### ✅ Required Python Packages
Ensure these are installed:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

Or use:
```bash
pip install -r requirements.txt
```

### ✅ Required Directories (Auto-created by script)
- `models/` - for trained model files
- `results/` - for metrics CSV
- `visualizations/` - for PNG plots
- `output/` - already exists from Day 2-3

---

## STEP-BY-STEP EXECUTION

### Step 1: Navigate to Project Directory
```bash
cd Real-Estate-Investment-Advisor
```

### Step 2: Verify Data File Exists
```bash
ls -lh output/data_engineered.csv
```
Expected output: File exists with size ~100-150 MB

### Step 3: Run the Classification Models Script
```bash
python src/day4_classification_models.py
```

### Step 4: Monitor Execution
Watch for these phases:
- **Phase 1** (30 seconds): Data loading and preparation
- **Phase 2** (2-5 minutes): Model training with cross-validation
- **Phase 3** (1 minute): Model evaluation
- **Phase 4** (2 minutes): Visualization generation

### Step 5: Verify Outputs
```bash
# Check model files
ls -lh models/

# Check results
cat results/classification_metrics.csv

# Check visualizations
ls -lh visualizations/
```

---

## EXPECTED RESULTS & SUCCESS CRITERIA

### ✅ All 4 Models Should Train
- Logistic Regression: Should complete in <1 min
- Random Forest: Should complete in 2-3 min
- XGBoost: Should complete in 2-3 min
- SVM: Should complete in 3-5 min

### ✅ Metrics to Expect
For a well-trained model:
- **ROC-AUC** should be > 0.85 for at least 2-3 models
- **Accuracy** typically 75-90%
- **F1-Score** typically 0.70-0.85

### ✅ Output Files Generated
You should see these 7 files created:
```
models/
  ├── logistic_regression_model.pkl
  ├── random_forest_model.pkl
  ├── xgboost_model.pkl
  └── svm_model.pkl

results/
  └── classification_metrics.csv

visualizations/
  ├── confusion_matrices.png
  ├── roc_curves.png
  └── feature_importance.png
```

---

## POTENTIAL ISSUES & SOLUTIONS

### Issue 1: "FileNotFoundError: data_engineered.csv not found"
**Solution:** Make sure Day 2-3 output files exist
```bash
ls output/*.csv
```
If missing, run Day 2-3 scripts first

### Issue 2: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:** Install XGBoost
```bash
pip install xgboost
```

### Issue 3: Script hangs on "Training SVM..."
**Solution:** SVM training is slower. This is normal - wait 5-10 minutes

### Issue 4: Memory error during Random Forest training
**Solution:** Reduce n_estimators in the script (line ~95)
Change `n_estimators=100` to `n_estimators=50`

---

## WHAT TO CHECK AFTER EXECUTION

### Check 1: Metrics CSV
```bash
cat results/classification_metrics.csv
```
Should show table with:
- Model names (rows)
- Accuracy, Precision, Recall, F1-Score, ROC-AUC (columns)

### Check 2: Model Files
```bash
ls -lh models/
```
All 4 .pkl files should be 20-50 MB each

### Check 3: Visualizations
```bash
open visualizations/confusion_matrices.png  # Mac
xdg-open visualizations/confusion_matrices.png  # Linux
```
Visuals should show:
- 4 confusion matrices (2x2 grid)
- 4 ROC curves overlaid on same plot
- Top 10 features for Random Forest + XGBoost

---

## NEXT STEPS AFTER DAY 4

### After Script Completes Successfully:
1. Create `DAY4_CHECKLIST.md` (similar to Day 3)
2. Commit all outputs to GitHub:
   ```bash
   git add -A
   git commit -m "Day 4: Classification models training complete"
   git push origin main
   ```

### Then Proceed to Day 5:
- Build 5 regression models
- Follow similar pattern to Day 4
- Use `src/day5_regression_models.py`

---

## QUICK REFERENCE: Script Methods

The ClassificationModels class contains these methods:

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `load_data()` | Load CSV | data_path | df |
| `prepare_data()` | Train-test split + scaling | df | X_train, X_test, y_train, y_test (scaled) |
| `train_logistic_regression()` | LR model | X_train_scaled, y_train | model saved |
| `train_random_forest()` | RF model | X_train, y_train | model + feature importance |
| `train_xgboost()` | XGB model | X_train, y_train | model + feature importance |
| `train_svm()` | SVM model | X_train_scaled, y_train | model saved |
| `evaluate_models()` | Calculate metrics | X_test, y_test | results dict |
| `save_models()` | Pickle models | models dict | .pkl files |
| `save_results()` | Export metrics | results dict | CSV file |
| `plot_confusion_matrices()` | Confusion matrices | results | PNG file |
| `plot_roc_curves()` | ROC curves | results | PNG file |
| `plot_feature_importance()` | Feature plots | results | PNG file |
| `run()` | Execute all methods | - | Full pipeline |

---

## QUICK START COMMAND

**Just want to get started? Run this:**
```bash
cd Real-Estate-Investment-Advisor && python src/day4_classification_models.py
```

**Expected total runtime:** 15-20 minutes

---

## FILE STRUCTURE AFTER DAY 4 COMPLETION

```
Real-Estate-Investment-Advisor/
├── DAY4_EXECUTION.md
├── DAY4_STARTUP_GUIDE.md
├── models/                        ← 4 new model files
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── svm_model.pkl
├── results/
│   ├── classification_metrics.csv  ← new
│   └── [regression files from future days]
├── visualizations/
│   ├── confusion_matrices.png       ← new
│   ├── roc_curves.png              ← new
│   ├── feature_importance.png      ← new
│   └── [other EDA visualizations from Days 3-7]
└── src/
    └── day4_classification_models.py  ← ready to run
```

---

**You're all set! Execute the script and proceed to create the Day 4 checklist when complete.**
