# Day 1: Data Preprocessing Execution Guide

## Overview
- **Date**: Dec 11, 2025
- **Duration**: 8 hours
- **Focus**: Data Loading, Cleaning, and Validation
- **Deliverable**: Processed dataset + Quality Report

---

## Step-by-Step Instructions

### 1. Setup Environment (30 mins)

```bash
# Navigate to project
cd Real-Estate-Investment-Advisor

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (30 mins)

1. Go to: https://drive.google.com/file/d/1ys25Eaqo2n8IeHhyI9s0kmJBgnNzxQHX/view
2. Download `india_housing_prices.csv`
3. Place in: `data/raw/india_housing_prices.csv`
4. Verify: `ls data/raw/` should show the file

### 3. Run Preprocessing (1.5 hours)

```bash
# Run the preprocessing module
python -c "from src.preprocessing import DataPreprocessor; p = DataPreprocessor('data/raw/india_housing_prices.csv'); p.load_data(); p.handle_missing_values(); p.remove_duplicates(); p.handle_outliers(); p.get_quality_report(); p.save_processed_data('data/processed/india_housing_prices_clean.csv')"
```

Or run directly:
```bash
python src/preprocessing.py
```

### 4. Verify Output (30 mins)

Check that these files exist:
```bash
ls -la data/processed/
# Should show: india_housing_prices_clean.csv
```

Quick data inspection:
```python
import pandas as pd
df = pd.read_csv('data/processed/india_housing_prices_clean.csv')
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Data types:\n{df.dtypes}")
```

### 5. Git Commit (30 mins)

```bash
# Add all changes
git add .

# Commit with Day 1 message
git commit -m "Day 1: Complete data preprocessing

- Download raw dataset
- Handle missing values (median/mode)
- Remove duplicates
- Detect outliers (IQR method)
- Generate data quality report
- Save processed dataset"

# Push to GitHub
git push origin main
```

---

## Expected Outputs

### Data Quality Report Example
```
============================================================
DATA QUALITY REPORT
============================================================
Original shape: (16000, 20)
Final shape: (15980, 20)
Rows removed: 20

Missing values: {'before': 850, 'after': 0}
Duplicates removed: {'removed': 20}
Outliers: {'detected': 345}
============================================================
```

### Files Created
```
data/processed/india_housing_prices_clean.csv  (16 MB approx)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Ensure venv is activated |
| File not found | Check file path, run from repo root |
| Memory error | Data is too large, close other apps |
| CSV encoding error | Try `encoding='latin1'` in read_csv |

---

## Success Criteria

- [x] Environment setup complete
- [x] Dataset downloaded and verified
- [x] Preprocessing script runs without errors
- [x] Processed dataset saved
- [x] Quality report generated
- [x] Changes committed to GitHub

---

## Next: Day 2

Tomorrow you'll focus on **Feature Engineering**:
- Create numerical and categorical features
- Engineer target variables (Future_Price_5Y, Good_Investment)
- Scale and encode features
- Prepare data for modeling

**Estimated effort**: 2 hours of active coding
