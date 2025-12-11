# Day 2-3: Feature Engineering & EDA Execution Guide

## Overview
- **Dates**: Dec 11-12, 2025
- **Duration**: 16 hours total (8 hrs each day)
- **Focus**: Feature Creation, Target Variables, EDA & Visualization
- **Deliverable**: Engineered dataset + EDA insights

---

## Day 2: Feature Engineering (8 hours)

### Step 1: Pull Latest Changes from GitHub (15 mins)

```bash
cd Real-Estate-Investment-Advisor
git pull origin main
```

### Step 2: Run Feature Engineering (2 hours)

```bash
# Activate venv if not already active
venv\Scripts\activate

# Run feature engineering
python src/feature_engineering.py
```

**Expected Output:**
```
🚀 Starting Feature Engineering Pipeline...
📊 Creating numerical features...
  ✓ Price_per_SqFt created
  ✓ Infrastructure_Score created
  ✓ Property_Age created
  ✓ Amenities_Count created
  ✓ Price_to_BHK created
  ✓ Floor_Advantage created

📂 Creating categorical features...
  ✓ Is_Luxury created
  ✓ Is_Well_Connected created
  ✓ Is_Premium_Area created
  ✓ Is_Secure created

🎯 Creating target variables...
  ✓ Future_Price_5Y created (8% annual growth)
  ✓ Good_Investment created (multi-factor scoring)
    - Good Investment: ~125000 properties
    - Not Good: ~125000 properties

🔤 Encoding categorical features...
  ✓ State encoded (29 classes)
  ✓ City encoded (250+ classes)
  ...

📈 Scaling numerical features...
  ✓ Scaled 15 numerical features

======================================================================
FEATURE ENGINEERING SUMMARY
======================================================================
Total columns: 35+
Total rows: 250000
...
======================================================================

✅ Feature Engineering Complete!
💾 Engineered data saved to: data/processed/india_housing_prices_engineered.csv
```

### Step 3: Verify Engineered Data (30 mins)

```bash
# Quick check
python -c "import pandas as pd; df = pd.read_csv('data/processed/india_housing_prices_engineered.csv'); print(f'Shape: {df.shape}'); print(f'\nTarget Variables:'); print(f'Good_Investment Distribution:\n{df.Good_Investment.value_counts()}'); print(f'\nFuture_Price_5Y Stats:\n{df.Future_Price_5Y.describe()}')"
```

### Step 4: Git Commit (1 hour)

```bash
git add .
git commit -m "Day 2: Complete feature engineering and target variable creation

- Create numerical features (Price_per_SqFt, Infrastructure_Score, etc.)
- Create categorical features (Is_Luxury, Is_Well_Connected, etc.)
- Engineer target variables:
  * Future_Price_5Y: 5-year price forecast using 8% annual growth
  * Good_Investment: Multi-factor scoring (price, infrastructure, location)
- Label encode categorical features (8 features, 29-250+ classes each)
- Standardize numerical features (15 features)
- Total engineered dataset: 35+ columns, 250k rows"

git push origin main
```

---

## Day 3: Exploratory Data Analysis (8 hours)

### Step 1: Load Engineered Data (30 mins)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/processed/india_housing_prices_engineered.csv')
print(df.shape)
print(df.head())
print(df.info())
```

### Step 2: Univariate Analysis (2 hours)

```python
# Price distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Price_in_Lakhs'], bins=50, edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['Future_Price_5Y'], bins=50, edgecolor='black', color='orange')
plt.title('Future Price 5Y Distribution')
plt.xlabel('Future Price (Lakhs)')
plt.savefig('output/01_price_distribution.png')
plt.show()

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Good_Investment'].value_counts().plot(kind='bar', ax=axes[0], color=['red', 'green'])
axes[0].set_title('Good Investment Distribution')
axes[0].set_ylabel('Count')

df['BHK'].value_counts().sort_index().plot(kind='bar', ax=axes[1])
axes[1].set_title('BHK Distribution')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('output/02_target_and_bhk.png')
plt.show()
```

### Step 3: Bivariate Analysis (2 hours)

```python
# Correlation heatmap
plt.figure(figsize=(14, 10))
numerical_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('output/03_correlation_heatmap.png')
plt.show()

# Price vs Size scatter
plt.figure(figsize=(10, 6))
plt.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'], alpha=0.5, s=20)
plt.xlabel('Size (SqFt)')
plt.ylabel('Price (Lakhs)')
plt.title('Price vs Size Relationship')
plt.savefig('output/04_price_vs_size.png')
plt.show()

# Good Investment by Infrastructure Score
plt.figure(figsize=(10, 6))
df.groupby('Good_Investment')['Infrastructure_Score'].hist(alpha=0.6, bins=30)
plt.xlabel('Infrastructure Score')
plt.ylabel('Frequency')
plt.title('Infrastructure Score by Investment Quality')
plt.legend(['Not Good', 'Good Investment'])
plt.savefig('output/05_infrastructure_by_investment.png')
plt.show()
```

### Step 4: Multivariate Analysis (2 hours)

```python
# Price by Property Type and Furnished Status
fig, ax = plt.subplots(figsize=(12, 6))
df.groupby(['Property_Type', 'Furnished_Status'])['Price_in_Lakhs'].mean().unstack().plot(kind='bar', ax=ax)
ax.set_title('Average Price by Property Type and Furnishing')
ax.set_ylabel('Average Price (Lakhs)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/06_price_by_type_furnishing.png')
plt.show()

# Good Investment percentage by features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df.groupby('Is_Luxury')['Good_Investment'].mean().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Good Investment % by Luxury Status')

df.groupby('Is_Well_Connected')['Good_Investment'].mean().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Good Investment % by Connectivity')

df.groupby('Is_Premium_Area')['Good_Investment'].mean().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Good Investment % by Premium Area')

df.groupby('Is_Secure')['Good_Investment'].mean().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Good Investment % by Security')

plt.tight_layout()
plt.savefig('output/07_investment_by_features.png')
plt.show()
```

### Step 5: Summary Statistics (1 hour)

```python
# Generate comprehensive summary
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("="*70)
print(f"\nDataset Shape: {df.shape}")
print(f"\nNumerical Summary:")
print(df.describe())
print(f"\nTarget Variable Distribution:")
print(f"Good Investment: {(df['Good_Investment']==1).sum()} ({100*(df['Good_Investment']==1).sum()/len(df):.2f}%)")
print(f"Not Good: {(df['Good_Investment']==0).sum()} ({100*(df['Good_Investment']==0).sum()/len(df):.2f}%)")
print(f"\nMissing Values: {df.isnull().sum().sum()}")
print("="*70)
```

### Step 6: Git Commit (1 hour)

```bash
mkdir -p output
# Save all visualizations

git add .
git commit -m "Day 3: Complete EDA with visualizations

- Univariate analysis: Price, Size, Target variables
- Bivariate analysis: Correlation, Price vs Size
- Multivariate analysis: Price by property type & furnishing
- Feature importance: Investment quality drivers
- Generated 7+ EDA visualizations
- Summary statistics and insights documented"

git push origin main
```

---

## Success Checklist

### Day 2
- [x] Feature engineering script created
- [x] Feature engineering executed successfully
- [x] Engineered dataset saved
- [x] Changes committed to GitHub

### Day 3
- [x] EDA analysis completed
- [x] Visualizations generated (7+ plots)
- [x] Summary statistics documented
- [x] Changes committed to GitHub

---

## Next: Days 4-5

Tomorrow you'll build the ML models:
- **Day 4**: Classification Model (Good Investment prediction)
- **Day 5**: Regression Model (Future Price prediction)

**Focus**: Model training, evaluation, hyperparameter tuning, and MLflow tracking
