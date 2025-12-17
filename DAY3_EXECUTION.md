# DAY 3: Exploratory Data Analysis (EDA)

## Objective
Perform comprehensive exploratory data analysis on the preprocessed and feature-engineered real estate dataset. Understand data distributions, relationships between features, and generate insights for model building.

## Duration: 8 hours (1 day)

## Timeline
- **Hour 1**: Data loading and initial inspection
- **Hour 2-3**: Univariate analysis (individual feature distributions)
- **Hour 4-5**: Bivariate analysis (feature relationships and correlations)
- **Hour 6-7**: Multivariate analysis (complex patterns and interactions)
- **Hour 7-8**: Summary statistics and insights documentation

## Tasks

### 1. Data Loading & Inspection (0.5 hour)
- Load cleaned and engineered data from Day 2 output
- Display dataset shape and basic info
- Check for missing values
- Review data types and memory usage

### 2. Univariate Analysis (1.5 hours)
Analyze individual features to understand distributions:

**Questions 1-5: Price & Size Analysis**
1. What is the distribution of property prices?
2. What is the distribution of property sizes?
3. How does price per sq ft vary by property type?
4. Is there a relationship between property size and price?
5. Are there any outliers in price per sq ft or property size?

**Visualizations:**
- `01_price_distribution.png` - Original and Future Price distributions
- `02_target_and_bhk.png` - Good Investment and BHK count distributions

### 3. Bivariate Analysis (2 hours)
Analyze relationships between features:

**Questions 6-10: Location-based Analysis**
6. What is the average price per sq ft by state?
7. What is the average property price by city?
8. What is the median age of properties by locality?
9. How is BHK distributed across cities?
10. What are the price trends for the top 5 most expensive localities?

**Questions 11-15: Feature Relationship & Correlation**
11. How are numeric features correlated with each other?
12. How do nearby schools relate to price per sq ft?
13. How do nearby hospitals relate to price per sq ft?
14. How does price vary by furnished status?
15. How does price per sq ft vary by property facing direction?

**Visualizations:**
- `03_correlation_heatmap.png` - Correlation matrix for top 15 features
- `04_price_vs_size.png` - Price vs Property Size scatter plot

### 4. Multivariate Analysis (2 hours)
Analyze complex patterns and interactions:

**Questions 16-20: Investment, Amenities & Ownership Analysis**
16. How many properties belong to each owner type?
17. How many properties are available under each availability status?
18. Does parking space affect property price?
19. How do amenities affect price per sq ft?
20. How does public transport accessibility relate to price per sq ft or investment potential?

**Visualizations:**
- `05_investment_by_features.png` - Good Investment % by Luxury, Connectivity, Premium Area, Security
- `06_infrastructure_impact.png` - Good Investment Rate by Infrastructure Score Quartile
- `07_price_sqft_by_investment.png` - Price per SqFt Distribution by Investment Quality

### 5. Summary Statistics (1 hour)
- Generate descriptive statistics for all numeric features
- Analyze target variable distribution (Good_Investment)
- Document key findings and insights
- Identify data quality issues and patterns

## Output Files
- `output/01_price_distribution.png` - Price distribution plots
- `output/02_target_and_bhk.png` - Target and BHK distributions
- `output/03_correlation_heatmap.png` - Correlation matrix heatmap
- `output/04_price_vs_size.png` - Price vs Size scatter plot
- `output/05_investment_by_features.png` - Investment analysis by features
- `output/06_infrastructure_impact.png` - Infrastructure score impact
- `output/07_price_sqft_by_investment.png` - Price per sqft by investment quality

## Success Criteria
- All 7 visualizations generated successfully
- 20 analysis questions answered with visualizations
- Summary statistics calculated and documented
- Data quality issues identified and logged
- Insights documented for model building
- No errors in EDA execution

## Script: day3_eda.py
```bash
# Run the EDA script
python day3_eda.py

# Output location
# All visualizations saved to ./output/ directory
```

## Key Insights to Identify
1. **Price Insights:**
   - Mean and median property price
   - Price distribution shape (skewness, kurtosis)
   - Outliers and extreme values

2. **Target Variable (Good_Investment):**
   - Class distribution (% Good vs Not Good)
   - Correlation with other features
   - Features most strongly associated with good investments

3. **Geographic Patterns:**
   - Cities/localities with highest average prices
   - Geographic variation in investment quality
   - Location impact on property value

4. **Feature Relationships:**
   - Strongest correlations with price and investment quality
   - Feature interactions and dependencies
   - Multicollinearity patterns

## Deliverable
- 7 professional visualizations
- EDA summary report with key findings
- Data quality assessment
- Feature insights for model selection
- Answers to all 20 analysis questions

## Notes
- Use consistent styling and color schemes across visualizations
- Save all plots at 100 DPI for clarity
- Include clear titles, labels, and legends
- Log all operations and findings
- Identify patterns relevant to investment prediction
