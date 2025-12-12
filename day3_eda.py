"""Day 3: Exploratory Data Analysis (EDA) for Real Estate Data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EDA:
    """Performs exploratory data analysis"""
    
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"\n📊 Loaded data: {self.df.shape}")
    
    def univariate_analysis(self):
        """Analyze individual features"""
        logger.info("\n📈 Univariate Analysis...")
        
        # Price distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self.df['Price_in_Lakhs'].hist(bins=50, ax=axes[0], edgecolor='black')
        axes[0].set_title('Price Distribution (Original)')
        axes[0].set_xlabel('Price (Lakhs)')
        axes[0].set_ylabel('Frequency')
        
        self.df['Future_Price_5Y'].hist(bins=50, ax=axes[1], color='orange', edgecolor='black')
        axes[1].set_title('Future Price 5Y Distribution')
        axes[1].set_xlabel('Future Price (Lakhs)')
        axes[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_price_distribution.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 01_price_distribution.png")
        plt.close()
        
        # Target variable distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self.df['Good_Investment'].value_counts().plot(kind='bar', ax=axes[0], color=['red', 'green'])
        axes[0].set_title('Good Investment Distribution')
        axes[0].set_xlabel('Investment Quality')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['Not Good (0)', 'Good (1)'], rotation=0)
        
        self.df['BHK'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='steelblue')
        axes[1].set_title('BHK Distribution')
        axes[1].set_xlabel('BHK')
        axes[1].set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_target_and_bhk.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 02_target_and_bhk.png")
        plt.close()
    
    def bivariate_analysis(self):
        """Analyze relationships between features"""
        logger.info("\n🔗 Bivariate Analysis...")
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:15]
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix (Top Features)')
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_correlation_heatmap.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 03_correlation_heatmap.png")
        plt.close()
        
        # Price vs Size scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Size_in_SqFt'], self.df['Price_in_Lakhs'], alpha=0.5, s=10, c='steelblue')
        plt.xlabel('Size (SqFt)')
        plt.ylabel('Price (Lakhs)')
        plt.title('Price vs Property Size Relationship')
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_price_vs_size.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 04_price_vs_size.png")
        plt.close()
    
    def multivariate_analysis(self):
        """Analyze complex patterns"""
        logger.info("\n📊 Multivariate Analysis...")
        
        # Good Investment by features
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        self.df.groupby('Is_Luxury')['Good_Investment'].mean().plot(kind='bar', ax=axes[0, 0], color=['coral', 'green'])
        axes[0, 0].set_title('Good Investment % by Luxury Status')
        axes[0, 0].set_xticklabels(['Not Luxury', 'Luxury'], rotation=0)
        axes[0, 0].set_ylabel('Proportion Good')
        
        self.df.groupby('Is_Well_Connected')['Good_Investment'].mean().plot(kind='bar', ax=axes[0, 1], color=['coral', 'green'])
        axes[0, 1].set_title('Good Investment % by Connectivity')
        axes[0, 1].set_xticklabels(['Not Connected', 'Connected'], rotation=0)
        axes[0, 1].set_ylabel('Proportion Good')
        
        self.df.groupby('Is_Premium_Area')['Good_Investment'].mean().plot(kind='bar', ax=axes[1, 0], color=['coral', 'green'])
        axes[1, 0].set_title('Good Investment % by Premium Area')
        axes[1, 0].set_xticklabels(['Not Premium', 'Premium'], rotation=0)
        axes[1, 0].set_ylabel('Proportion Good')
        
        self.df.groupby('Is_Secure')['Good_Investment'].mean().plot(kind='bar', ax=axes[1, 1], color=['coral', 'green'])
        axes[1, 1].set_title('Good Investment % by Security')
        axes[1, 1].set_xticklabels(['Not Secure', 'Secure'], rotation=0)
        axes[1, 1].set_ylabel('Proportion Good')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_investment_by_features.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 05_investment_by_features.png")
        plt.close()
    
    def feature_insights(self):
        """Generate feature insights"""
        logger.info("\n💡 Feature Insights...")
        
        # Infrastructure score impact
        plt.figure(figsize=(12, 5))
        self.df.groupby(pd.cut(self.df['Infrastructure_Score'], bins=5))['Good_Investment'].mean().plot(kind='bar', color='steelblue')
        plt.title('Good Investment Rate by Infrastructure Score Quartile')
        plt.xlabel('Infrastructure Score Range')
        plt.ylabel('Good Investment %')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_infrastructure_impact.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 06_infrastructure_impact.png")
        plt.close()
        
        # Price per SqFt distribution by investment quality
        plt.figure(figsize=(10, 6))
        for quality in [0, 1]:
            label = 'Good Investment' if quality == 1 else 'Not Good'
            self.df[self.df['Good_Investment'] == quality]['Price_per_SqFt'].hist(alpha=0.6, bins=30, label=label)
        plt.xlabel('Price per SqFt')
        plt.ylabel('Frequency')
        plt.title('Price per SqFt Distribution by Investment Quality')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_price_sqft_by_investment.png', dpi=100, bbox_inches='tight')
        logger.info("  ✓ Saved: 07_price_sqft_by_investment.png")
        plt.close()
    
    def summary_statistics(self):
        """Generate summary statistics"""
        logger.info(f"\n{'='*70}")
        logger.info("EXPLORATORY DATA ANALYSIS SUMMARY")
        logger.info(f"{'='*70}")
        
        logger.info(f"\n📊 Dataset Shape: {self.df.shape}")
        logger.info(f"\n📈 Numerical Features Summary:")
        logger.info(f"{self.df.describe().round(2)}")
        
        logger.info(f"\n🎯 Target Variable Distribution:")
        good_count = (self.df['Good_Investment'] == 1).sum()
        bad_count = (self.df['Good_Investment'] == 0).sum()
        good_pct = 100 * good_count / len(self.df)
        bad_pct = 100 * bad_count / len(self.df)
        logger.info(f"  Good Investment: {good_count:,} ({good_pct:.2f}%)")
        logger.info(f"  Not Good: {bad_count:,} ({bad_pct:.2f}%)")
        
        logger.info(f"\n❌ Missing Values: {self.df.isnull().sum().sum()}")
        logger.info(f"\n📚 Categorical Features Count:")
        categorical = self.df.select_dtypes(include='object').columns
        for col in categorical[:5]:
            logger.info(f"  {col}: {self.df[col].nunique()} unique values")
        
        logger.info(f"{'='*70}")
        logger.info(f"✅ EDA Complete! Generated 7 visualizations in ./output/")
        logger.info(f"{'='*70}")
    
    def execute_all(self):
        """Execute all analysis"""
        logger.info("\n🚀 Starting Day 3 EDA...")
        self.univariate_analysis()
        self.bivariate_analysis()
        self.multivariate_analysis()
        self.feature_insights()
        self.summary_statistics()


if __name__ == "__main__":
    eda = EDA('data/processed/india_housing_prices_engineered.csv')
    eda.execute_all()
    logger.info("\n💾 All visualizations saved to ./output/ directory")
