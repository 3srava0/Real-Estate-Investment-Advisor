"""Feature Engineering Module for Real Estate Data"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature creation and transformation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        
    def create_numerical_features(self) -> None:
        """Create new numerical features"""
        logger.info("\n📊 Creating numerical features...")
    
    # Convert to numeric, handling any non-numeric values
        self.df['Nearby_Schools'] = pd.to_numeric(self.df['Nearby_Schools'], errors='coerce').fillna(0)
        self.df['Nearby_Hospitals'] = pd.to_numeric(self.df['Nearby_Hospitals'], errors='coerce').fillna(0)
        self.df['Public_Transport_Accessibility'] = pd.to_numeric(self.df['Public_Transport_Accessibility'], errors='coerce').fillna(0)
    
    # Price per square foot
        self.df['Price_per_SqFt'] = self.df['Price_in_Lakhs'] / (self.df['Size_in_SqFt'] + 1)
        logger.info("  ✓ Price_per_SqFt created")
    
    # Infrastructure score (normalized combination)
        self.df['Infrastructure_Score'] = (
        self.df['Nearby_Schools'] + 
        self.df['Nearby_Hospitals'] + 
        self.df['Public_Transport_Accessibility']
    ) / 3
        logger.info("  ✓ Infrastructure_Score created")
    
    # Property age
        current_year = 2024
        self.df['Property_Age'] = current_year - self.df['Year_Built']
        logger.info("  ✓ Property_Age created")
    
        # Amenities count
        self.df['Amenities_Count'] = self.df['Amenities'].fillna('').str.split(',').apply(len)
        logger.info("  ✓ Amenities_Count created")
        
        # Price-to-size ratio
        self.df['Price_to_BHK'] = self.df['Price_in_Lakhs'] / (self.df['BHK'] + 1)
        logger.info("  ✓ Price_to_BHK created")
    
        # Total floor advantage
        self.df['Floor_Advantage'] = self.df['Floor_No'] / (self.df['Total_Floors'] + 1)
        logger.info("  ✓ Floor_Advantage created")

    
    def create_categorical_features(self) -> None:
        """Create binary categorical features"""
        logger.info("\n📂 Creating categorical features...")
        
        # Luxury apartment indicator
        self.df['Is_Luxury'] = (
            (self.df['Furnished_Status'] == 'Fully') & 
            (self.df['BHK'] >= 3) &
            (self.df['Amenities_Count'] > 0)
        ).astype(int)
        logger.info("  ✓ Is_Luxury created")
        
        # Well-connected location
        self.df['Is_Well_Connected'] = (
            self.df['Public_Transport_Accessibility'] >= 3
        ).astype(int)
        logger.info("  ✓ Is_Well_Connected created")
        
        # Premium location (high school density)
        self.df['Is_Premium_Area'] = (
            self.df['Nearby_Schools'] >= 3
        ).astype(int)
        logger.info("  ✓ Is_Premium_Area created")
        
        # Secure property indicator
        self.df['Is_Secure'] = (
            self.df['Security'].notna() & 
            (self.df['Security'] != 'None')
        ).astype(int)
        logger.info("  ✓ Is_Secure created")
    
    def create_target_variables(self) -> None:
        """Create classification and regression target variables"""
        logger.info("\n🎯 Creating target variables...")
        
        # Regression Target: Future Price in 5 years
        # Using 8% annual appreciation rate (common in Indian real estate)
        annual_rate = 0.08
        self.df['Future_Price_5Y'] = self.df['Price_in_Lakhs'] * ((1 + annual_rate) ** 5)
        logger.info(f"  ✓ Future_Price_5Y created (8% annual growth)")
        
        # Classification Target: Good Investment
        # Multi-factor scoring approach
        median_price_per_sqft = self.df['Price_per_SqFt'].median()
        median_infrastructure = self.df['Infrastructure_Score'].median()
        
        score = (
            (self.df['Price_per_SqFt'] <= median_price_per_sqft).astype(int) * 2 +
            (self.df['Infrastructure_Score'] >= median_infrastructure).astype(int) * 2 +
            (self.df['Is_Well_Connected'] == 1).astype(int) * 1 +
            (self.df['Is_Premium_Area'] == 1).astype(int) * 1
        )
        
        self.df['Good_Investment'] = (score >= 3).astype(int)
        logger.info(f"  ✓ Good_Investment created (multi-factor scoring)")
        logger.info(f"    - Good Investment: {(self.df['Good_Investment']==1).sum()} properties")
        logger.info(f"    - Not Good: {(self.df['Good_Investment']==0).sum()} properties")
    
    def encode_categorical_features(self) -> None:
        """Encode categorical features using label encoding"""
        logger.info("\n🔤 Encoding categorical features...")
        
        categorical_cols = [
            'State', 'City', 'Property_Type', 'Furnished_Status', 
            'Owner_Type', 'Availability_Status', 'Facing', 'Security'
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  ✓ {col} encoded ({len(le.classes_)} classes)")
    
    def scale_numerical_features(self) -> None:
        """Scale numerical features"""
        logger.info("\n📈 Scaling numerical features...")
    
        numerical_cols = ['Price_in_Lakhs', 'Size_in_SqFt', 'BHK', 'Year_Built',
        'Floor_No', 'Total_Floors', 'Nearby_Schools', 'Nearby_Hospitals',
        'Public_Transport_Accessibility', 'Parking_Space',
        'Price_per_SqFt', 'Infrastructure_Score', 'Property_Age',
        'Price_to_BHK', 'Floor_Advantage']
    
    # Get columns that exist and convert all to numeric
        available_cols = []
        for col in numerical_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col].fillna(self.df[col].median(), inplace=True)
                available_cols.append(col)
    
    # Now scale
        self.df[available_cols] = self.scaler.fit_transform(self.df[available_cols])
        logger.info(f"  ✓ Scaled {len(available_cols)} numerical features")

    
    def get_feature_summary(self) -> Dict:
        """Generate feature engineering summary"""
        logger.info(f"\n" + "="*70)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total columns: {self.df.shape[1]}")
        logger.info(f"Total rows: {self.df.shape[0]}")
        logger.info(f"\nNew Features Created:")
        logger.info(f"  - Numerical: Price_per_SqFt, Infrastructure_Score, Property_Age, etc.")
        logger.info(f"  - Categorical: Is_Luxury, Is_Well_Connected, Is_Premium_Area, etc.")
        logger.info(f"\nTarget Variables:")
        logger.info(f"  - Regression: Future_Price_5Y (5-year price forecast)")
        logger.info(f"  - Classification: Good_Investment (binary prediction)")
        logger.info(f"\nEncoding: {len(self.label_encoders)} categorical features encoded")
        logger.info(f"Scaling: Numerical features standardized (mean=0, std=1)")
        logger.info("="*70)
        
        return {
            'total_columns': self.df.shape[1],
            'total_rows': self.df.shape[0],
            'new_features': len(self.df.columns),
            'encoded_features': len(self.label_encoders)
        }
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Return engineered dataset"""
        return self.df
    
    def execute_all(self) -> pd.DataFrame:
        """Execute all feature engineering steps"""
        logger.info("\n🚀 Starting Feature Engineering Pipeline...")
        self.create_numerical_features()
        self.create_categorical_features()
        self.create_target_variables()
        self.encode_categorical_features()
        self.scale_numerical_features()
        self.get_feature_summary()
        logger.info("\n✅ Feature Engineering Complete!")
        return self.df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/processed/india_housing_prices_clean.csv')
    engineer = FeatureEngineer(df)
    engineered_df = engineer.execute_all()
    
    # Save engineered data
    engineered_df.to_csv('data/processed/india_housing_prices_engineered.csv', index=False)
    logger.info("\n💾 Engineered data saved to: data/processed/india_housing_prices_engineered.csv")
