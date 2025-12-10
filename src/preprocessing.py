"""Data Preprocessing Module for Real Estate Data"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and validation"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.original_shape = None
        self.quality_report = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data from specified path"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.original_shape = self.df.shape
            logger.info(f"✅ Data loaded: {self.original_shape[0]} rows, {self.original_shape[1]} columns")
            return self.df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, strategy: str = 'median') -> None:
        """Handle missing values in dataset"""
        missing_before = self.df.isnull().sum().sum()
        logger.info(f"\nHandling missing values...")
        logger.info(f"Total missing values before: {missing_before}")
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    logger.info(f"  - {col}: Filled with median")
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    logger.info(f"  - {col}: Filled with mode")
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Total missing values after: {missing_after}")
        self.quality_report['missing_values'] = {'before': missing_before, 'after': missing_after}
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows"""
        duplicates_before = self.df.duplicated().sum()
        logger.info(f"\nRemoving duplicates...")
        logger.info(f"Duplicate rows before: {duplicates_before}")
        
        self.df.drop_duplicates(inplace=True)
        
        duplicates_after = self.df.duplicated().sum()
        logger.info(f"Duplicate rows after: {duplicates_after}")
        logger.info(f"Rows removed: {duplicates_before - duplicates_after}")
        self.quality_report['duplicates'] = {'removed': duplicates_before - duplicates_after}
    
    def handle_outliers(self, columns: list = None) -> None:
        """Remove outliers using IQR method"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        logger.info(f"\nHandling outliers (IQR method)...")
        outliers_removed = 0
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_in_col = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outliers_in_col > 0:
                logger.info(f"  - {col}: Found {outliers_in_col} outliers")
                outliers_removed += outliers_in_col
        
        self.quality_report['outliers'] = {'detected': outliers_removed}
    
    def get_quality_report(self) -> Dict:
        """Generate data quality report"""
        logger.info(f"\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)
        logger.info(f"Original shape: {self.original_shape}")
        logger.info(f"Final shape: {self.df.shape}")
        logger.info(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        logger.info(f"\nMissing values: {self.quality_report.get('missing_values', {})}")
        logger.info(f"Duplicates removed: {self.quality_report.get('duplicates', {})}")
        logger.info(f"Outliers: {self.quality_report.get('outliers', {})}")
        logger.info("="*60)
        
        return self.quality_report
    
    def save_processed_data(self, output_path: str) -> None:
        """Save cleaned data to CSV"""
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"\n✅ Processed data saved to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving data: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('data/raw/india_housing_prices.csv')
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.remove_duplicates()
    preprocessor.handle_outliers()
    preprocessor.get_quality_report()
    preprocessor.save_processed_data('data/processed/india_housing_prices_clean.csv')
