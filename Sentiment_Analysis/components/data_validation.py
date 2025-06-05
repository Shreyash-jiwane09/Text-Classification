import os
import sys
import pandas as pd

from Sentiment_Analysis.logger import logging
from Sentiment_Analysis.exception import CustomException
from Sentiment_Analysis.entity.config_entity import DataValidationConfig
from Sentiment_Analysis.entity.artifact_entity import DataValidationArtifact


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

        self.expected_raw_schema = {
            "Unnamed: 0": int,
            "count": int,
            "hate_speech": int,
            "offensive_language": int,
            "neither": int,
            "class": int,
            "tweet": str
        }

        self.expected_imbalance_schema = {
            "id": int,
            "label": int,
            "tweet": str
        }

    def validate_schema(self, df: pd.DataFrame, expected_schema: dict, file_name: str) -> bool:
        try:
            logging.info(f"Validating schema for: {file_name}")
            actual_columns = list(df.columns)
            expected_columns = list(expected_schema.keys())

            if actual_columns != expected_columns:
                raise ValueError(f"Schema mismatch in {file_name}. Expected: {expected_columns}, Found: {actual_columns}")

            for col, expected_type in expected_schema.items():
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col} in {file_name}")

                # Validate type (basic check)
                actual_dtype = df[col].dtype
                if expected_type == int and not pd.api.types.is_integer_dtype(actual_dtype):
                    raise TypeError(f"Column '{col}' expected to be int, found {actual_dtype} in {file_name}")
                elif expected_type == str and not pd.api.types.is_string_dtype(actual_dtype):
                    raise TypeError(f"Column '{col}' expected to be str, found {actual_dtype} in {file_name}")

            return True
        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self, df: pd.DataFrame, file_name: str) -> bool:
        try:
            logging.info(f"Checking for missing values in {file_name}")
            if df.isnull().sum().sum() > 0:
                raise ValueError(f"Missing values found in {file_name}")
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process...")

            raw_df = pd.read_csv(self.config.raw_data_path)
            imbalance_df = pd.read_csv(self.config.imbalance_data_path)

            raw_schema_ok = self.validate_schema(raw_df, self.expected_raw_schema, "raw_data.csv")
            imbalance_schema_ok = self.validate_schema(imbalance_df, self.expected_imbalance_schema, "imbalanced_data.csv")

            raw_missing_ok = self.validate_missing_values(raw_df, "raw_data.csv")
            imbalance_missing_ok = self.validate_missing_values(imbalance_df, "imbalanced_data.csv")

            artifact = DataValidationArtifact(
                        
                        schema_validation_status=True,  # or actual result from schema check
                        missing_values_status=True,     # or actual result from missing values check
                        validated_raw_data_path=self.config.raw_data_path,  # or cleaned/validated path
                        validated_imbalance_data_path=self.config.imbalance_data_path,  # or cleaned/validated path
                        raw_data_file_path=self.config.raw_data_path,
                        imbalance_data_file_path=self.config.imbalance_data_path,
                        validation_status=True,  # final status of validation
                        message="Schema and missing values validated successfully.",
                        schema_file_path=self.config.schema_file_path
)

            logging.info(f"Data validation successful: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)
