from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataValidationArtifact:
    schema_validation_status: bool
    missing_values_status: bool
    validated_raw_data_path: str
    validated_imbalance_data_path: str
    raw_data_file_path: str
    imbalance_data_file_path: str
    validation_status: bool
    message: str
    schema_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str
    tokenizer_path: str = 'tokenizer.pickle'


@dataclass
class ModelTrainerArtifacts: 
    trained_model_path:str
    x_test_path: list
    y_test_path: list

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool 

@dataclass
class ModelPusherArtifacts:
    bucket_name: str