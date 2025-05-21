import os
import io
import sys
from tensorflow import keras
import pickle
from PIL import Image
from Sentiment_Analysis.logger import logging
from Sentiment_Analysis.constants import *
from Sentiment_Analysis.exception import CustomException
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from Sentiment_Analysis.configuration.gcloud_syncer import GCloudSync
from Sentiment_Analysis.components.data_transforamation import DataTransformation
from Sentiment_Analysis.entity.config_entity import DataTransformationConfig
from Sentiment_Analysis.entity.artifact_entity import DataIngestionArtifacts


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.gcloud = GCloudSync()

         # Provide actual paths here
        imbalance_data_path = os.path.join("artifacts", "data", "imbalance.csv")
        raw_data_path = os.path.join("artifacts", "data", "raw.csv")

        data_ingestion_artifacts = DataIngestionArtifacts(
            imbalance_data_file_path=imbalance_data_path,
            raw_data_file_path=raw_data_path
        )
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig(),data_ingestion_artifacts=data_ingestion_artifacts)


    
    def get_model_from_gcloud(self) -> str:
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            # Path to model directory
            model_dir_path = self.model_path  # e.g., 'artifacts/PredictModel'
            # Full path to model file inside model_dir_path
            best_model_path = os.path.join(model_dir_path, self.model_name)  # e.g., 'artifacts/PredictModel/model.h5'

            # Check if directory exists and model file exists inside it
            if not os.path.exists(model_dir_path) or not os.path.isfile(best_model_path):
                logging.info("Model not found locally, downloading from Google Cloud Storage...")
                os.makedirs(model_dir_path, exist_ok=True)
                self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, model_dir_path)
                # Check again if file exists after sync
                if not os.path.isfile(best_model_path):
                    raise FileNotFoundError(f"Model file not found after syncing: {best_model_path}")
            else:
                logging.info("Model found locally, skipping download.")

            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e



    
    def predict(self,best_model_path,text):
        """load image, returns cuda tensor"""
        logging.info("Running the predict function")
        try:
            
            load_model=keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text=self.data_transformation.concat_data_cleaning(text)
            text = [text]            
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)
            pred = load_model.predict(padded)
            
            print("pred", pred)
            if pred>0.5:

                print("hate and abusive")
                return "hate and abusive"
            else:
                print("no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def run_pipeline(self,text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:

            best_model_path: str = self.get_model_from_gcloud() 
            predicted_text = self.predict(best_model_path,text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e
