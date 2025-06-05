import os
import sys
from zipfile import ZipFile
from Sentiment_Analysis.logger import logging
from Sentiment_Analysis.exception import CustomException
from Sentiment_Analysis.configuration.gcloud_syncer import GCloudSync
from Sentiment_Analysis.entity.config_entity import DataIngestionConfig
from Sentiment_Analysis.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def push_data_to_gcloud(self, filepath: str, filename: str = None) -> None:
        """
        Push data to GCP Cloud Storage bucket
        
        Args:
            filepath: Complete path to the file or directory to upload
            filename: (Optional) Name of the file in the bucket. 
                     If not provided, uses the original filename from filepath
        """
        try:
            
            logging.info("Entered the push_data_to_gcloud method of Data ingestion class")
            
            
            if filename is None:
                # Extract just the filename if full path is provided
                filename = os.path.basename(filepath)
                filepath = os.path.dirname(filepath)
            
            self.gcloud.sync_folder_to_gcloud(
                bucket_name=self.data_ingestion_config.BUCKET_NAME,
                filepath=filepath,
                filename=filename
            )
            
            logging.info(f"File {filename} pushed to GCloud bucket {self.data_ingestion_config.BUCKET_NAME}")
            logging.info("Exited the push_data_to_gcloud method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    
    
    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_NAME,
                                                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                                                )
            
            logging.info("Exited the get_data_from_gcloud method of Data ingestion class")

        
        except Exception as e:
            raise CustomException(e, sys) from e



    
        
    
    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try: 
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e
        
    

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            local_dataset_path = self.data_ingestion_config.LOCAL_DATASET_PATH  # Safely access from config
            
            # Step 1: Push local dataset.zip to GCloud if it exists
            if os.path.exists(local_dataset_path):
                logging.info(f"Local zip file found at {local_dataset_path}, pushing to GCloud.")
                self.push_data_to_gcloud(
                    filepath=os.path.dirname(local_dataset_path),
                    filename=os.path.basename(local_dataset_path)
            )
            else:
                logging.warning(f"Local zip file not found at {local_dataset_path}, skipping upload.")

            self.get_data_from_gcloud()
            logging.info("Fetched the data from gcloud bucket")
        
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and split into train and valid")

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path= imbalance_data_file_path,
                raw_data_file_path = raw_data_file_path
            )

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e