import os
import sys
import tensorflow
from tensorflow import keras
import pickle
import numpy as np
import pandas as pd
from Sentiment_Analysis.logger import logging
from Sentiment_Analysis.exception import CustomException
from tensorflow.keras.utils import pad_sequences
from Sentiment_Analysis.constants import *
from Sentiment_Analysis.configuration.gcloud_syncer import GCloudSync
from sklearn.metrics import confusion_matrix, classification_report
from Sentiment_Analysis.entity.config_entity import ModelEvaluationConfig
from Sentiment_Analysis.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts
)


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        :param data_transformation_artifacts: Output of data transformation stage
        """
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        """
        Fetch best model from gcloud storage and return its path.
        """
        try:
            logging.info("Entered get_best_model_from_gcloud method.")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(
                bucket_name=self.model_evaluation_config.BUCKET_NAME,
                folder_name=self.model_evaluation_config.MODEL_NAME,
                destination=self.model_evaluation_config.BEST_MODEL_DIR_PATH
            )

            best_model_path = os.path.join(
                self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                self.model_evaluation_config.MODEL_NAME
            )

            logging.info("Fetched best model from GCloud.")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, model_path: str) -> float:
        """
        Evaluate a model against the test dataset.

        :param model_path: Path to the model
        :return: accuracy of the model
        """
        try:
            logging.info(f"Evaluating model at: {model_path}")

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path[0], index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path[0], index_col=0)

            with open(self.data_transformation_artifacts.tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)

            x_test = x_test['tweet'].astype(str)
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            model = keras.models.load_model(model_path)
            evaluation_result = model.evaluate(test_sequences_matrix, y_test)
            accuracy = evaluation_result[1] if isinstance(evaluation_result, (list, tuple)) else evaluation_result

            predictions = model.predict(test_sequences_matrix)
            predicted_classes = [1 if pred[0] >= 0.5 else 0 for pred in predictions]

            cm = confusion_matrix(y_test, predicted_classes)
            report = classification_report(y_test, predicted_classes)

            logging.info(f"Model Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{report}")

            return accuracy
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Main method to initiate model evaluation process.
        Returns whether the trained model is better than the existing model or not.
        """
        try:
            logging.info("Initiating model evaluation...")

            trained_model_accuracy = self.evaluate(self.model_trainer_artifacts.trained_model_path)

            best_model_path = self.get_best_model_from_gcloud()

            if not os.path.isfile(best_model_path):
                logging.info("No best model found in GCloud. Accepting trained model.")
                is_model_accepted = True
            else:
                best_model_accuracy = self.evaluate(best_model_path)
                is_model_accepted = trained_model_accuracy >= best_model_accuracy
                logging.info(f"Trained Model Accuracy: {trained_model_accuracy}")
                logging.info(f"Best Model Accuracy: {best_model_accuracy}")
                logging.info("Trained model accepted." if is_model_accepted else "Trained model rejected.")

            return ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)

        except Exception as e:
            raise CustomException(e, sys)
