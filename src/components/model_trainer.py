import pandas as pd
import sys

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier

from src.entity.config_entity import (
    DataTransformationArtifact,
    ModelTrainerConfig,
    ModelTrainerArtifact,
)

from src.exception.exception import CustomException
from src.utils.common import save_object
from src.logger.logger import logger

class ModelTrainer:
    """
    Train a machine learning model using transformed
    training and testing datasets.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
        ):

        """
        Initialize ModelTrainer with configuration and
        transformed data artifact.
        """
        self.config = config
        self.data_transformation_artifact = (
            data_transformation_artifact
        )

    def load_data(self):
        """
        Load transformed training and testing data
        from the data transformation artifacts.
        """
        try:
            logger.info("Loading transformed training data")

            train_df = pd.read_csv(
                self.data_transformation_artifact.transformed_train_path
            )

            logger.info(
                f"Transformed training data loaded: {train_df.shape}"
            )

            logger.info("Loading transformed testing data")

            test_df = pd.read_csv(
                self.data_transformation_artifact.transformed_test_path
            )

            logger.info(
                f"Transformed testing data loaded: {test_df.shape}"
            )

            return train_df, test_df

        except Exception as e:
            logger.exception("Failed to load transformed data")
            raise CustomException(e, sys)

    def prepare_data(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ):
        """
        Separate input features and target variable from
        transformed training and testing datasets.
        """
        try:
            logger.info(
                "Separating features and target variable"
            )

            # Training data
            X_train = train_df.drop(
                columns=["loan_status"]
            )

            y_train = train_df["loan_status"]

            # Testing data
            X_test = test_df.drop(
                columns=["loan_status"]
            )

            y_test = test_df["loan_status"]

            logger.info(
                f"X_train shape: {X_train.shape}, "
                f"y_train shape: {y_train.shape}"
            )

            logger.info(
                f"X_test shape: {X_test.shape}, "
                f"y_test shape: {y_test.shape}"
            )

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.exception(
                "Failed to prepare training and testing data"
            )
            raise CustomException(e, sys)

    def train_model(
    self,
    X_train,
    y_train,
    ):
        """
        Train a Random Forest classification model
        using the prepared training data.
        """
        try:
            logger.info("Model Training Started")

            model = RandomForestClassifier(
                random_state=42
            )

            logger.info(
                "Fitting Random Forest model on training data"
            )

            model.fit(
                X_train,
                y_train,
            )

            logger.info(
                "Model Training Completed"
            )

            return model

        except Exception as e:
            logger.exception(
                "Model Training failed"
            )
            raise CustomException(e, sys)

    def predict_model(
    self,
    model,
    X_test,
    ):
        """
        Generate predictions on testing data
        using the trained model.
        """
        try:
            logger.info("Model Prediction Started")

            y_pred = model.predict(X_test)

            logger.info("Model Prediction Completed")

            return y_pred

        except Exception as e:
            logger.exception("Model Prediction failed")
            raise CustomException(e, sys)

    def evaluate_model(
    self,
    y_test,
    y_pred,
    ):
        """
        Evaluate model predictions by comparing
        predicted values with actual test values.
        """
        try:
            logger.info("Model Evaluation Started")

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            logger.info(
                f"Model Accuracy: {accuracy:.4f}"
            )

            logger.info("Model Evaluation Completed")

            return accuracy

        except Exception as e:
            logger.exception("Model Evaluation failed")
            raise CustomException(e, sys)

    def get_confusion_matrix(
    self,
    y_test,
    y_pred,
    ):
        """
        Calculate the confusion matrix to analyze
        correct and incorrect model predictions.
        """
        try:
            logger.info("Confusion Matrix Calculation Started")

            matrix = confusion_matrix(
                y_test,
                y_pred,
            )

            logger.info(
                f"Confusion Matrix:\n{matrix}"
            )

            logger.info(
                "Confusion Matrix Calculation Completed"
            )

            return matrix

        except Exception as e:
            logger.exception(
                "Confusion Matrix calculation failed"
            )
            raise CustomException(e, sys)

    def calculate_classification_metrics(self, y_true, y_pred):
        """
        Calculate precision, recall, and F1-score
        for the trained classification model.
        """
        try:
            logger.info("Classification Metrics Calculation Started")

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")

            logger.info("Classification Metrics Calculation Completed")

            return precision, recall, f1

        except Exception as e:
            logger.exception("Classification Metrics Calculation failed")
            raise CustomException(e, sys)

    def save_model(self, model):
        """
        Save the trained machine learning model
        for future predictions.
        """
        try:
            logger.info("Saving trained model")

            save_object(
                self.config.trained_model_path,
                model,
            )

            logger.info(
                f"Trained model saved at: "
                f"{self.config.trained_model_path}"
            )

        except Exception as e:
            logger.exception("Failed to save trained model")
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        """
        Run the complete model training workflow and return
        the final ModelTrainerArtifact.
        """
        try:
            logger.info("Model Training Pipeline Started")

            # Load transformed training and testing data.
            train_df, test_df = self.load_data()

            # Separate input features and target variable.
            X_train, y_train, X_test, y_test = self.prepare_data(
                train_df,
                test_df,
            )

            # Train the machine learning model.
            model = self.train_model(
                X_train,
                y_train,
            )

            # Generate predictions on testing data.
            y_pred = self.predict_model(
                model,
                X_test,
            )

            # Calculate model accuracy.
            accuracy = self.evaluate_model(
                y_test,
                y_pred,
            )

            # Calculate confusion matrix.
            confusion_matrix_result = self.get_confusion_matrix(
                y_test,
                y_pred,
            )

            # Calculate detailed classification metrics.
            precision, recall, f1 = self.calculate_classification_metrics(
                y_test,
                y_pred,
            )

            # Save the trained model.
            self.save_model(model)

            # Create the final training artifact.
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
            )

            logger.info(
                "Model Training Artifact created successfully"
            )

            logger.info("Model Training Pipeline Completed")

            return model_trainer_artifact

        except Exception as e:
            logger.exception("Model Training Pipeline failed")
            raise CustomException(e)