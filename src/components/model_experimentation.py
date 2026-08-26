import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.entity.config_entity import ModelExperimentConfig
from src.exception.exception import CustomException
from src.logger.logger import logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import sys


class ModelExperimentation:

    def __init__(
        self,
        config: ModelExperimentConfig,
    ):
        """
        Initialize ModelExperimentation with configuration.
        """
        self.config = config

    def load_data(self):
        """
        Load transformed training and testing datasets.
        """
        try:
            logger.info("Loading experimentation training data")

            train_df = pd.read_csv(
                self.config.train_data_path
            )

            logger.info(
                f"Experimentation training data loaded: "
                f"{train_df.shape}"
            )

            logger.info("Loading experimentation testing data")

            test_df = pd.read_csv(
                self.config.test_data_path
            )

            logger.info(
                f"Experimentation testing data loaded: "
                f"{test_df.shape}"
            )

            return train_df, test_df

        except Exception as e:
            logger.exception(
                "Failed to load experimentation data"
            )

            raise CustomException(e, sys)

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """
        Separate input features and target variable
        for model experimentation.
        """
        try:
            logger.info(
                "Preparing data for model experimentation"
            )

            X_train = train_df.drop(
                columns=["loan_status"]
            )

            y_train = train_df["loan_status"]

            X_test = test_df.drop(
                columns=["loan_status"]
            )

            y_test = test_df["loan_status"]

            logger.info(
                f"Experiment X_train shape: {X_train.shape}"
            )

            logger.info(
                f"Experiment y_train shape: {y_train.shape}"
            )

            logger.info(
                f"Experiment X_test shape: {X_test.shape}"
            )

            logger.info(
                f"Experiment y_test shape: {y_test.shape}"
            )

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.exception(
                "Failed to prepare experimentation data"
            )

            raise CustomException(e, sys)

    def get_models(self):
        """
        Define candidate models for experimentation.
        """
        try:
            logger.info("Defining candidate models")

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=42,
                ),

                "XGBoost": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                ),
            }

            logger.info(
                f"Candidate models defined: "
                f"{list(models.keys())}"
            )

            return models

        except Exception as e:
            logger.exception(
                "Failed to define candidate models"
            )

            raise CustomException(e, sys)

    def train_and_evaluate_models(
        self,
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """
        Train all candidate models and evaluate them
        on the testing dataset.
        """
        try:
            logger.info("Model experimentation started")

            model_results = {}

            for model_name, model in models.items():

                logger.info(
                    f"Training model: {model_name}"
                )

                # Train model on training data
                model.fit(X_train, y_train)

                logger.info(
                    f"Training completed: {model_name}"
                )

                # Make predictions on unseen test data
                y_pred = model.predict(X_test)

                # Calculate evaluation metrics
                accuracy = accuracy_score(
                    y_test,
                    y_pred,
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )

                model_results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }

                logger.info(
                    f"{model_name} results - "
                    f"Accuracy: {accuracy:.4f}, "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"F1: {f1:.4f}"
                )

            logger.info(
                "Model experimentation completed"
            )

            return model_results

        except Exception as e:
            logger.exception(
                "Model experimentation failed"
            )

            raise CustomException(e, sys)


    def select_best_model(
    self,
    model_results,
    ):
        """
        Select the best model based on the highest F1-score.
        """
        try:
            logger.info(
                "Selecting best model based on F1-score"
            )

            best_model_name = max(
                model_results,
                key=lambda model_name: model_results[
                    model_name
                ]["f1_score"],
            )

            best_f1_score = model_results[
                best_model_name
            ]["f1_score"]

            logger.info(
                f"Best model selected: {best_model_name}"
            )

            logger.info(
                f"Best F1-score: {best_f1_score:.4f}"
            )

            return best_model_name, best_f1_score

        except Exception as e:
            logger.exception(
                "Best model selection failed"
            )

            raise CustomException(e, sys)

    def get_xgboost_param_grid(self):
        """
        Define hyperparameter values to test
        for the XGBoost model.
        """
        try:
            logger.info(
                "Defining XGBoost hyperparameter grid"
            )

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
            }

            logger.info(
                f"XGBoost parameter grid: {param_grid}"
            )

            return param_grid

        except Exception as e:
            logger.exception(
                "Failed to define XGBoost parameter grid"
            )

            raise CustomException(e, sys)

    def tune_xgboost_model(
    self,
    X_train,
    y_train,
    ):
        """
        Tune XGBoost hyperparameters using GridSearchCV
        and return the best trained model.
        """
        try:
            logger.info(
                "XGBoost hyperparameter tuning started"
            )

            param_grid = self.get_xgboost_param_grid()

            xgboost_model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            )

            logger.info(
                "Creating GridSearchCV for XGBoost"
            )

            grid_search = GridSearchCV(
                estimator=xgboost_model,
                param_grid=param_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1,
                verbose=1,
            )

            logger.info(
                "Starting GridSearchCV fitting"
            )

            grid_search.fit(
                X_train,
                y_train,
            )

            logger.info(
                "GridSearchCV fitting completed"
            )

            logger.info(
                f"Best parameters: "
                f"{grid_search.best_params_}"
            )

            logger.info(
                f"Best cross-validation F1-score: "
                f"{grid_search.best_score_:.4f}"
            )

            return (
                grid_search.best_estimator_,
                grid_search.best_params_,
                grid_search.best_score_,
            )

        except Exception as e:
            logger.exception(
                "XGBoost hyperparameter tuning failed"
            )

            raise CustomException(e, sys)

    def evaluate_tuned_model(
        self,
        best_model,
        X_test,
        y_test,
    ):
        """
        Evaluate the best tuned model on unseen test data.
        """
        try:
            logger.info(
                "Tuned model evaluation started"
            )

            logger.info(
                "Generating predictions using tuned model"
            )

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            precision = precision_score(
                y_test,
                y_pred,
            )

            recall = recall_score(
                y_test,
                y_pred,
            )

            f1 = f1_score(
                y_test,
                y_pred,
            )

            logger.info(
                f"Tuned model Accuracy: {accuracy:.4f}"
            )

            logger.info(
                f"Tuned model Precision: {precision:.4f}"
            )

            logger.info(
                f"Tuned model Recall: {recall:.4f}"
            )

            logger.info(
                f"Tuned model F1-score: {f1:.4f}"
            )

            logger.info(
                "Tuned model evaluation completed"
            )

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        except Exception as e:
            logger.exception(
                "Tuned model evaluation failed"
            )

            raise CustomException(e, sys)

    def select_final_model(
    self,
    baseline_model,
    baseline_metrics,
    tuned_model,
    tuned_metrics,
    ):
        """
        Compare baseline and tuned models and select
        the final model based on F1-score.
        """
        try:
            logger.info(
                "Final model selection started"
            )

            baseline_f1 = baseline_metrics["f1_score"]
            tuned_f1 = tuned_metrics["f1_score"]

            logger.info(
                f"Baseline model F1-score: "
                f"{baseline_f1:.4f}"
            )

            logger.info(
                f"Tuned model F1-score: "
                f"{tuned_f1:.4f}"
            )

            if tuned_f1 > baseline_f1:

                final_model = tuned_model
                final_model_name = "Tuned XGBoost"
                final_metrics = tuned_metrics

            else:

                final_model = baseline_model
                final_model_name = "Baseline XGBoost"
                final_metrics = baseline_metrics

            logger.info(
                f"Final model selected: "
                f"{final_model_name}"
            )

            logger.info(
                f"Final model F1-score: "
                f"{final_metrics['f1_score']:.4f}"
            )

            logger.info(
                "Final model selection completed"
            )

            return (
                final_model,
                final_model_name,
                final_metrics,
            )

        except Exception as e:

            logger.exception(
                "Final model selection failed"
            )

            raise CustomException(e, sys)