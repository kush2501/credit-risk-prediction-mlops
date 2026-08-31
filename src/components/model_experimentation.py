import pandas as pd
import sys
import json 
import mlflow
import mlflow.xgboost
import dagshub



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.entity.config_entity import (
    ModelExperimentConfig,
    ModelExperimentArtifact,
)
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.common import save_object


class ModelExperimentation:

    def __init__(
        self,
        config: ModelExperimentConfig,
    ):
        self.config = config
    def load_data(self):
        """
        Load transformed training and testing datasets.
        """

        try:
            logger.info(
                "Loading experimentation training data"
            )

            train_df = pd.read_csv(
                self.config.train_data_path
            )

            logger.info(
                f"Experimentation training data loaded: "
                f"{train_df.shape}"
            )

            logger.info(
                "Loading experimentation testing data"
            )

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
    train_df,
    test_df,
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

            return (
                X_train,
                y_train,
                X_test,
                y_test,
            )

        except Exception as e:

            logger.exception(
                "Failed to prepare experimentation data"
            )

            raise CustomException(e, sys)

    def get_models(self):
        """
        Define candidate baseline models for experimentation.
        """
        try:

            logger.info(
                "Defining candidate baseline models"
            )

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
        Train all candidate baseline models, evaluate them,
        and log their parameters and metrics to MLflow.
        """
        try:

            logger.info(
                "Baseline model experimentation started"
            )

            model_results = {}
            trained_models = {}

            for model_name, model in models.items():

                logger.info(
                    f"Starting MLflow run for: {model_name}"
                )

                with mlflow.start_run(
                    run_name=model_name
                ):

                    # -----------------------------
                    # Log model information
                    # -----------------------------

                    mlflow.log_param(
                        "model",
                        model_name
                    )

                    mlflow.log_params(
                        model.get_params()
                    )

                    logger.info(
                        f"Training model: {model_name}"
                    )

                    # -----------------------------
                    # Train model
                    # -----------------------------

                    model.fit(
                        X_train,
                        y_train,
                    )

                    logger.info(
                        f"Training completed: {model_name}"
                    )

                    # -----------------------------
                    # Prediction
                    # -----------------------------

                    y_pred = model.predict(
                        X_test
                    )

                    # -----------------------------
                    # Calculate metrics
                    # -----------------------------

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

                    # -----------------------------
                    # Log metrics to MLflow
                    # -----------------------------

                    mlflow.log_metrics({
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    })

                    # -----------------------------
                    # Store trained model
                    # -----------------------------

                    trained_models[model_name] = model

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
                        f"MLflow run completed: {model_name}"
                    )

            logger.info(
                "Baseline model experimentation completed"
            )

            return (
                trained_models,
                model_results,
            )

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
        Select the best baseline model based on
        the highest F1-score.
        """
        try:

            logger.info(
                "Selecting best baseline model based on F1-score"
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
                f"Best baseline model selected: "
                f"{best_model_name}"
            )

            logger.info(
                f"Best baseline F1-score: "
                f"{best_f1_score:.4f}"
            )

            return (
                best_model_name,
                best_f1_score,
            )

        except Exception as e:

            logger.exception(
                "Best baseline model selection failed"
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
        Evaluate the best tuned model on
        unseen test data.
        """
        try:

            logger.info(
                "Tuned model evaluation started"
            )

            logger.info(
                "Generating predictions using tuned model"
            )

            y_pred = best_model.predict(
                X_test
            )

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
    trained_models,
    model_results,
    tuned_model,
    tuned_metrics,
    ):
        """
        Compare all baseline models and the tuned XGBoost
        model, then select the model with the highest
        F1-score.
        """
        try:

            logger.info(
                "Final model selection started"
            )

            trained_models[
                "Tuned XGBoost"
            ] = tuned_model

            model_results[
                "Tuned XGBoost"
            ] = tuned_metrics

            final_model_name = max(
                model_results,
                key=lambda model_name: model_results[
                    model_name
                ]["f1_score"],
            )

            final_model = trained_models[
                final_model_name
            ]

            final_metrics = model_results[
                final_model_name
            ]

            logger.info(
                f"Final model selected: "
                f"{final_model_name}"
            )

            logger.info(
                f"Final Accuracy: "
                f"{final_metrics['accuracy']:.4f}"
            )

            logger.info(
                f"Final Precision: "
                f"{final_metrics['precision']:.4f}"
            )

            logger.info(
                f"Final Recall: "
                f"{final_metrics['recall']:.4f}"
            )

            logger.info(
                f"Final F1-score: "
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


    def initiate_model_experimentation(self):
        """
        Run the complete model experimentation workflow,
        select the best model, log the final model to MLflow,
        save it locally, log artifacts, and return the final
        experimentation artifact.
        """

        try:

            logger.info(
                "Model Experimentation Pipeline Started"
            )

            # ==========================================================
            # Connect DagsHub with MLflow
            # ==========================================================

            dagshub.init(
                repo_owner="kush2501",
                repo_name="credit-risk-prediction-mlops",
                mlflow=True,
            )

            mlflow.set_experiment(
                "Credit Risk Model Experimentation"
            )

            # ==========================================================
            # Step 1: Load transformed datasets
            # ==========================================================

            train_df, test_df = self.load_data()

            # ==========================================================
            # Step 2: Separate features and target
            # ==========================================================

            (
                X_train,
                y_train,
                X_test,
                y_test,
            ) = self.prepare_data(
                train_df,
                test_df,
            )

            # ==========================================================
            # Step 3: Define baseline candidate models
            # ==========================================================

            models = self.get_models()

            # ==========================================================
            # Step 4: Train and evaluate baseline models
            # ==========================================================

            (
                trained_models,
                model_results,
            ) = self.train_and_evaluate_models(
                models,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            # ==========================================================
            # Step 5: Find best baseline model
            # ==========================================================

            (
                best_baseline_model_name,
                best_baseline_f1_score,
            ) = self.select_best_model(
                model_results
            )

            logger.info(
                f"Best baseline model: "
                f"{best_baseline_model_name}"
            )

            logger.info(
                f"Best baseline F1-score: "
                f"{best_baseline_f1_score:.4f}"
            )

            # ==========================================================
            # Step 6: Tune XGBoost
            # ==========================================================

            with mlflow.start_run(
                run_name="Tuned XGBoost"
            ):

                logger.info(
                    "Tuned XGBoost MLflow run started"
                )

                (
                    tuned_xgboost_model,
                    best_params,
                    best_cv_score,
                ) = self.tune_xgboost_model(
                    X_train,
                    y_train,
                )

                # Log tuned XGBoost parameters
                mlflow.log_params(
                    best_params
                )

                # Log cross-validation F1 score
                mlflow.log_metric(
                    "cv_f1_score",
                    best_cv_score,
                )

                # Evaluate tuned model
                tuned_metrics = self.evaluate_tuned_model(
                    tuned_xgboost_model,
                    X_test,
                    y_test,
                )

                # Log test metrics
                mlflow.log_metrics({
                    "accuracy": tuned_metrics["accuracy"],
                    "precision": tuned_metrics["precision"],
                    "recall": tuned_metrics["recall"],
                    "f1_score": tuned_metrics["f1_score"],
                })

                logger.info(
                    "Tuned XGBoost MLflow run completed"
                )

            # ==========================================================
            # Step 7: Compare all models and select final model
            # ==========================================================

            (
                final_model,
                final_model_name,
                final_metrics,
            ) = self.select_final_model(
                trained_models,
                model_results,
                tuned_xgboost_model,
                tuned_metrics,
            )

            logger.info(
                f"Final model selected: "
                f"{final_model_name}"
            )

            logger.info(
                f"Final F1-score: "
                f"{final_metrics['f1_score']:.4f}"
            )

            # ==========================================================
            # Step 8: Create Final Model MLflow Run
            # ==========================================================

            with mlflow.start_run(
                run_name="Final Model"
            ) as final_run:

                logger.info(
                    "Final model MLflow run started"
                )

                # ------------------------------------------------------
                # 8.1 Log model name
                # ------------------------------------------------------

                mlflow.log_param(
                    "model",
                    final_model_name,
                )

                # ------------------------------------------------------
                # 8.2 Log model parameters
                # ------------------------------------------------------

                if final_model_name == "Logistic Regression":

                    mlflow.log_params({
                        "max_iter": final_model.max_iter,
                        "random_state": final_model.random_state,
                    })

                elif final_model_name == "Random Forest":

                    mlflow.log_params({
                        "n_estimators": final_model.n_estimators,
                        "max_depth": final_model.max_depth,
                        "random_state": final_model.random_state,
                    })

                elif final_model_name == "XGBoost":

                    mlflow.log_params({
                        "n_estimators": final_model.n_estimators,
                        "max_depth": final_model.max_depth,
                        "learning_rate": final_model.learning_rate,
                        "subsample": final_model.subsample,
                        "random_state": final_model.random_state,
                    })

                # ------------------------------------------------------
                # 8.3 Log final model metrics
                # ------------------------------------------------------

                mlflow.log_metrics({
                    "accuracy": final_metrics["accuracy"],
                    "precision": final_metrics["precision"],
                    "recall": final_metrics["recall"],
                    "f1_score": final_metrics["f1_score"],
                })

                # ------------------------------------------------------
                # 8.4 Log final trained model
                # ------------------------------------------------------

                if final_model_name == "XGBoost":

                    logged_model = mlflow.xgboost.log_model(
                        xgb_model=final_model,
                        name="final_model",
                        model_format="json",
                    )

                else:

                    logged_model = mlflow.sklearn.log_model(
                        sk_model=final_model,
                        name="final_model",
                    )

                logger.info(
                    "Final model logged to MLflow successfully"
                )

                # ------------------------------------------------------
                # 8.5 Get MLflow 3.x Model URI
                # ------------------------------------------------------

                mlflow_model_uri = logged_model.model_uri

                logger.info(
                    f"MLflow model URI: "
                    f"{mlflow_model_uri}"
                )

                # ======================================================
                # Step 9: Save final model locally
                # ======================================================

                logger.info(
                    "Saving final best model"
                )

                save_object(
                    self.config.best_model_path,
                    final_model,
                )

                logger.info(
                    f"Best model saved at: "
                    f"{self.config.best_model_path}"
                )

                # ======================================================
                # Step 10: Save MLflow Model URI locally
                # ======================================================

                mlflow_uri_path = (
                    self.config.root_dir / "model_uri.txt"
                )

                with open(
                    mlflow_uri_path,
                    "w"
                ) as file:

                    file.write(
                        mlflow_model_uri
                    )

                logger.info(
                    f"MLflow model URI saved at: "
                    f"{mlflow_uri_path}"
                )

                # ======================================================
                # Step 11: Create experimentation JSON artifact
                # ======================================================

                model_experiment_artifact = (
                    ModelExperimentArtifact(
                        best_model_name=final_model_name,
                        best_model_f1_score=final_metrics["f1_score"],
                        best_model_path=self.config.best_model_path,
                        mlflow_model_uri=mlflow_model_uri,
                    )
                )

                experiment_artifact_path = (
                    self.config.root_dir
                    / "model_experimentation.json"
                )

                with open(
                    experiment_artifact_path,
                    "w"
                ) as file:

                    json.dump(
                        {
                            "best_model_name":
                                model_experiment_artifact.best_model_name,

                            "best_model_f1_score":
                                model_experiment_artifact.best_model_f1_score,

                            "best_model_path":
                                str(
                                    model_experiment_artifact.best_model_path
                                ),

                            "mlflow_model_uri":
                                model_experiment_artifact.mlflow_model_uri,
                        },
                        file,
                        indent=4,
                    )

                logger.info(
                    f"Model experimentation artifact saved at: "
                    f"{experiment_artifact_path}"
                )

                # ======================================================
                # Step 12: Log local files to MLflow Run Artifacts
                # ======================================================

                # Save best_model.pkl inside:
                # Final Model → Artifacts → model
                mlflow.log_artifact(
                    str(self.config.best_model_path),
                    artifact_path="model",
                )

                # Save model_uri.txt inside:
                # Final Model → Artifacts → metadata
                mlflow.log_artifact(
                    str(mlflow_uri_path),
                    artifact_path="metadata",
                )

                # Save model_experimentation.json inside:
                # Final Model → Artifacts → metadata
                mlflow.log_artifact(
                    str(experiment_artifact_path),
                    artifact_path="metadata",
                )

                logger.info(
                    "Final model artifacts logged to MLflow successfully"
                )

                # ======================================================
                # Final logs
                # ======================================================

                logger.info(
                    "Model Experiment Artifact created successfully"
                )

                logger.info(
                    "Model Experimentation Pipeline Completed"
                )

                logger.info(
                    f"Best model: {final_model_name}"
                )

                logger.info(
                    f"Best model F1-score: "
                    f"{final_metrics['f1_score']:.4f}"
                )

                logger.info(
                    f"Best model saved at: "
                    f"{self.config.best_model_path}"
                )

            # ==========================================================
            # Return artifact after MLflow run is safely closed
            # ==========================================================

            return model_experiment_artifact

        except Exception as e:

            logger.exception(
                "Model Experimentation Pipeline failed"
            )

            raise CustomException(e, sys)