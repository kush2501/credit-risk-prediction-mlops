import pandas as pd
import sys

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.entity.config_entity import DataTransformationConfig, DataTransformationArtifact
from src.exception.exception import CustomException
from src.logger.logger import logger
from src.utils.common import save_object

from src.entity.config_entity import (
    DataTransformationConfig,
    DataTransformationArtifact,
)


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize DataTransformation with the transformation configuration.
        """
        self.config = config

    def split_data(self):
        """
        Load the dataset, clean invalid values, and split it into
        training and testing datasets.
        """
        try:
            logger.info("Data Transformation Started")

            df = pd.read_csv(self.config.data_path)

            logger.info(f"Dataset loaded: {df.shape}")

            # Clean invalid values before splitting the dataset.
            df = self.clean_invalid_values(df)

            # Separate input features (X) and target variable (y).
            X = df.drop(columns=["loan_status"])
            y = df["loan_status"]

            # Split data into training and testing sets.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            logger.info(
                f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.exception("Data Transformation failed")
            raise CustomException(e, sys)

    def clean_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace invalid person_age values with NaN so they can
        be handled later by the numerical imputation pipeline.
        """
        try:
            logger.info("Data Cleaning Started")

            df = df.copy()

            # Treat ages above 100 as invalid values.
            invalid_age_count = (df["person_age"] > 100).sum()

            df.loc[df["person_age"] > 100, "person_age"] = pd.NA

            logger.info(
                f"Invalid person_age values replaced with NaN: "
                f"{invalid_age_count}"
            )

            return df

        except Exception as e:
            logger.exception("Data Cleaning failed")
            raise CustomException(e, sys)

    def create_numerical_pipeline(self):
        """
        Create a pipeline for numerical features.

        Missing values are filled using the median and then
        numerical features are standardized.
        """
        try:
            logger.info("Creating Numerical Pipeline")

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            logger.info("Numerical Pipeline Created")

            return numerical_pipeline

        except Exception as e:
            logger.exception("Numerical Pipeline creation failed")
            raise CustomException(e, sys)

    def create_categorical_encoder(self):
        """
        Create a One-Hot Encoder for nominal categorical features.

        Unknown categories are ignored during transformation.
        """
        try:
            logger.info("Creating Categorical Encoder")

            categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )

            logger.info("Categorical Encoder Created")

            return categorical_encoder

        except Exception as e:
            logger.exception("Categorical Encoder creation failed")
            raise CustomException(e, sys)

    def create_column_transformer(self):
        """
        Combine numerical, nominal, ordinal, and binary
        preprocessing into a single ColumnTransformer.
        """
        try:
            logger.info("Creating ColumnTransformer")

            # Numerical features require imputation and scaling.
            numerical_features = [
                "person_age",
                "person_income",
                "person_emp_length",
                "loan_amnt",
                "loan_int_rate",
                "loan_percent_income",
                "cb_person_cred_hist_length",
            ]

            # Nominal categorical features require One-Hot Encoding.
            nominal_features = [
                "person_home_ownership",
                "loan_intent",
            ]

            # Loan grade has a meaningful order.
            ordinal_features = [
                "loan_grade",
            ]

            # Binary feature: N / Y.
            binary_features = [
                "cb_person_default_on_file",
            ]

            # Explicit ordering for loan grades.
            ordinal_encoder = OrdinalEncoder(
                categories=[["A", "B", "C", "D", "E", "F", "G"]]
            )

            # Explicit ordering for binary values.
            binary_encoder = OrdinalEncoder(
                categories=[["N", "Y"]]
            )

            numerical_pipeline = self.create_numerical_pipeline()

            categorical_encoder = self.create_categorical_encoder()

            # Combine all preprocessing strategies.
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_features,
                    ),
                    (
                        "categorical",
                        categorical_encoder,
                        nominal_features,
                    ),
                    (
                        "ordinal",
                        ordinal_encoder,
                        ordinal_features,
                    ),
                    (
                        "binary",
                        binary_encoder,
                        binary_features,
                    ),
                ]
            )

            logger.info("ColumnTransformer Created")

            return preprocessor

        except Exception as e:
            logger.exception("ColumnTransformer creation failed")
            raise CustomException(e, sys)

    def transform_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
         ):
        
        """
        Fit the preprocessor only on training data, transform
        training and testing data, save the fitted preprocessor,
        and save transformed train/test datasets.
        """
        try:
            logger.info("Data Transformation Started")

            # Create the complete preprocessing pipeline.
            preprocessor = self.create_column_transformer()

            # ---------------------------------------------------------
            # FIT + TRANSFORM TRAIN DATA
            # ---------------------------------------------------------
            logger.info("Fitting preprocessor on training data")

            X_train_transformed = preprocessor.fit_transform(
                X_train
            )

            # ---------------------------------------------------------
            # TRANSFORM TEST DATA
            # ---------------------------------------------------------
            logger.info("Transforming test data")

            X_test_transformed = preprocessor.transform(
                X_test
            )

            logger.info(
                f"Transformed train shape: "
                f"{X_train_transformed.shape}"
            )

            logger.info(
                f"Transformed test shape: "
                f"{X_test_transformed.shape}"
            )

            # ---------------------------------------------------------
            # GET TRANSFORMED FEATURE NAMES
            # ---------------------------------------------------------
            feature_names = preprocessor.get_feature_names_out()

            logger.info(
                f"Number of transformed features: "
                f"{len(feature_names)}"
            )

            # ---------------------------------------------------------
            # CONVERT TRANSFORMED ARRAYS TO DATAFRAMES
            # ---------------------------------------------------------
            train_df = pd.DataFrame(
                X_train_transformed,
                columns=feature_names,
            )

            test_df = pd.DataFrame(
                X_test_transformed,
                columns=feature_names,
            )

            # ---------------------------------------------------------
            # ADD TARGET COLUMN
            # ---------------------------------------------------------
            train_df["loan_status"] = y_train.to_numpy()

            test_df["loan_status"] = y_test.to_numpy()

            logger.info(
                f"Final transformed train shape: {train_df.shape}"
            )

            logger.info(
                f"Final transformed test shape: {test_df.shape}"
            )

            # ---------------------------------------------------------
            # SAVE TRANSFORMED TRAIN DATA
            # ---------------------------------------------------------
            logger.info("Saving transformed training data")

            train_df.to_csv(
                self.config.train_data_path,
                index=False,
            )

            logger.info(
                f"Transformed training data saved at: "
                f"{self.config.train_data_path}"
            )

            # ---------------------------------------------------------
            # SAVE TRANSFORMED TEST DATA
            # ---------------------------------------------------------
            logger.info("Saving transformed testing data")

            test_df.to_csv(
                self.config.test_data_path,
                index=False,
            )

            logger.info(
                f"Transformed testing data saved at: "
                f"{self.config.test_data_path}"
            )

            # ---------------------------------------------------------
            # SAVE FITTED PREPROCESSOR
            # ---------------------------------------------------------
            logger.info("Saving fitted preprocessor")

            save_object(
                self.config.preprocessor_path,
                preprocessor,
            )

            logger.info(
                f"Preprocessor saved at: "
                f"{self.config.preprocessor_path}"
            )
        
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_path=self.config.train_data_path,
                transformed_test_path=self.config.test_data_path,
                preprocessor_path=self.config.preprocessor_path,
            )
            
            logger.info("Data Transformation Artifact created successfully")

            return data_transformation_artifact

        except Exception as e:
            logger.exception("Data Transformation failed")
            raise CustomException(e, sys)