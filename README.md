# Credit Risk Prediction MLOps

An end-to-end Machine Learning and MLOps project for predicting whether a loan applicant is likely to default.

The project covers the complete journey from data preparation and model training to model tracking, explainability, API serving, Docker, CI/CD, and cloud deployment.

---

## Project Overview

The system takes customer and loan information and predicts the risk of loan default.

The prediction is:

- `0` → Non-default
- `1` → Default

The project is not only about training a machine learning model. It also includes the complete MLOps workflow required to manage, test, explain, serve, and deploy the model.

### Complete Flow

```text
Customer Data
     ↓
Data Ingestion
     ↓
Data Validation
     ↓
Data Transformation
     ↓
Model Training
     ↓
Model Experimentation
     ↓
Model Evaluation
     ↓
MLflow Model Registry
     ↓
Champion Model
     ↓
FastAPI Prediction Service
     ↓
SHAP Explanation
     ↓
Frontend
     ↓
Docker
     ↓
GitHub Actions
     ↓
AWS Deployment Architecture
```

### Project Architecture

![Credit Risk Prediction MLOps Architecture](docs/architecture.png)

---

## Technology Stack

| Area | Technology |
|---|---|
| Programming | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Data Versioning | DVC |
| Remote Storage | DagsHub |
| API | FastAPI |
| Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS |
| Cloud Storage | Amazon S3 |
| Serverless | AWS Lambda |
| Database | DynamoDB |
| Monitoring | CloudWatch |
| Audit | CloudTrail |
| Security | IAM |

---

## Machine Learning Pipeline

The ML pipeline is managed through DVC.

The pipeline contains the following stages:

```text
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Trainer
      ↓
Model Experimentation
      ↓
Model Evaluation
      ↓
Model Registry
```

DVC helps make the pipeline reproducible and tracks the relationship between stages and artifacts.

The DVC remote is configured with DagsHub S3 storage.

---

## Data Transformation

The original dataset contains 11 input features.

The preprocessing pipeline handles different types of data separately.

### Numerical Features

- `person_age`
- `person_income`
- `person_emp_length`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_cred_hist_length`

Numerical features use:

- Median imputation
- Standard scaling

### Categorical Features

- `person_home_ownership`
- `loan_intent`

Categorical features use One-Hot Encoding.

### Ordinal Feature

- `loan_grade`

Loan grades are converted using the order:

```text
A → B → C → D → E → F → G
```

### Binary Feature

- `cb_person_default_on_file`

Values are encoded as:

```text
N → 0
Y → 1
```

The final transformed dataset contains **19 features**.

The fitted preprocessor is saved at:

```text
artifacts/data_transformation/preprocessor.pkl
```

---

## Model Experimentation

Three machine learning models were compared:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8565 | 0.7482 | 0.5162 | 0.6109 |
| Random Forest | 0.9362 | 0.9809 | 0.7215 | 0.8314 |
| XGBoost | 0.9357 | 0.9588 | 0.7370 | **0.8334** |

XGBoost achieved the best F1 score and was selected as the final model.

### XGBoost Tuning

Hyperparameter tuning was performed using 5-fold cross-validation.

The search included:

```text
n_estimators: [100, 200]
max_depth: [3, 5]
learning_rate: [0.01, 0.1]
subsample: [0.8, 1.0]
```

Best parameters:

```text
learning_rate = 0.1
max_depth = 5
n_estimators = 200
subsample = 1.0
```

The tuned model achieved:

```text
Cross-validation F1: 0.8323
Test F1:              0.8317
```

The original baseline XGBoost model remained the final selected model because its test F1 score was slightly higher:

```text
Final XGBoost F1: 0.8334
```

---

## Model Evaluation

The project uses a minimum F1-score threshold of:

```text
0.80
```

The final model achieved:

```text
F1 Score:      0.8333996023856859
Minimum F1:    0.8
Model Status:  Accepted
```

The evaluation result is saved at:

```text
artifacts/model_evaluation/evaluation_metrics.json
```

The model is accepted because:

```text
0.8334 > 0.80
```

---

## MLflow and Model Registry

MLflow is used for:

- Experiment tracking
- Model logging
- Model versioning
- Model registration

The registered model is:

```text
CreditRiskModel
```

The production-serving model is selected using the:

```text
champion
```

alias.

The project uses the MLflow Model Registry so the application does not need to depend on a hard-coded model version.

The logged model URI is stored in:

```text
artifacts/model_experimentation/model_uri.txt
```

Current logged model URI:

```text
models:/m-ee422c42b5204413aecbf7842dd1f208
```

---

## DVC and DagsHub

DVC is used to manage pipeline artifacts and data-related files.

The project uses DagsHub as the remote storage location.

The configured DVC remote is:

```text
dagshub-s3
```

The remote was verified successfully:

```text
Cache and remote 'dagshub-s3' are in sync.
```

Authentication credentials are kept outside the source code.

---

## Model Explainability with SHAP

SHAP is used to explain the model prediction.

The project uses:

```text
SHAP TreeExplainer
```

with the XGBoost model.

The explanation provides:

- Important features
- Customer feature values
- Feature importance
- Direction of model influence

The API converts the SHAP results into customer-friendly explanations such as:

```text
Feature
Value
Impact
Importance
```

The explanations describe how features influenced the model prediction.

They do **not** prove that a feature caused the outcome.

---

## Prediction Service

The prediction service provides two main capabilities:

```text
predict()
```

and

```text
explain()
```

The prediction service returns:

- Prediction
- Non-default probability
- Default probability
- Important feature explanations

The model and preprocessing pipeline are loaded before making predictions.

---

## FastAPI

The machine learning model is served through FastAPI.

Main endpoints:

```text
GET /health
POST /predict
```

### Health Endpoint

The health endpoint is used to verify that the API is running correctly.

### Prediction Endpoint

The `/predict` endpoint accepts customer and loan information and returns:

- Prediction
- Non-default probability
- Default probability
- SHAP-based explanations

Input validation is handled using Pydantic.

Invalid values such as unsupported loan grades are rejected by the API.

---

## Frontend

A simple web frontend is included in:

```text
src/api/frontend/
```

The frontend provides:

- Customer information form
- Prediction result
- Default risk interpretation
- Probability percentages
- Probability bars
- SHAP explanations
- Customer values
- Model influence information
- Make Another Prediction button

The frontend communicates with the FastAPI prediction endpoint.

---

## Docker

The FastAPI application is containerized using Docker.

The Docker image contains:

- Application source code
- Required Python dependencies
- Configuration
- Preprocessing artifact

The application runs on:

```text
Port 8000
```

The ML dependencies use CPU-based XGBoost to avoid unnecessary GPU packages.

The Docker image was tested locally with:

```text
Docker build
Docker run
Health check
Prediction request
```

---

## Testing

The project includes automated tests for:

- FastAPI health endpoint
- Prediction endpoint
- Invalid input validation
- Model prediction
- Champion model loading
- Prediction service
- Preprocessing output

Final test result:

```text
7 passed
4 warnings
```

The warnings were non-blocking dependency/deprecation warnings.

A focused SHAP/prediction-service test also passed successfully.

---

## GitHub Actions CI/CD

GitHub Actions is used to automatically verify the project.

The CI pipeline performs:

```text
Checkout Code
      ↓
Install Dependencies
      ↓
Configure DVC Authentication
      ↓
Check DVC Remote
      ↓
Pull DVC Artifacts
      ↓
Run Tests
      ↓
Build Docker Image
      ↓
Run Docker Container
      ↓
Health Check
      ↓
Stop Container
```

The GitHub Actions workflow was successfully verified with a green build.

DVC authentication is handled using GitHub Secrets.

No tokens or credentials are stored in the source code.

---

## Logging and Error Handling

The project contains a custom logging system.

The logger supports:

- File logging
- Console logging
- UTF-8 encoding
- Structured formatting
- Duplicate-handler protection

A custom exception class is also implemented to provide useful error information including:

- File name
- Line number
- Original error message

---

## AWS Deployment Architecture

The project was also implemented and verified on AWS.

The AWS architecture included:

```text
Customer Request
      ↓
Lambda Function URL
      ↓
AWS Lambda
      ↓
S3 Model + Preprocessor
      ↓
Prediction
      ↓
DynamoDB Prediction History

Lambda
  ├── CloudWatch → Logs and Monitoring
  ├── CloudTrail → AWS Activity Audit
  └── IAM → Permissions
```

### AWS Services Used

#### IAM

IAM was used to control permissions.

A dedicated Lambda execution role was created with limited permissions for:

- S3 model access
- DynamoDB prediction storage
- CloudWatch logging

#### S3

S3 was used to store:

```text
model/best_model.pkl
model/preprocessor.pkl
```

Public access was blocked and server-side encryption was enabled.

#### ECR

Amazon ECR was used to store the Lambda container image.

#### Lambda

AWS Lambda was used as the serverless prediction service.

The Lambda function:

1. Loads the model from S3
2. Loads the preprocessor from S3
3. Transforms customer data
4. Generates prediction
5. Calculates probabilities
6. Stores prediction history
7. Returns the prediction response

#### Lambda Function URL

A public HTTPS Function URL was used to expose the Lambda prediction service.

#### DynamoDB

DynamoDB stored prediction history including:

- Prediction ID
- Prediction
- Non-default probability
- Default probability
- Timestamp

#### CloudWatch

CloudWatch was used for Lambda logs and monitoring.

#### CloudTrail

CloudTrail was used to record AWS management activity for auditing.

#### VPC and Security Group

A custom VPC and security group were created to understand the AWS networking architecture.

The Lambda function was not attached to the custom VPC.

No NAT Gateway was created in order to avoid unnecessary cost.

---

## AWS Verification

The AWS Lambda prediction service was successfully tested.

A sample prediction returned:

```text
Prediction:              0
Non-default probability: 98.3253%
Default probability:      1.6746%
```

The prediction result was also verified against the DynamoDB record.

CloudWatch logs and CloudTrail activity were verified successfully.

After completing the AWS implementation and verification, the project AWS resources were deleted to avoid ongoing charges.

The final billing verification showed:

```text
Estimated Grand Total: $0.00
```

The main IAM user and MFA were kept.

---

## Project Structure

```text
credit-risk-prediction-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
│   ├── data_transformation/
│   ├── model_evaluation/
│   └── model_experimentation/
│
├── config/
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
│
├── docs/
│   └── architecture.png
│
├── lambda/
│   ├── Dockerfile
│   ├── lambda_function.py
│   └── requirements.txt
│
├── notebooks/
│
├── requirements/
│   ├── base.txt
│   ├── ml.txt
│   ├── tracking.txt
│   ├── api.txt
│   ├── deployment.txt
│   ├── dev.txt
│   └── ci.txt
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── components/
│   ├── config/
│   ├── constants/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipeline/
│   └── services/
│
├── tests/
│
├── .dvc/
├── .dockerignore
├── Dockerfile
├── pyproject.toml
├── dvc.yaml
├── dvc.lock
├── .gitignore
└── README.md
```

---

## Local Setup

Clone the repository and move into the project directory:

```bash
git clone https://github.com/kush2501/credit-risk-prediction-mlops.git
cd credit-risk-prediction-mlops
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
pip install -r requirements/tracking.txt
pip install -r requirements/api.txt
```

---

## Run the Pipeline

To reproduce the complete DVC pipeline:

```powershell
dvc repro
```

To check DVC status:

```powershell
dvc status
```

To verify the remote:

```powershell
dvc status -c --remote dagshub-s3
```

---

## Run the API

Start FastAPI with:

```powershell
uvicorn src.api.main:app --reload
```

The API will be available on:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Expected final result:

```text
7 passed
```

---

## Final Model Result

The final selected model is:

```text
XGBoost
```

Final F1 score:

```text
0.8334
```

Required minimum F1 score:

```text
0.80
```

Model accepted:

```text
Yes
```

---

## Final Project Flow

The complete project follows this workflow:

```text
Customer Data
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Experimentation
      ↓
Model Evaluation
      ↓
MLflow Tracking
      ↓
CreditRiskModel Registry
      ↓
Champion Model
      ↓
FastAPI
      ↓
Prediction + SHAP Explanation
      ↓
Frontend
      ↓
Docker
      ↓
GitHub Actions
      ↓
AWS Deployment
```

---

## Project Status

The project has been completed and verified across the major MLOps components.

- Machine Learning pipeline completed
- Model experimentation completed
- Final XGBoost model selected
- Model evaluation completed
- MLflow tracking completed
- Model Registry completed
- Champion model configured
- SHAP explainability completed
- FastAPI completed
- Frontend completed
- Docker completed
- DVC and DagsHub completed
- GitHub Actions CI/CD completed
- AWS architecture implemented and verified
- AWS resources cleaned up after verification
- Final tests passed
- Architecture documentation added

---

## Conclusion

This project demonstrates a complete production-style MLOps workflow for a credit risk prediction system.

It combines machine learning, data versioning, experiment tracking, model registry, explainability, API serving, frontend integration, containerization, CI/CD, and cloud architecture into one end-to-end project.

The final model achieved an F1 score of **0.8334**, exceeding the required minimum F1 score of **0.80**.
