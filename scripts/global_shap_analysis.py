import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import dagshub

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Connect to DagsHub + MLflow
# --------------------------------------------------

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)

print("DagsHub + MLflow connected")


# --------------------------------------------------
# 2. Load transformed test data
# --------------------------------------------------

test_data = pd.read_csv(
    "artifacts/data_transformation/test.csv"
)

print("\nTest data shape:")
print(test_data.shape)

print("\nTest data columns:")
print(test_data.columns.tolist())


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X_test = test_data.drop(
    columns=["loan_status"]
)

y_test = test_data["loan_status"]

print("\nX_test shape:")
print(X_test.shape)

print("\nX_test columns:")
print(X_test.columns.tolist())


# --------------------------------------------------
# 4. Load Champion XGBoost model
# --------------------------------------------------

MODEL_URI = "models:/CreditRiskModel@champion"

xgb_model = mlflow.xgboost.load_model(
    MODEL_URI
)

print("\nChampion XGBoost model loaded successfully")


# --------------------------------------------------
# 5. Create SHAP TreeExplainer
# --------------------------------------------------

explainer = shap.TreeExplainer(
    xgb_model
)

print("\nSHAP TreeExplainer created successfully")


# --------------------------------------------------
# 6. Calculate SHAP values
# --------------------------------------------------

shap_values = explainer(
    X_test
)

print("\nSHAP values shape:")
print(shap_values.values.shape)


# --------------------------------------------------
# 7. Calculate Global SHAP Importance
# --------------------------------------------------

global_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": (
        abs(shap_values.values)
        .mean(axis=0)
    ),
})


# --------------------------------------------------
# 8. Sort by importance
# --------------------------------------------------

global_importance = global_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)


# --------------------------------------------------
# 9. Display result
# --------------------------------------------------

print("\nGlobal SHAP importance:")
print(
    global_importance.to_string(index=False)
)


# --------------------------------------------------
# 10. Map transformed features to original features
# --------------------------------------------------

FEATURE_MAPPING = {
    "numerical__person_age": "person_age",
    "numerical__person_income": "person_income",
    "numerical__person_emp_length": "person_emp_length",
    "numerical__loan_amnt": "loan_amnt",
    "numerical__loan_int_rate": "loan_int_rate",
    "numerical__loan_percent_income": "loan_percent_income",
    "numerical__cb_person_cred_hist_length":
        "cb_person_cred_hist_length",

    "categorical__person_home_ownership_MORTGAGE":
        "person_home_ownership",
    "categorical__person_home_ownership_OTHER":
        "person_home_ownership",
    "categorical__person_home_ownership_OWN":
        "person_home_ownership",
    "categorical__person_home_ownership_RENT":
        "person_home_ownership",

    "categorical__loan_intent_DEBTCONSOLIDATION":
        "loan_intent",
    "categorical__loan_intent_EDUCATION":
        "loan_intent",
    "categorical__loan_intent_HOMEIMPROVEMENT":
        "loan_intent",
    "categorical__loan_intent_MEDICAL":
        "loan_intent",
    "categorical__loan_intent_PERSONAL":
        "loan_intent",
    "categorical__loan_intent_VENTURE":
        "loan_intent",

    "ordinal__loan_grade": "loan_grade",

    "binary__cb_person_default_on_file":
        "cb_person_default_on_file",
}


global_importance["original_feature"] = (
    global_importance["feature"]
    .map(FEATURE_MAPPING)
)


# --------------------------------------------------
# 11. Verify feature mapping
# --------------------------------------------------

print("\nFeature mapping:")
print(
    global_importance[
        [
            "feature",
            "original_feature",
            "mean_abs_shap",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 12. Aggregate into business-level features
# --------------------------------------------------

business_global_importance = (
    global_importance
    .groupby(
        "original_feature",
        as_index=False
    )
    .agg(
        mean_abs_shap=(
            "mean_abs_shap",
            "sum"
        )
    )
)


# --------------------------------------------------
# 13. Business-friendly names
# --------------------------------------------------

BUSINESS_FEATURE_NAMES = {
    "person_age": "Age",
    "person_income": "Income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan-to-income ratio",
    "cb_person_cred_hist_length":
        "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan intent",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file":
        "Previous default history",
}


business_global_importance["business_feature"] = (
    business_global_importance["original_feature"]
    .map(BUSINESS_FEATURE_NAMES)
)


# --------------------------------------------------
# 14. Sort by business-level importance
# --------------------------------------------------

business_global_importance = (
    business_global_importance
    .sort_values(
        by="mean_abs_shap",
        ascending=False
    )
)


# --------------------------------------------------
# 15. Display final business-level importance
# --------------------------------------------------

print("\nBusiness-level Global SHAP importance:")

print(
    business_global_importance[
        [
            "business_feature",
            "mean_abs_shap",
        ]
    ].to_string(index=False)
)

plt.figure(figsize=(10, 7))

shap.plots.waterfall(
    shap_values[0],
    max_display=11,
    show=False
)

plt.title(
    "Local SHAP Explanation - Customer"
)

plt.tight_layout()
plt.show()