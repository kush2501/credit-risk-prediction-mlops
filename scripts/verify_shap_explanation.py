import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import dagshub

from src.utils.common import load_object


# ==========================================================
# 1. Connect DagsHub + MLflow
# ==========================================================

dagshub.init(
    repo_owner="kush2501",
    repo_name="credit-risk-prediction-mlops",
    mlflow=True,
)

print("DagsHub + MLflow connected")


# ==========================================================
# 2. Load Raw Dataset and Verify Target Mapping
# ==========================================================

raw_data = pd.read_csv(
    "artifacts/data_ingestion/credit_risk.csv"
)

print("\nLoan status distribution:")
print(
    raw_data["loan_status"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# 3. Load Fitted Preprocessor
# ==========================================================

preprocessor = load_object(
    "artifacts/data_transformation/preprocessor.pkl"
)

print("\nPreprocessor loaded successfully")


# ==========================================================
# 4. Define Champion Model URI
# ==========================================================

MODEL_URI = "models:/CreditRiskModel@champion"


# ==========================================================
# 5. Load Underlying XGBoost Champion Model
# ==========================================================

xgb_model = mlflow.xgboost.load_model(MODEL_URI)

print("Underlying XGBoost model loaded successfully")


# ==========================================================
# 6. Create New Customer Data
# ==========================================================

customer_data = pd.DataFrame([{
    "person_age": 25,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.20,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5,
}])


print("\nCustomer input:")
print(customer_data)


# ==========================================================
# 7. Transform Customer Data
# ==========================================================

transformed_data = preprocessor.transform(
    customer_data
)

print(
    "\nTransformed shape:",
    transformed_data.shape
)


# ==========================================================
# 8. Generate Model Prediction
# ==========================================================

prediction = xgb_model.predict(
    transformed_data
)

print("\nModel prediction:")
print(prediction)


# ==========================================================
# 9. Generate Prediction Probabilities
# ==========================================================

probability = xgb_model.predict_proba(
    transformed_data
)

print("\nModel probability:")
print(probability)


# ==========================================================
# 10. Create SHAP TreeExplainer
# ==========================================================

explainer = shap.TreeExplainer(
    xgb_model
)

print("\nSHAP TreeExplainer created successfully")


# ==========================================================
# 11. Calculate SHAP Values
# ==========================================================

shap_values = explainer(
    transformed_data
)

print(
    "\nSHAP values shape:",
    shap_values.values.shape
)


# ==========================================================
# 12. Get Transformed Feature Names
# ==========================================================

feature_names = (
    preprocessor.get_feature_names_out()
)

print(
    "Number of feature names:",
    len(feature_names)
)


# ==========================================================
# 13. Create SHAP Explanation DataFrame
# ==========================================================

explanation_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values.values[0],
})


# ==========================================================
# 14. Calculate Absolute SHAP Values
# ==========================================================

explanation_df["abs_shap_value"] = (
    explanation_df["shap_value"].abs()
)


# ==========================================================
# 15. Sort Features by Contribution Magnitude
# ==========================================================

explanation_df = explanation_df.sort_values(
    by="abs_shap_value",
    ascending=False,
)


# ==========================================================
# 16. Display Top SHAP Features
# ==========================================================

print("\nTop SHAP features:")

print(
    explanation_df
    .head(10)
    .to_string(index=False)
)


# ==========================================================
# 17. Display SHAP Base Value
# ==========================================================

print("\nSHAP base value:")
print(explainer.expected_value)


# ==========================================================
# 18. Display All SHAP Values
# ==========================================================

print("\nSHAP values:")
print(shap_values.values)

# ==========================================================
# 19. Inspect SHAP Output Configuration
# ==========================================================

print("\nSHAP model output configuration:")

print(
    "Model output:",
    explainer.model.model_output
)

print(
    "Feature perturbation:",
    explainer.feature_perturbation
)

# ==========================================================
# 20. Inspect Raw Model Output
# ==========================================================

raw_output = xgb_model.predict(
    transformed_data,
    output_margin=True
)

print("\nRaw model output:")
print(raw_output)


# ==========================================================
# 21. Verify SHAP Additivity
# ==========================================================

shap_sum = (
    explainer.expected_value
    + shap_values.values[0].sum()
)

print("\nSHAP reconstructed output:")
print(shap_sum)

print("\nXGBoost raw model output:")
print(raw_output[0])


# ==========================================================
# 22. Inspect Preprocessor Transformers
# ==========================================================

print("\nPreprocessor transformers:")

for name, transformer, columns in preprocessor.transformers_:
    print(f"\nTransformer: {name}")
    print(f"Columns: {columns}")


# ==========================================================
# 23. Inspect Transformed Customer Data
# ==========================================================

transformed_df = pd.DataFrame(
    transformed_data,
    columns=feature_names
)

print("\nTransformed customer features:")

print(
    transformed_df.T.to_string()
)


print("\nDetailed preprocessor structure:")
print(preprocessor)

print("\nNamed transformers:")
for name, transformer in preprocessor.named_transformers_.items():
    print(f"\n{name}:")
    print(transformer)


# 24. Inspect categorical encoder
categorical_encoder = (
    preprocessor.named_transformers_["categorical"]
)

print("\nCategorical encoder:")
print(categorical_encoder)

print("\nCategorical categories:")
for column, categories in zip(
    ["person_home_ownership", "loan_intent"],
    categorical_encoder.categories_
):
    print(f"{column}: {categories}")


categorical_feature_names = (
    categorical_encoder.get_feature_names_out(
        ["person_home_ownership", "loan_intent"]
    )
)

print("\nCategorical transformed feature names:")
print(categorical_feature_names)

feature_mapping = {}

# Numerical
numerical_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

for feature in numerical_features:
    feature_mapping[f"numerical__{feature}"] = feature


# Categorical
categorical_columns = [
    "person_home_ownership",
    "loan_intent",
]

for column, categories in zip(
    categorical_columns,
    categorical_encoder.categories_
):
    for category in categories:
        transformed_name = f"{column}_{category}"
        technical_name = f"categorical__{transformed_name}"

        feature_mapping[technical_name] = column


# Ordinal
feature_mapping["ordinal__loan_grade"] = "loan_grade"


# Binary
feature_mapping[
    "binary__cb_person_default_on_file"
] = "cb_person_default_on_file"


print("\nSHAP feature mapping:")

for technical, original in feature_mapping.items():
    print(f"{technical}  -->  {original}")

# 25. Create SHAP explanation table

explanation_df = pd.DataFrame({
    "technical_feature": feature_names,
    "transformed_value": transformed_data[0],
    "shap_value": shap_values.values[0],
})

# Map technical feature to original feature
explanation_df["original_feature"] = (
    explanation_df["technical_feature"]
    .map(feature_mapping)
)

# Contribution strength
explanation_df["abs_shap_value"] = (
    explanation_df["shap_value"].abs()
)

# Sort by contribution strength
explanation_df = explanation_df.sort_values(
    by="abs_shap_value",
    ascending=False,
)

print("\nSHAP Explanation Table:")
print(
    explanation_df.to_string(index=False)
)

# 26. Add SHAP direction

explanation_df["direction"] = explanation_df["shap_value"].apply(
    lambda x: "toward_default" if x > 0 else "toward_non_default"
)

print("\nSHAP directions:")
print(
    explanation_df[
        [
            "original_feature",
            "shap_value",
            "direction",
        ]
    ].to_string(index=False)
)

# 27. Add contribution strength

def get_strength(abs_shap_value):
    if abs_shap_value >= 0.5:
        return "strong"
    elif abs_shap_value >= 0.1:
        return "moderate"
    else:
        return "weak"


explanation_df["strength"] = (
    explanation_df["abs_shap_value"]
    .apply(get_strength)
)



print("\nSHAP strength:")
print(
    explanation_df[
        [
            "original_feature",
            "shap_value",
            "direction",
            "strength",
        ]
    ].to_string(index=False)
)

# 28. Business-friendly feature names

BUSINESS_FEATURE_NAMES = {
    "person_age": "Age",
    "person_income": "Income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan-to-income ratio",
    "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan intent",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Previous default history",
}

explanation_df["business_feature"] = (
    explanation_df["original_feature"]
    .map(BUSINESS_FEATURE_NAMES)
)

print("\nBusiness-friendly SHAP table:")
print(
    explanation_df[
        [
            "business_feature",
            "shap_value",
            "direction",
            "strength",
        ]
    ].to_string(index=False)
)

# 29. Top SHAP contributors

top_features = (
    explanation_df
    .sort_values(
        by="abs_shap_value",
        ascending=False
    )
    .head(5)
)

print("\nTop 5 SHAP contributors:")
print(
    top_features[
        [
            "business_feature",
            "shap_value",
            "direction",
            "strength",
        ]
    ].to_string(index=False)
)

# ==========================================================
# 30. Aggregate SHAP values at original business-feature level
# ==========================================================

business_shap_df = (
    explanation_df
    .groupby("original_feature", as_index=False)
    .agg(
        shap_value=("shap_value", "sum")
    )
)

# Contribution magnitude

business_shap_df["abs_shap_value"] = (
    business_shap_df["shap_value"].abs()
)

# Direction

business_shap_df["direction"] = (
    business_shap_df["shap_value"]
    .apply(
        lambda x:
        "toward_default"
        if x > 0
        else "toward_non_default"
    )
)

# Business-friendly name

business_shap_df["business_feature"] = (
    business_shap_df["original_feature"]
    .map(BUSINESS_FEATURE_NAMES)
)

# Contribution strength

business_shap_df["strength"] = (
    business_shap_df["abs_shap_value"]
    .apply(get_strength)
)

# Sort by contribution strength

business_shap_df = (
    business_shap_df
    .sort_values(
        by="abs_shap_value",
        ascending=False
    )
)

print("\nBusiness-level SHAP explanation:")

print(
    business_shap_df[
        [
            "business_feature",
            "shap_value",
            "abs_shap_value",
            "direction",
            "strength",
        ]
    ].to_string(index=False)
)

# 31. Add original customer values to business-level SHAP
customer_values = {
    "person_age": customer_data["person_age"].iloc[0],
    "person_income": customer_data["person_income"].iloc[0],
    "person_emp_length": customer_data["person_emp_length"].iloc[0],
    "loan_amnt": customer_data["loan_amnt"].iloc[0],
    "loan_int_rate": customer_data["loan_int_rate"].iloc[0],
    "loan_percent_income": customer_data["loan_percent_income"].iloc[0],
    "cb_person_cred_hist_length": customer_data[
        "cb_person_cred_hist_length"
    ].iloc[0],
    "person_home_ownership": customer_data[
        "person_home_ownership"
    ].iloc[0],
    "loan_intent": customer_data["loan_intent"].iloc[0],
    "loan_grade": customer_data["loan_grade"].iloc[0],
    "cb_person_default_on_file": customer_data[
        "cb_person_default_on_file"
    ].iloc[0],
}

business_shap_df["customer_value"] = (
    business_shap_df["original_feature"]
    .map(customer_values)
)

print("\nBusiness-level SHAP with customer values:")

print(
    business_shap_df[
        [
            "business_feature",
            "customer_value",
            "shap_value",
            "direction",
        ]
    ].to_string(index=False)
)

# 32. Top factors supporting non-default

supporting_factors = (
    business_shap_df[
        business_shap_df["shap_value"] < 0
    ]
    .sort_values(
        by="abs_shap_value",
        ascending=False
    )
    .head(4)
)

print("\nTop factors supporting non-default:")
print(
    supporting_factors[
        [
            "business_feature",
            "customer_value",
            "shap_value",
            "strength",
        ]
    ].to_string(index=False)
)

# 33. Top factors pushing toward default

risk_factors = (
    business_shap_df[
        business_shap_df["shap_value"] > 0
    ]
    .sort_values(
        by="abs_shap_value",
        ascending=False
    )
    .head(4)
)

print("\nTop factors pushing toward default:")
print(
    risk_factors[
        [
            "business_feature",
            "customer_value",
            "shap_value",
            "strength",
        ]
    ].to_string(index=False)
)

# 34. Generate customer-facing explanation

def generate_explanation(
    prediction,
    probabilities,
    supporting_factors,
    risk_factors,
):
    if prediction == 0:
        prediction_label = "Non-default"
    else:
        prediction_label = "Default"

    print("\n" + "=" * 60)
    print("CUSTOMER CREDIT RISK EXPLANATION")
    print("=" * 60)

    print(f"\nPrediction: {prediction_label}")

    print(
        f"Non-default probability: "
        f"{probabilities[0][0] * 100:.2f}%"
    )

    print(
        f"Default probability: "
        f"{probabilities[0][1] * 100:.2f}%"
    )

    print("\nMain factors supporting the prediction:")

    for _, row in supporting_factors.iterrows():
        print(
            f"- {row['business_feature']} "
            f"({row['customer_value']}): "
            f"{row['strength']} contribution "
            f"toward non-default"
        )

    print("\nFactors pushing toward higher default-risk:")

    for _, row in risk_factors.iterrows():
        print(
            f"- {row['business_feature']} "
            f"({row['customer_value']}): "
            f"{row['strength']} contribution "
            f"toward default"
        )

# ==========================================================
# 35. Generate Final Customer Explanation
# ==========================================================

generate_explanation(
    prediction=prediction[0],
    probabilities=probability,
    supporting_factors=supporting_factors,
    risk_factors=risk_factors,
)