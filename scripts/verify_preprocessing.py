import pandas as pd

from src.utils.common import load_object


# ---------------------------------------------------------
# Step 1: Load saved preprocessor
# ---------------------------------------------------------

preprocessor_path = (
    "artifacts/data_transformation/preprocessor.pkl"
)

preprocessor = load_object(preprocessor_path)

print("====================================")
print("PREPROCESSOR LOADED SUCCESSFULLY")
print("====================================")


# ---------------------------------------------------------
# Step 2: Create one new customer
# ---------------------------------------------------------

customer_data = pd.DataFrame(
    [
        {
            "person_age": 35,
            "person_income": 50000,
            "person_home_ownership": "RENT",
            "person_emp_length": 5,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 10000,
            "loan_int_rate": 10.5,
            "loan_percent_income": 0.20,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 8,
        }
    ]
)


print("\nOriginal customer data:")
print(customer_data)


# ---------------------------------------------------------
# Step 3: Transform new customer
# ---------------------------------------------------------

transformed_data = preprocessor.transform(customer_data)


print("\n====================================")
print("TRANSFORMATION SUCCESSFUL")
print("====================================")

print("Original shape:", customer_data.shape)
print("Transformed shape:", transformed_data.shape)