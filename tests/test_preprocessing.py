import pandas as pd

from src.utils.common import load_object


def test_preprocessing_output_shape():

    # Load saved preprocessor
    preprocessor_path = (
        "artifacts/data_transformation/preprocessor.pkl"
    )

    preprocessor = load_object(preprocessor_path)

    # Create one new customer
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

    # Verify raw input shape
    assert customer_data.shape == (1, 11)

    # Transform customer data
    transformed_data = preprocessor.transform(customer_data)

    # Verify transformed output shape
    assert transformed_data.shape == (1, 19)