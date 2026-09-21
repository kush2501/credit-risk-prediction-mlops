from src.api.schemas import CustomerPredictionRequest
from src.api.schemas import CustomerPredictionResponse


customer = CustomerPredictionRequest(
    person_age=25,
    person_income=50000,
    person_home_ownership="RENT",
    person_emp_length=3,
    loan_intent="EDUCATION",
    loan_grade="C",
    loan_amnt=10000,
    loan_int_rate=10.5,
    loan_percent_income=0.20,
    cb_person_default_on_file="N",
    cb_person_cred_hist_length=5,
)

print("Schema validation successful!")

print("\nValidated customer:")
print(customer)

print("\nDictionary:")
print(customer.model_dump())


response = CustomerPredictionResponse(
    prediction=0,
    non_default_probability=0.98325354,
    default_probability=0.01674648,
)

print("\nResponse schema validation successful!")

print("\nValidated response:")
print(response)

print("\nResponse dictionary:")
print(response.model_dump())