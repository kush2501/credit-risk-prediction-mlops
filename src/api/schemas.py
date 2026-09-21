from typing import Literal

from pydantic import BaseModel, Field


class CustomerPredictionRequest(BaseModel):

    person_age: int = Field(
        ...,
        ge=18,
        description="Age of the applicant"
    )

    person_income: float = Field(
        ...,
        gt=0,
        description="Annual income of the applicant"
    )

    person_home_ownership: Literal[
        "MORTGAGE",
        "OTHER",
        "OWN",
        "RENT",
    ] = Field(
        ...,
        description="Home ownership status"
    )

    person_emp_length: float = Field(
        ...,
        ge=0,
        description="Employment length in years"
    )

    loan_intent: Literal[
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ] = Field(
        ...,
        description="Purpose of the loan"
    )

    loan_grade: Literal[
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
    ] = Field(
        ...,
        description="Loan grade"
    )

    loan_amnt: float = Field(
        ...,
        gt=0,
        description="Loan amount"
    )

    loan_int_rate: float = Field(
        ...,
        gt=0,
        description="Loan interest rate"
    )

    loan_percent_income: float = Field(
        ...,
        ge=0,
        le=1,
        description="Loan amount as percentage of income"
    )

    cb_person_default_on_file: Literal[
        "N",
        "Y",
    ] = Field(
        ...,
        description="Previous default history"
    )

    cb_person_cred_hist_length: float = Field(
        ...,
        ge=0,
        description="Credit history length"
    )


class CustomerPredictionResponse(BaseModel):

    prediction: int = Field(
        ...,
        description="Prediction: 0 = non-default, 1 = default"
    )

    non_default_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of non-default"
    )

    default_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of default"
    )