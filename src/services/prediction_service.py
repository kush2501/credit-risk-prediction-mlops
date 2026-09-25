class PredictionService:

    FEATURE_GROUPS = {
    "person_age": ["numerical__person_age"],
    "person_income": ["numerical__person_income"],
    "person_emp_length": ["numerical__person_emp_length"],
    "loan_amnt": ["numerical__loan_amnt"],
    "loan_int_rate": ["numerical__loan_int_rate"],
    "loan_percent_income": ["numerical__loan_percent_income"],
    "cb_person_cred_hist_length": [
        "numerical__cb_person_cred_hist_length"
    ],
    "person_home_ownership": [
        "categorical__person_home_ownership_MORTGAGE",
        "categorical__person_home_ownership_OTHER",
        "categorical__person_home_ownership_OWN",
        "categorical__person_home_ownership_RENT",
    ],
    "loan_intent": [
        "categorical__loan_intent_DEBTCONSOLIDATION",
        "categorical__loan_intent_EDUCATION",
        "categorical__loan_intent_HOMEIMPROVEMENT",
        "categorical__loan_intent_MEDICAL",
        "categorical__loan_intent_PERSONAL",
        "categorical__loan_intent_VENTURE",
    ],
    "loan_grade": ["ordinal__loan_grade"],
    "cb_person_default_on_file": [
        "binary__cb_person_default_on_file"
    ],
}

    def __init__(
        self,
        model,
        preprocessor,
        explainer,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer


    def predict(self, customer_data):
        transformed_data = self.preprocessor.transform(
            customer_data
        )

        prediction = self.model.predict(
            transformed_data
        )

        probability = self.model.predict_proba(
            transformed_data
        )

        return prediction, probability

    def explain(self, customer_data, top_n=5):
        transformed_data = self.preprocessor.transform(customer_data)

        shap_output = self.explainer(transformed_data)

        feature_names = self.preprocessor.get_feature_names_out()
        shap_values = shap_output.values[0]

        explanations = []

        for feature_name, feature_list in self.FEATURE_GROUPS.items():
            indices = [
                i for i, name in enumerate(feature_names)
                if name in feature_list
            ]

            contribution = float(shap_values[indices].sum())

            value = customer_data.iloc[0][feature_name]

            impact = (
                "increased_default_risk"
                if contribution > 0
                else "decreased_default_risk"
            )

            explanations.append({
                "feature": feature_name,
                "value": str(value),
                "impact": impact,
                "importance": abs(contribution),
            })

        explanations.sort(
            key=lambda item: item["importance"],
            reverse=True
        )

        return explanations[:top_n]