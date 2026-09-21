class PredictionService:

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

    def explain(self, customer_data):

        transformed_data = self.preprocessor.transform(
            customer_data
        )

        shap_values = self.explainer(
            transformed_data
        )

        return shap_values