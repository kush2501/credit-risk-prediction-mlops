from src.config.configuration import ConfigurationManager
from src.components.model_experimentation import (
    ModelExperimentation,
)


if __name__ == "__main__":

    config_manager = ConfigurationManager()

    config = (
        config_manager.get_model_experiment_config()
    )

    model_experimentation = ModelExperimentation(
        config=config
    )

    train_df, test_df = (
        model_experimentation.load_data()
    )

    X_train, y_train, X_test, y_test = (
        model_experimentation.prepare_data(
            train_df,
            test_df,
        )
    )

    print("\nExperimentation Data:")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    models = model_experimentation.get_models()

    print("\nCandidate Models:")

    for model_name, model in models.items():
        print(f"{model_name}: {model}")

    model_results = (
    model_experimentation.train_and_evaluate_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        )
    )

    print("\nModel Experiment Results:")

    for model_name, metrics in model_results.items():

        print(f"\n{model_name}")

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1_score']:.4f}"
        )

    param_grid = (
    model_experimentation.get_xgboost_param_grid()
    )

    print("\nXGBoost Parameter Grid:")

    for parameter, values in param_grid.items():
        print(f"{parameter}: {values}")

    (
    best_model,
    best_params,
    best_cv_f1_score,
    ) = model_experimentation.tune_xgboost_model(
        X_train=X_train,
        y_train=y_train,
    )

    print("\nXGBoost Hyperparameter Tuning Results:")

    print(
        f"\nBest Parameters: {best_params}"
    )

    print(
        f"Best CV F1 Score: "
        f"{best_cv_f1_score:.4f}"
    )

    print(
        f"\nBest Model:\n{best_model}"
    )

    tuned_model_metrics = (
    model_experimentation.evaluate_tuned_model(
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
        )
    )

    print("\nTuned XGBoost Final Evaluation:")

    print(
        f"Accuracy : "
        f"{tuned_model_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{tuned_model_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{tuned_model_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{tuned_model_metrics['f1_score']:.4f}"
    )

    baseline_model = models["XGBoost"]

    baseline_metrics = model_results["XGBoost"]

    (
        final_model,
        final_model_name,
        final_metrics,
    ) = model_experimentation.select_final_model(
        baseline_model=baseline_model,
        baseline_metrics=baseline_metrics,
        tuned_model=best_model,
        tuned_metrics=tuned_model_metrics,
    )

    print("\nFinal Model Selection:")

    print(f"Model Name: {final_model_name}")

    print(
        f"F1 Score  : "
        f"{final_metrics['f1_score']:.4f}"
    )