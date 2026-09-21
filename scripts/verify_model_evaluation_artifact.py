from src.entity.config_entity import ModelEvaluationArtifact


if __name__ == "__main__":

    model_evaluation_artifact = ModelEvaluationArtifact(
        is_model_accepted=True,
        f1_score=0.8314,
        minimum_f1_score=0.80,
    )

    print(model_evaluation_artifact)