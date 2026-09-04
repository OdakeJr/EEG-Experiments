def wrap_model(
    model,
    predict,
    predict_proba,
    classes,
    training_history=None,
    artifacts=None,
):
    """Standardize the output of all training methods."""

    return {
        "model": model,
        "predict": predict,
        "predict_proba": predict_proba,
        "classes": classes,
        "training_history": training_history,
        "artifacts": artifacts or {},
    }