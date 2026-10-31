
# main/domain_generalization_vision/params/common.py

# ============================================================
# Execution
# ============================================================

EXECUTION_PARAMS = {
    "max_workers": 1,
    "evaluation_workers": 1,
}


# ============================================================
# Representations
# ============================================================

FEATURE_EXTRACTION_PARAMS = []
FEATURE_SELECTION_PARAMS = []
SIGNAL_TRANSFORM_PARAMS = []

IMAGE_TRANSFORM_PARAMS = [
    {
        "name": "imagenet_224",
        "params": {"image_size": 224},
    },
]


# ============================================================
# Model
# ============================================================

MODEL_NAME = "mobilenet_v3_small"

MOBILENET_PARAMS = {
    "pretrained": True,
}


# ============================================================
# Neural training
# ============================================================

NEURAL_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "device": "mps",
    "seed": 0,
    "validation_fraction": 0.1,
    "patience": 20,
    "min_delta": 0.0,
    "num_workers": 0,
    "domains_per_batch": 3,
}


# ============================================================
# Domain generalization methods
# ============================================================

DG_METHODS = {
    # "irm": {"irm_lambda": 100.0},
    # "vrex": {"vrex_lambda": 10.0},
    # "groupdro": {"eta": 0.01},
    # "coral": {"coral_lambda": 1.0},
    # "mmd": {"mmd_lambda": 1.0},
    # "dg_dann": {"dann_lambda": 1.0, "dann_gamma": 10.0},
    # "mldg": {"mldg_alpha": 1e-3, "mldg_beta": 1.0},

    "residual_moments__linear_pooled": {
        "constraint_lambda": 1.0,
        "risk_weighting": "pooled",
        "constraint_scope": "pooled",
        "predicates": ["constant", "linear"],
        "normalize_features": True,
    },
    "residual_moments__linear_domain": {
        "constraint_lambda": 1.0,
        "risk_weighting": "pooled",
        "constraint_scope": "domain",
        "predicates": ["constant", "linear"],
        "normalize_features": True,
    },
    "residual_moments__quadratic_pooled": {
        "constraint_lambda": 1.0,
        "risk_weighting": "pooled",
        "constraint_scope": "pooled",
        "predicates": ["constant", "linear", "quadratic"],
        "normalize_features": True,
    },
    "residual_moments__quadratic_domain": {
        "constraint_lambda": 1.0,
        "risk_weighting": "pooled",
        "constraint_scope": "domain",
        "predicates": ["constant", "linear", "quadratic"],
        "normalize_features": True,
    },
}


# ============================================================
# Training configurations
# ============================================================

TRAINING_PARAMS = [
    {
        "name": "mobilenet_erm",
        "learning": f"neural_erm__{MODEL_NAME}",
        "model": MODEL_NAME,
        "model_params": MOBILENET_PARAMS,
        "training_params": NEURAL_PARAMS,
    },
    *[
        {
            "name": f"mobilenet_{method}",
            "learning": f"{method}__{MODEL_NAME}",
            "model": MODEL_NAME,
            "model_params": MOBILENET_PARAMS,
            "training_params": {
                **NEURAL_PARAMS,
                **method_params,
            },
        }
        for method, method_params in DG_METHODS.items()
    ],
]


# ============================================================
# Evaluation
# ============================================================

MODEL_EVALUATION_PARAMS = {}


# ============================================================
# Benchmark display
# ============================================================

DG_LABELS = {
    "irm": "IRM",
    "vrex": "VREx",
    "groupdro": "GroupDRO",
    "coral": "CORAL",
    "mmd": "MMD",
    "dg_dann": "DG-DANN",
    "mldg": "MLDG",
    "residual_moments__linear_pooled": "Residual Moments (Linear, Pooled)",
    "residual_moments__linear_domain": "Residual Moments (Linear, Domain)",
    "residual_moments__quadratic_pooled": "Residual Moments (Quadratic, Pooled)",
    "residual_moments__quadratic_domain": "Residual Moments (Quadratic, Domain)",
}

METHOD_DISPLAY = [
    {
        "learning_method": f"neural_erm__{MODEL_NAME}",
        "model_name": MODEL_NAME,
        "regime": "Baseline",
        "method": "ERM",
    },
    *[
        {
            "learning_method": f"{method}__{MODEL_NAME}",
            "model_name": MODEL_NAME,
            "regime": "Domain Generalization",
            "method": DG_LABELS[method],
        }
        for method in DG_METHODS
    ],
]


# ============================================================
# Benchmark
# ============================================================

BENCHMARK_TABLES_PARAMS = {
    "method_display": METHOD_DISPLAY,
    "tables": [
        {
            "name": "cross_domain_vision",
            "scenario": "cross_domain",
            "setting_column": "Dataset",
            "output_name": "cross_domain_vision_table.csv",
            "include_discrepancy": False,
            "filters": {"target_fraction": 0.0},
        },
    ],
}
