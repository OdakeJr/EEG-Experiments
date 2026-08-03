model_dict = {
    "logreg": {
        "function": classical.train_logistic_regression,
        "params": {
            "max_iter": 1000,
            "C": 1.0
        }
    },
    "rf": {
        "function": classical.train_random_forest,
        "params": {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42
        }
    },
    "svm_rbf": {
        "function": classical.train_svm,
        "params": {
            "C": 1.0,
            "gamma": "scale"
        }
    },
    "nn_erm": {
        "function": classical.train_nn_erm,
        "params": {
            "hidden_dim": 128,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128
        }
    },
    "nn_l2": {
        "function": classical.train_nn_l2,
        "params": {
            "hidden_dim": 128,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128,
            "weight_decay": 1e-4
        }
    },
    "nn_dropout": {
        "function": classical.train_nn_dropout,
        "params": {
            "hidden_dim": 128,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128,
            "dropout_p": 0.5
        }
    },
    "dann": {
        "function": domain_aware.train_dann,
        "params": {
            "hidden_dim": 128,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128,
            "lambda_grl": 1.0,
            "lambda_domain": 1.0
        }
    },
    "vrex": {
        "function": domain_aware.train_vrex,
        "params": {
            "hidden_dim": 128,
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128,
            "lambda_vrex": 1.0
        }
    },
    "nn_ensemble": {
        "function": ensemble.train_standard_ensemble,
        "params": {
            "model_kwargs": {
                "hidden_dim": 128,
                "output_dim": 4
            },
            "epochs": 50,
            "lr": 1e-3,
            "batch_size": 128
        }
    },
    "nn_domain_only": {
        "function": ensemble.train_domain_agreement,
        "params": {
            "model_kwargs": {
                "hidden_dim": 128,
                "output_dim": 4
            },
            "erm_epochs": 50,
            "do_alignment": False,
            "lr": 1e-3,
            "batch_size": 128
        }
    },
    "nn_domain_agreement": {
        "function": ensemble.train_domain_agreement,
        "params": {
            "model_kwargs": {
                "hidden_dim": 128,
                "output_dim": 4
            },
            "erm_epochs": 50,
            "align_epochs": 10,
            "sequential_rounds": 3,
            "lambda_agree": 0.5,
            "disagreement_mode": "consensus",
            "disagreement_distance": "l2",
            "do_alignment": True,
            "lr": 1e-3,
            "batch_size": 128
        }
    }
}


fs_dict = {
    "none": {
        "function": stats_fs.fs_none,
        "params": {}
    },

    "mi_64": {
        "function": stats_fs.fs_mi,
        "params": {
            "n_features": n_features
        }
    },

    "anova_64": {
        "function": stats_fs.fs_anova,
        "params": {
            "n_features": n_features
        }
    },

    "variance_64": {
        "function": stats_fs.fs_variance,
        "params": {
            "n_features": n_features
        }
    },

    "pearson_64": {
        "function": stats_fs.fs_pearson,
        "params": {
            "n_features": n_features
        }
    },

    "chi2_64": {
        "function": stats_fs.fs_chi2,
        "params": {
            "n_features": n_features
        }
    },
}