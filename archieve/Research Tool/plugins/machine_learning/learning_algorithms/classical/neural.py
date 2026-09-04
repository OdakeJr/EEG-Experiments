# plugins/machine_learning/learning_algorithms/classical/neural.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from plugins.pipe import Pipe
from plugins.machine_learning.models.trained_model import TrainedModel
from plugins.machine_learning.models.neural_models import MLP


class MLPPipe(Pipe):
    """Classical ERM training using a multilayer perceptron."""

    def expand(self, input_nodes, params):
        return [
            {
                "inputs": [node.id],
                "params": params.copy()
            }
            for node in input_nodes
        ]

    def run(self, inputs, params):
        """Train and return an MLP model."""

        data = inputs[0]

        X_source = data.X_source
        y_source = data.y_source

        config = self._get_config(params)

        classes, X, y = self._prepare_data(
            X_source,
            y_source
        )

        model = self._build_model(
            X_source,
            classes,
            config
        )

        history = self._train(
            model,
            X,
            y,
            config
        )

        model = model.to("cpu")

        return TrainedModel(
            model=model,
            classes=classes,
            training_history=history,
            artifacts={
                "scenario": "classical",
                "method": "mlp"
            }
        )

    def _get_config(self, params):
        """Resolve training and model parameters."""

        return {
            "hidden_dims": params.get("hidden_dims", (128, 128)),
            "activation": params.get("activation", "relu"),
            "dropout": params.get("dropout", 0.0),
            "batch_norm": params.get("batch_norm", False),
            "epochs": params.get("epochs", 50),
            "lr": params.get("lr", 1e-3),
            "batch_size": params.get("batch_size", 128),
            "weight_decay": params.get("weight_decay", 0.0),
        }

    def _prepare_data(self, X_source, y_source):
        """Prepare source data and encode class labels."""

        classes = np.unique(y_source)

        class_to_idx = {
            label: index
            for index, label in enumerate(classes)
        }

        y_encoded = np.array(
            [class_to_idx[label] for label in y_source],
            dtype=np.int64
        )

        X = torch.tensor(
            X_source,
            dtype=torch.float32
        )

        y = torch.tensor(
            y_encoded,
            dtype=torch.long
        )

        return classes, X, y

    def _build_model(self, X_source, classes, config):
        """Create the MLP."""

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        return MLP(
            input_dim=X_source.shape[1],
            hidden_dims=config["hidden_dims"],
            output_dim=len(classes),
            activation=config["activation"],
            dropout=config["dropout"],
            batch_norm=config["batch_norm"],
            classes=classes
        ).to(device)

    def _train(self, model, X, y, config):
        """Train the MLP."""

        device = next(model.parameters()).device

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=config["batch_size"],
            shuffle=True
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        criterion = nn.CrossEntropyLoss()

        history = {
            "loss": []
        }

        for _ in range(config["epochs"]):

            model.train()
            total_loss = 0.0

            for xb, yb in loader:

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                loss = criterion(
                    model(xb),
                    yb
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(xb)

            history["loss"].append(
                total_loss / len(X)
            )

        return history