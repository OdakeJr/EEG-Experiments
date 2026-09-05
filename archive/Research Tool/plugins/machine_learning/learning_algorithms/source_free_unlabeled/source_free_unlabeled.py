import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.abspath("../../../../"))

from archieve.lib.learning_algorithms.helper import wrap_model


# ============================================================
# 1. Frozen Source Model
# ============================================================

def train_frozen_source(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    **kwargs
):
    """Use pretrained source model without target adaptation."""

    if pretrained_model is None:
        raise ValueError("Frozen Source requires a pretrained model.")

    # pretrained_model follows our standardized wrapped format
    model = pretrained_model["model"]
    predict = pretrained_model["predict"]
    predict_proba = pretrained_model["predict_proba"]
    classes = pretrained_model["classes"]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=None,
        artifacts={
            "scenario": "source_free_unlabeled",
            "method": "frozen_source",
            "target_labels_used": False,
            "n_target_samples": 0 if X_target is None else len(X_target),
        }
    )
    
def train_shot(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=30,
    lr=1e-4,
    batch_size=128,
    pseudo_weight=0.3,
    entropy_weight=1.0,
    random_state=42,
    **kwargs
):
    """Source Hypothesis Transfer (SHOT)."""

    if pretrained_model is None:
        raise ValueError("SHOT requires a pretrained source model.")
    if X_target is None:
        raise ValueError("SHOT requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Copy source model so the original model is preserved
    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])

    if not hasattr(model, "feature_extractor") or not hasattr(model, "classifier"):
        raise ValueError("SHOT requires a model with feature_extractor and classifier.")

    # Freeze source hypothesis (classifier)
    for param in model.classifier.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        model.feature_extractor.parameters(),
        lr=lr
    )

    X_target = np.asarray(X_target)
    X_tensor = torch.tensor(X_target, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(
        X_tensor,
        torch.arange(len(X_tensor))
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    def obtain_pseudo_labels():
        model.eval()

        with torch.no_grad():
            X = X_tensor.to(device)
            features = model.extract_features(X)
            probs = torch.softmax(model.classifier(features), dim=1)

            # Normalize features for cosine-distance clustering
            features = torch.cat(
                [features, torch.ones(len(features), 1, device=device)], dim=1
            )
            features = nn.functional.normalize(features, dim=1)

            # Soft class centroids
            centroids = probs.T @ features
            centroids /= probs.sum(dim=0).unsqueeze(1).clamp_min(1e-8)
            centroids = nn.functional.normalize(centroids, dim=1)

            pseudo = torch.argmax(features @ centroids.T, dim=1)

            # One centroid-refinement step
            one_hot = nn.functional.one_hot(
                pseudo, num_classes=len(classes)
            ).float()

            centroids = one_hot.T @ features
            centroids /= one_hot.sum(dim=0).unsqueeze(1).clamp_min(1.0)
            centroids = nn.functional.normalize(centroids, dim=1)

            return torch.argmax(
                features @ centroids.T, dim=1
            ).cpu()

    history = {"loss": [], "im_loss": [], "pseudo_loss": []}

    for _ in range(epochs):
        pseudo_labels = obtain_pseudo_labels()
        model.train()

        total_loss = total_im = total_pseudo = 0.0

        for xb, idx in loader:
            xb = xb.to(device)
            pseudo = pseudo_labels[idx].to(device)

            features = model.extract_features(xb)
            logits = model.classifier(features)
            probs = torch.softmax(logits, dim=1)

            # Information maximization
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            mean_probs = probs.mean(dim=0)
            diversity = (mean_probs * torch.log(mean_probs + 1e-8)).sum()
            im_loss = entropy + diversity

            pseudo_loss = nn.functional.cross_entropy(logits, pseudo)

            loss = (
                entropy_weight * im_loss
                + pseudo_weight * pseudo_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_im += im_loss.item()
            total_pseudo += pseudo_loss.item()

        n = len(loader)
        history["loss"].append(total_loss / n)
        history["im_loss"].append(total_im / n)
        history["pseudo_loss"].append(total_pseudo / n)

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_unlabeled",
            "method": "shot",
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )
    
def train_nrc(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=30,
    lr=1e-4,
    batch_size=128,
    k_neighbors=5,
    k_expanded=5,
    reciprocal_weight=1.0,
    nonreciprocal_weight=0.1,
    expanded_weight=0.1,
    diversity_weight=1.0,
    random_state=42,
    **kwargs
):
    """Neighborhood Reciprocity Clustering (NRC)."""

    if pretrained_model is None:
        raise ValueError("NRC requires a pretrained source model.")
    if X_target is None:
        raise ValueError("NRC requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])
    X_target = np.asarray(X_target)

    if not hasattr(model, "feature_extractor") or not hasattr(model, "classifier"):
        raise ValueError("NRC requires feature_extractor and classifier.")

    n_samples = len(X_target)
    K = min(k_neighbors, n_samples - 1)
    KK = min(k_expanded, n_samples - 1)

    X_tensor = torch.tensor(X_target, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        X_tensor, torch.arange(n_samples)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # --------------------------------------------------------
    # Initial feature and score banks
    # --------------------------------------------------------

    model.eval()
    with torch.no_grad():
        X_all = X_tensor.to(device)
        features = F.normalize(model.extract_features(X_all), dim=1)
        scores = torch.softmax(model.classifier(features), dim=1)

    feature_bank = features.detach().clone()
    score_bank = scores.detach().clone()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        "loss": [],
        "neighbor_loss": [],
        "expanded_loss": [],
        "diversity_loss": [],
    }

    # --------------------------------------------------------
    # Target adaptation
    # --------------------------------------------------------

    for _ in range(epochs):
        model.train()
        totals = np.zeros(4)

        for xb, idx in loader:
            xb, idx = xb.to(device), idx.to(device)

            features = model.extract_features(xb)
            features_norm = F.normalize(features, dim=1)
            logits = model.classifier(features)
            probs = torch.softmax(logits, dim=1)

            with torch.no_grad():
                feature_bank[idx] = features_norm.detach()
                score_bank[idx] = probs.detach()

                # K nearest neighbors
                similarity = features_norm @ feature_bank.T
                similarity.scatter_(1, idx[:, None], -float("inf"))
                idx_near = similarity.topk(K, dim=1).indices
                score_near = score_bank[idx_near]

                # Neighbors of neighbors
                neighbor_features = feature_bank[idx_near]
                similarity_nn = torch.einsum(
                    "bkd,nd->bkn", neighbor_features, feature_bank
                )
                similarity_nn.scatter_(
                    2, idx_near.unsqueeze(-1), -float("inf")
                )
                idx_near_near = similarity_nn.topk(KK, dim=2).indices

                # Reciprocal-neighbor weighting
                reciprocal = (
                    idx_near_near == idx[:, None, None]
                ).any(dim=2)

                weights = torch.where(
                    reciprocal,
                    torch.full_like(reciprocal, reciprocal_weight, dtype=torch.float),
                    torch.full_like(reciprocal, nonreciprocal_weight, dtype=torch.float),
                )

                score_expanded = score_bank[idx_near_near]

            # Neighbor consistency
            similarity_loss = -(probs[:, None, :] * score_near).sum(dim=2)
            neighbor_loss = (
                similarity_loss * weights
            ).sum() / weights.sum().clamp_min(1e-8)

            # Expanded-neighborhood consistency
            expanded_loss = -(
                probs[:, None, None, :] * score_expanded
            ).sum(dim=3).mean()

            # Prediction diversity
            mean_probs = probs.mean(dim=0)
            diversity_loss = (
                mean_probs * torch.log(mean_probs + 1e-8)
            ).sum()

            loss = (
                neighbor_loss
                + expanded_weight * expanded_loss
                + diversity_weight * diversity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals += [
                loss.item(),
                neighbor_loss.item(),
                expanded_loss.item(),
                diversity_loss.item(),
            ]

        totals /= len(loader)
        for key, value in zip(history, totals):
            history[key].append(value)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_unlabeled",
            "method": "nrc",
            "target_labels_used": False,
            "n_target_samples": n_samples,
            "k_neighbors": K,
            "k_expanded": KK,
        }
    )
    
def train_sfda_de(
    X_source=None, y_source=None, source_domains=None,
    X_target=None, y_target=None, target_domains=None,
    pretrained_model=None,
    epochs=30,
    lr=1e-4,
    batch_size=32,
    gamma=1.0,
    confidence_threshold=0.5,
    cdd_lambda=1.0,
    kmeans_iterations=10,
    random_state=42,
    **kwargs
):
    """Source-Free Domain Adaptation via Distribution Estimation (SFDA-DE)."""

    if pretrained_model is None:
        raise ValueError("SFDA-DE requires a pretrained source model.")
    if X_target is None:
        raise ValueError("SFDA-DE requires unlabeled target data.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(pretrained_model["model"]).to(device)
    classes = np.asarray(pretrained_model["classes"])
    X_target = np.asarray(X_target)

    if not hasattr(model, "feature_extractor") or not hasattr(model, "classifier"):
        raise ValueError("SFDA-DE requires feature_extractor and classifier.")

    # Freeze source classifier / anchors
    for p in model.classifier.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.feature_extractor.parameters(), lr=lr)
    X_all = torch.tensor(X_target, dtype=torch.float32, device=device)

    def rbf_mmd(x, y):
        scale = 1.0 / x.shape[1]
        return (
            torch.exp(-scale * torch.cdist(x, x).pow(2)).mean()
            + torch.exp(-scale * torch.cdist(y, y).pow(2)).mean()
            - 2 * torch.exp(-scale * torch.cdist(x, y).pow(2)).mean()
        )

    def estimate_distributions():
        model.eval()

        with torch.no_grad():
            features = model.extract_features(X_all)
            features_n = F.normalize(features, dim=1)

            # Classifier weights are source anchors
            anchors = F.normalize(model.classifier.weight.detach(), dim=1)
            centers = anchors.clone()

            # Spherical k-means
            for _ in range(kmeans_iterations):
                similarity = features_n @ centers.T
                pseudo = similarity.argmax(dim=1)

                new_centers = centers.clone()
                for k in range(len(classes)):
                    mask = pseudo == k
                    if mask.any():
                        new_centers[k] = F.normalize(
                            features_n[mask].mean(dim=0), dim=0
                        )
                centers = new_centers

            similarity = features_n @ centers.T
            confidence, pseudo = similarity.max(dim=1)

            # cosine distance = (1 - cosine) / 2
            distance = 0.5 * (1.0 - confidence)
            confident = distance < confidence_threshold

            distributions = {}

            for k in range(len(classes)):
                mask = (pseudo == k) & confident
                fk = features[mask]

                if len(fk) < 2:
                    continue

                target_mean = fk.mean(dim=0)

                # Estimated source mean follows source anchor direction
                source_mean = (
                    target_mean.norm()
                    * F.normalize(model.classifier.weight[k], dim=0)
                )

                centered = fk - target_mean
                covariance = gamma * centered.T @ centered / len(fk)
                covariance += 1e-5 * torch.eye(
                    covariance.shape[0], device=device
                )

                distributions[k] = {
                    "mean": source_mean,
                    "cov": covariance,
                    "indices": torch.where(mask)[0]
                }

        return pseudo, distributions

    history = {"loss": [], "cdd_loss": []}

    for _ in range(epochs):
        _, distributions = estimate_distributions()
        valid_classes = list(distributions.keys())

        if len(valid_classes) < 2:
            raise RuntimeError(
                "SFDA-DE requires confident samples from at least two classes."
            )

        model.train()
        losses = []

        n_steps = max(
            1,
            max(
                len(distributions[k]["indices"])
                for k in valid_classes
            ) // batch_size
        )

        for _ in range(n_steps):
            target_features = {}
            source_features = {}

            for k in valid_classes:
                indices = distributions[k]["indices"]
                selected = indices[
                    torch.randint(
                        len(indices),
                        (min(batch_size, len(indices)),),
                        device=device
                    )
                ]

                target_features[k] = model.extract_features(X_all[selected])

                dist = torch.distributions.MultivariateNormal(
                    distributions[k]["mean"],
                    covariance_matrix=distributions[k]["cov"]
                )

                source_features[k] = dist.sample(
                    (len(selected),)
                )

            intra = torch.stack([
                rbf_mmd(source_features[k], target_features[k])
                for k in valid_classes
            ]).mean()

            inter = torch.stack([
                rbf_mmd(source_features[i], target_features[j])
                for i in valid_classes
                for j in valid_classes
                if i != j
            ]).mean()

            cdd_loss = intra - inter
            loss = cdd_lambda * cdd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        history["loss"].append(np.mean(losses))
        history["cdd_loss"].append(np.mean(losses))

    def predict_proba(X):
        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.softmax(model(X), dim=1).cpu().numpy()

    def predict(X):
        return classes[np.argmax(predict_proba(X), axis=1)]

    return wrap_model(
        model,
        predict,
        predict_proba,
        classes,
        training_history=history,
        artifacts={
            "scenario": "source_free_unlabeled",
            "method": "sfda_de",
            "target_labels_used": False,
            "n_target_samples": len(X_target),
        }
    )




