import lib.learning_algorithms.z_version_01.benchmark_architectures as my_models

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)


class DANN(nn.Module):
    def __init__(self, input_dim, n_domains, hidden_dim=128, lambda_grl=1.0):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Label predictor
        self.label_classifier = nn.Linear(hidden_dim, 1)

        # Domain classifier
        self.grl = GradientReversal(lambda_grl)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_domains)
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        label_logits = self.label_classifier(features)

        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        return label_logits, domain_logits
    
    

def train_dann(
    X_train, y_train, domains_train,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    lambda_grl=1.0,
    lambda_domain=1.0,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map domains to 0..K-1
    unique_domains = np.unique(domains_train)
    domain_mapping = {d: i for i, d in enumerate(unique_domains)}
    domains_mapped = np.array([domain_mapping[d] for d in domains_train])

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    d_tensor = torch.tensor(domains_mapped, dtype=torch.long).to(device)

    n_domains = len(unique_domains)

    model = DANN(
        input_dim=X_train.shape[1],
        n_domains=n_domains,
        hidden_dim=hidden_dim,
        lambda_grl=lambda_grl
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    label_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, d_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for _ in range(epochs):
        for xb, yb, db in loader:
            optimizer.zero_grad()
            label_logits, domain_logits = model(xb)

            label_loss = label_criterion(label_logits, yb)
            domain_loss = domain_criterion(domain_logits, db)

            loss = label_loss + lambda_domain * domain_loss
            loss.backward()
            optimizer.step()

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, _ = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(X):
        probs = predict_proba(X)
        return (probs >= 0.5).astype(int)

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }
    
    
def train_vrex(
    X_train, y_train, domains_train,
    hidden_dim=128,
    epochs=50,
    lr=1e-3,
    batch_size=128,
    lambda_vrex=1.0,
    **kwargs
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    d_tensor = torch.tensor(domains_train, dtype=torch.long).to(device)

    model = my_models.SimpleNN(X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    unique_domains = torch.unique(d_tensor)

    model.train()

    for _ in range(epochs):

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, d_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for xb, yb, db in loader:

            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)

            domain_risks = []
            for d in unique_domains:
                mask = (db == d)
                if mask.sum() > 0:
                    domain_risks.append(losses[mask].mean())

            domain_risks = torch.stack(domain_risks)

            mean_risk = domain_risks.mean()
            variance_penalty = domain_risks.var(unbiased=False)

            loss = mean_risk + lambda_vrex * variance_penalty

            loss.backward()
            optimizer.step()

    def predict_proba(X):
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(X):
        probs = predict_proba(X)
        return (probs >= 0.5).astype(int)

    return {
        "model": model,
        "predict_proba": predict_proba,
        "predict": predict
    }