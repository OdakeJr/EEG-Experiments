import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# For Autoencoder (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================================
# 1) PCA
# ==========================================================

def dr_pca(X_train, y_train, domains_train,
           X_test, domains_test,
           n_components=64):

    reducer = PCA(n_components=n_components)
    Z_train = reducer.fit_transform(X_train)
    Z_test = reducer.transform(X_test)

    return Z_train, Z_test, {"model": reducer}


# ==========================================================
# 2) Kernel PCA (RBF)
# ==========================================================

def dr_kernel_pca(X_train, y_train, domains_train,
                  X_test, domains_test,
                  n_components=64,
                  gamma=None):

    reducer = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=False
    )

    Z_train = reducer.fit_transform(X_train)
    Z_test = reducer.transform(X_test)

    return Z_train, Z_test, {"model": reducer}


# ==========================================================
# 3) LDA (Supervised)
# ==========================================================

def dr_lda(X_train, y_train, domains_train,
           X_test, domains_test,
           n_components=1):

    # LDA maximum components = n_classes - 1
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    Z_train = lda.fit_transform(X_train, y_train)
    Z_test = lda.transform(X_test)

    return Z_train, Z_test, {"model": lda}


# ==========================================================
# 4) Autoencoder (Simple Stable Version - PyTorch)
# ==========================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


def dr_autoencoder(X_train, y_train, domains_train,
                   X_test, domains_test,
                   n_components=64,
                   epochs=50,
                   batch_size=128,
                   random_state=42):

    torch.manual_seed(random_state)

    input_dim = X_train.shape[1]

    model = Autoencoder(input_dim, n_components)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # --- validation split (like Keras) ---
    n = X_train_t.shape[0]
    val_size = int(0.1 * n)

    X_val = X_train_t[:val_size]
    X_tr = X_train_t[val_size:]

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(X_tr.size(0))
        X_tr_shuffled = X_tr[perm]

        for i in range(0, X_tr_shuffled.size(0), batch_size):
            batch = X_tr_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = criterion(val_recon, X_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Z_train = model.encoder(X_train_t).numpy()
        Z_test = model.encoder(X_test_t).numpy()

    return Z_train, Z_test, {"model": model.encoder}