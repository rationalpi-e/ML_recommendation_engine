"""Training loop for Neural Collaborative Filtering."""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.load_data import load_ratings
from src.data.split import temporal_split
from src.models.ncf_dataset import NCFDataset
from src.models.ncf import NCF


def train_ncf(epochs: int = 5, batch_size: int = 1024, lr: float = 0.001,
              embedding_dim: int = 32, n_negatives: int = 4, weight_decay: float = 1e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ratings = load_ratings()
    train_df, val_df, test_df = temporal_split(ratings)

    t0 = time.time()
    train_dataset = NCFDataset(train_df, n_negatives=n_negatives)
    print(f"Train dataset built in {time.time() - t0:.1f}s")

    t0 = time.time()
    val_dataset = NCFDataset(
        val_df, n_negatives=n_negatives,
        user_id_to_idx=train_dataset.user_id_to_idx,
        movie_id_to_idx=train_dataset.movie_id_to_idx,
    )
    print(f"Val dataset built in {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NCF(
        n_users=train_dataset.n_users,
        n_movies=train_dataset.n_movies,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Training ---
        model.train()
        train_loss = 0.0
        for users, movies, labels in train_loader:
            users, movies, labels = users.to(device), movies.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(users, movies)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, movies, labels in val_loader:
                users, movies, labels = users.to(device), movies.to(device), labels.to(device)

                predictions = model(users, movies)
                loss = criterion(predictions, labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_dataset)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ({elapsed:.1f}s)")

    return model, train_dataset


if __name__ == "__main__":
    model, train_dataset = train_ncf(epochs=5, batch_size=4096)