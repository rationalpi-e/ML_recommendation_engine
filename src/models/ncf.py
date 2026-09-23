"""Neural Collaborative Filtering model architecture."""
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = 32,
                 hidden_dims: list = [64, 32, 16], dropout: float = 0.2):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # Build dense layers dynamically based on hidden_dims
        layers = []
        input_dim = embedding_dim * 2  # concatenated user + movie vectors

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))  # final output: single score
        layers.append(nn.Sigmoid())  # squash to 0-1 (probability-like)

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # Small random init for embeddings — helps stable early training
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)

        combined = torch.cat([user_vec, movie_vec], dim=-1)
        score = self.mlp(combined)

        return score.squeeze(-1)


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split
    from src.models.ncf_dataset import NCFDataset

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    dataset = NCFDataset(train, n_negatives=4)

    model = NCF(n_users=dataset.n_users, n_movies=dataset.n_movies, embedding_dim=32)

    print(model)

    # Sanity check: run a tiny batch through the model
    sample_users = dataset.users[:5]
    sample_movies = dataset.movies[:5]
    sample_labels = dataset.labels[:5]

    with torch.no_grad():
        predictions = model(sample_users, sample_movies)

    print(f"\nSample predictions: {predictions}")
    print(f"Sample true labels: {sample_labels}")