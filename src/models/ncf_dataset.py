"""Negative sampling dataset for Neural Collaborative Filtering."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, train: pd.DataFrame, n_negatives: int = 4,
                 user_id_to_idx: dict = None, movie_id_to_idx: dict = None):
        """
        train: DataFrame with user_id, movie_id columns (positives).
        n_negatives: number of negative samples per positive.
        user_id_to_idx / movie_id_to_idx: shared ID mappings (so train/val/test
            all use the same indices — important for consistent embeddings).
        """
        self.n_negatives = n_negatives

        # Build or reuse ID mappings
        if user_id_to_idx is None:
            user_ids = train["user_id"].unique()
            user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        if movie_id_to_idx is None:
            movie_ids = train["movie_id"].unique()
            movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

        self.user_id_to_idx = user_id_to_idx
        self.movie_id_to_idx = movie_id_to_idx
        self.n_users = len(user_id_to_idx)
        self.n_movies = len(movie_id_to_idx)

        # Map real IDs to matrix indices
        self.user_idxs = train["user_id"].map(user_id_to_idx).values
        self.movie_idxs = train["movie_id"].map(movie_id_to_idx).values

        # For fast "has this user seen this movie" lookups during negative sampling
        self.user_seen = {}
        for u, m in zip(self.user_idxs, self.movie_idxs):
            self.user_seen.setdefault(u, set()).add(m)

        # Popularity-weighted sampling: movies that appear more often in
        # training data are more likely to be picked as negatives.
        movie_counts = np.bincount(self.movie_idxs, minlength=self.n_movies)
        self.movie_popularity = movie_counts / movie_counts.sum()

        # Pre-generate the full positive + negative training set
        self._generate_samples()

    def _generate_samples(self):
        users, movies, labels = [], [], []

        for u, m in zip(self.user_idxs, self.movie_idxs):
            # Positive example
            users.append(u)
            movies.append(m)
            labels.append(1.0)

            # Negative examples: sample movies this user hasn't seen,
            # weighted by overall popularity
            seen = self.user_seen[u]
            negatives_added = 0
            attempts = 0
            while negatives_added < self.n_negatives and attempts < self.n_negatives * 10:
                candidate = np.random.choice(self.n_movies, p=self.movie_popularity)
                attempts += 1
                if candidate not in seen:
                    users.append(u)
                    movies.append(candidate)
                    labels.append(0.0)
                    negatives_added += 1

        self.users = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    dataset = NCFDataset(train, n_negatives=4)

    print(f"Number of users: {dataset.n_users}")
    print(f"Number of movies: {dataset.n_movies}")
    print(f"Total samples (positives + negatives): {len(dataset)}")
    print(f"Expected total (train_size * (1 + n_negatives)): {len(train) * 5}")

    # Peek at a few samples
    for i in range(5):
        user, movie, label = dataset[i]
        print(f"User idx: {user.item()}, Movie idx: {movie.item()}, Label: {label.item()}")