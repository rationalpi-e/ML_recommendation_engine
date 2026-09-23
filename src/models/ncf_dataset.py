"""Negative sampling dataset for Neural Collaborative Filtering (vectorized)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    def __init__(self, train: pd.DataFrame, n_negatives: int = 4,
                 user_id_to_idx: dict = None, movie_id_to_idx: dict = None):
        self.n_negatives = n_negatives

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

        pos_users = train["user_id"].map(user_id_to_idx).values.astype(np.int64)
        pos_movies = train["movie_id"].map(movie_id_to_idx).values.astype(np.int64)

        movie_counts = np.bincount(pos_movies, minlength=self.n_movies)
        self.movie_popularity = movie_counts / movie_counts.sum()

        self._generate_samples_vectorized(pos_users, pos_movies)

    def _generate_samples_vectorized(self, pos_users, pos_movies):
        n_pos = len(pos_users)

        # Encode every (user, movie) pair as one integer, so membership
        # checks are fast vectorized array lookups instead of Python sets.
        seen_codes = np.sort(pos_users.astype(np.int64) * self.n_movies + pos_movies.astype(np.int64))

        # Repeat each user n_negatives times, so we can sample all negatives at once.
        neg_users = np.repeat(pos_users, self.n_negatives)
        neg_movies = np.random.choice(
            self.n_movies, size=len(neg_users), p=self.movie_popularity
        )

        # Vectorized collision check + resample loop (rare collisions, few rounds needed).
        for _ in range(10):
            candidate_codes = neg_users.astype(np.int64) * self.n_movies + neg_movies.astype(np.int64)
            is_collision = np.isin(candidate_codes, seen_codes, assume_unique=False)
            n_collisions = is_collision.sum()
            if n_collisions == 0:
                break
            neg_movies[is_collision] = np.random.choice(
                self.n_movies, size=n_collisions, p=self.movie_popularity
            )

        users = np.concatenate([pos_users, neg_users])
        movies = np.concatenate([pos_movies, neg_movies])
        labels = np.concatenate([np.ones(n_pos), np.zeros(len(neg_users))])

        self.users = torch.from_numpy(users).long()
        self.movies = torch.from_numpy(movies).long()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


if __name__ == "__main__":
    import time
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    t0 = time.time()
    dataset = NCFDataset(train, n_negatives=4)
    print(f"Dataset built in {time.time() - t0:.2f}s")

    print(f"Number of users: {dataset.n_users}")
    print(f"Number of movies: {dataset.n_movies}")
    print(f"Total samples: {len(dataset)}")