"""ALS matrix factorization baseline (implicit feedback)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


class ALSModel:
    def __init__(self, factors: int = 50, regularization: float = 0.01,
                 iterations: int = 15, alpha: float = 40.0):
        """
        factors: size of latent vectors (the "50 numbers" per user/movie).
        regularization: penalty to prevent overfitting.
        iterations: number of alternating optimization rounds.
        alpha: confidence scaling factor for implicit feedback (see fit()).
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        self.alpha = alpha
        self.user_item_matrix = None
        self.user_id_to_idx = None
        self.idx_to_user_id = None
        self.movie_id_to_idx = None
        self.idx_to_movie_id = None

    def fit(self, train: pd.DataFrame):
        user_ids = train["user_id"].unique()
        movie_ids = train["movie_id"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        rows = train["user_id"].map(self.user_id_to_idx)
        cols = train["movie_id"].map(self.movie_id_to_idx)

        # Convert explicit ratings into implicit "confidence" values.
        # Higher rating = higher confidence the interaction was meaningful.
        confidence = 1 + self.alpha * train["rating"]

        self.user_item_matrix = csr_matrix(
            (confidence, (rows, cols)),
            shape=(len(user_ids), len(movie_ids)),
        )

        self.model.fit(self.user_item_matrix)

        return self

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> list:
        exclude = exclude or set()

        if user_id not in self.user_id_to_idx:
            return []

        user_idx = self.user_id_to_idx[user_id]

        # implicit's recommend() already excludes items the user has interacted
        # with in the training matrix by default (filter_already_liked_items=True)
        movie_idxs, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=k + len(exclude),  # over-fetch in case of extra exclusions
            filter_already_liked_items=True,
        )

        recs = []
        for idx in movie_idxs:
            movie_id = int(self.idx_to_movie_id[idx])
            if movie_id not in exclude:
                recs.append(movie_id)
            if len(recs) == k:
                break

        return recs


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split
    from src.eval.metrics import evaluate_model

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    model = ALSModel(factors=50, regularization=0.01, iterations=15).fit(train)

    sample_user = train["user_id"].iloc[0]
    seen = set(train[train["user_id"] == sample_user]["movie_id"])
    recs = model.recommend(sample_user, k=10, exclude=seen)
    print(f"Top 10 recommendations for user {sample_user}: {recs}")

    print("\nEvaluating on test set...")
    results = evaluate_model(model, train, test, k=10)
    print(f"ALS — Hit Rate@10: {results['hit_rate@k']:.4f}, NDCG@10: {results['ndcg@k']:.4f}")