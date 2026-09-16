"""User-based Collaborative Filtering baseline."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class UserCFModel:
    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_id_to_idx = None
        self.idx_to_user_id = None
        self.movie_id_to_idx = None
        self.idx_to_movie_id = None

    def fit(self, train: pd.DataFrame):
        # Map real user/movie IDs to consecutive matrix indices
        user_ids = train["user_id"].unique()
        movie_ids = train["movie_id"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        rows = train["user_id"].map(self.user_id_to_idx)
        cols = train["movie_id"].map(self.movie_id_to_idx)
        values = train["rating"]

        # Build a sparse user-item matrix (users as rows, movies as columns)
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids), len(movie_ids)),
        )

        # Compute similarity between every pair of users
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        return self

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> list:
        exclude = exclude or set()

        if user_id not in self.user_id_to_idx:
            return []  # unseen user, no basis for recommendation

        user_idx = self.user_id_to_idx[user_id]

        # Similarity of this user to every other user
        sim_scores = self.user_similarity[user_idx]

        # Find the top-N most similar users (excluding the user themself)
        similar_user_idxs = np.argsort(sim_scores)[::-1]
        similar_user_idxs = similar_user_idxs[similar_user_idxs != user_idx][: self.n_neighbors]

        # Weighted sum of ratings from similar users for every movie
        neighbor_ratings = self.user_item_matrix[similar_user_idxs].toarray()
        neighbor_weights = sim_scores[similar_user_idxs].reshape(-1, 1)

        weighted_scores = (neighbor_ratings * neighbor_weights).sum(axis=0)

        # Rank movies by aggregated weighted score
        ranked_movie_idxs = np.argsort(weighted_scores)[::-1]

        recs = []
        for idx in ranked_movie_idxs:
            movie_id = self.idx_to_movie_id[idx]
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

    model = UserCFModel(n_neighbors=20).fit(train)

    sample_user = train["user_id"].iloc[0]
    seen = set(train[train["user_id"] == sample_user]["movie_id"])
    recs = model.recommend(sample_user, k=10, exclude=seen)
    print(f"Top 10 recommendations for user {sample_user}: {recs}")

    print("\nEvaluating on test set (this may take a minute)...")
    results = evaluate_model(model, train, test, k=10)
    print(f"User-based CF — Hit Rate@10: {results['hit_rate@k']:.4f}, NDCG@10: {results['ndcg@k']:.4f}")