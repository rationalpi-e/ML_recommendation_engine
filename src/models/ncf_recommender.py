"""Wraps a trained NCF model with a recommend() interface for evaluation."""
import torch
import numpy as np


class NCFRecommender:
    def __init__(self, model, user_id_to_idx: dict, movie_id_to_idx: dict, device=None):
        self.model = model
        self.user_id_to_idx = user_id_to_idx
        self.idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}
        self.n_movies = len(movie_id_to_idx)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Pre-build the "all movie indices" tensor once, reused for every user
        self.all_movie_idxs = torch.arange(self.n_movies, dtype=torch.long, device=self.device)

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> list:
        exclude = exclude or set()

        if user_id not in self.user_id_to_idx:
            return []

        user_idx = self.user_id_to_idx[user_id]

        # Repeat this one user's index once per movie, so shapes line up
        user_idxs = torch.full((self.n_movies,), user_idx, dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.model(user_idxs, self.all_movie_idxs)

        scores = scores.cpu().numpy()

        # Sort movie indices by score, descending
        ranked_idxs = np.argsort(scores)[::-1]

        recs = []
        for idx in ranked_idxs:
            movie_id = int(self.idx_to_movie_id[idx])
            if movie_id not in exclude:
                recs.append(movie_id)
            if len(recs) == k:
                break

        return recs


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split
    from src.models.train_ncf import train_ncf
    from src.models.ncf_recommender import NCFRecommender
    from src.eval.metrics import evaluate_model

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    print("Training NCF...")
    model, train_dataset = train_ncf(epochs=5, batch_size=4096)

    recommender = NCFRecommender(
        model,
        user_id_to_idx=train_dataset.user_id_to_idx,
        movie_id_to_idx=train_dataset.movie_id_to_idx,
    )

    sample_user = train["user_id"].iloc[0]
    seen = set(train[train["user_id"] == sample_user]["movie_id"])
    recs = recommender.recommend(sample_user, k=10, exclude=seen)
    print(f"Top 10 recommendations for user {sample_user}: {recs}")

    print("\nEvaluating on test set (this may take a few minutes)...")
    results = evaluate_model(recommender, train, test, k=10)
    print(f"NCF — Hit Rate@10: {results['hit_rate@k']:.4f}, NDCG@10: {results['ndcg@k']:.4f}")