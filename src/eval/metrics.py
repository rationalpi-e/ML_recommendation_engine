"""Ranking metrics: NDCG@k, Hit Rate@k."""
import numpy as np


def hit_rate_at_k(recommended: list, relevant: set, k: int = 10) -> int:
    """1 if any relevant item appears in top-k recommendations, else 0."""
    return int(len(set(recommended[:k]) & relevant) > 0)


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    NDCG@k for a single user. Since we typically have exactly one
    relevant (held-out) item in leave-one-out eval, this reduces to
    1 / log2(rank + 2) if the relevant item is in the top-k, else 0.
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i is 0-indexed, rank starts at 1

    # ideal DCG: best case is the relevant item(s) ranked at the top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, train, test, k: int = 10):
    """
    Evaluate a model on the test set (leave-one-out).
    Assumes `model.recommend(user_id, k, exclude)` interface.
    Returns average Hit Rate@k and NDCG@k across all test users.
    """
    hit_rates = []
    ndcgs = []

    for user_id, group in test.groupby("user_id"):
        relevant = set(group["movie_id"])
        seen = set(train[train["user_id"] == user_id]["movie_id"])

        recs = model.recommend(user_id, k=k, exclude=seen)

        hit_rates.append(hit_rate_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        "hit_rate@k": np.mean(hit_rates),
        "ndcg@k": np.mean(ndcgs),
    }


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split
    from src.models.popularity import PopularityModel

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    model = PopularityModel().fit(train)
    results = evaluate_model(model, train, test, k=10)

    print(f"Popularity baseline — Hit Rate@10: {results['hit_rate@k']:.4f}, NDCG@10: {results['ndcg@k']:.4f}")