"""Diversity metrics: catalog coverage and intra-list diversity."""
import numpy as np
import pandas as pd


def catalog_coverage(all_recommendations: list, total_catalog_size: int) -> float:
    """
    What fraction of the entire catalog gets recommended to *anyone*,
    across all users combined.
    `all_recommendations` is a list of lists (one list of movie_ids per user).
    """
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)

    return len(unique_recommended) / total_catalog_size


def intra_list_diversity(recs: list, movie_genres: dict) -> float:
    """
    Average pairwise dissimilarity between movies within a single
    recommendation list, based on genre overlap.
    `movie_genres` maps movie_id -> set of genre strings.
    Returns a value between 0 (all identical genres) and 1 (no genre overlap at all).
    """
    if len(recs) < 2:
        return 0.0

    dissimilarities = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            genres_i = movie_genres.get(recs[i], set())
            genres_j = movie_genres.get(recs[j], set())

            if not genres_i or not genres_j:
                continue  # skip if we don't have genre info

            # Jaccard similarity: overlap / union
            overlap = len(genres_i & genres_j)
            union = len(genres_i | genres_j)
            similarity = overlap / union if union > 0 else 0
            dissimilarities.append(1 - similarity)

    return np.mean(dissimilarities) if dissimilarities else 0.0


def evaluate_diversity(model, train: pd.DataFrame, movies: pd.DataFrame, k: int = 10):
    """
    Compute catalog coverage and average intra-list diversity for a model
    across all users in the training set.
    """
    # Build movie_id -> set of genres lookup
    movie_genres = {
        row.movie_id: set(row.genres.split("|"))
        for row in movies.itertuples()
    }

    total_catalog_size = movies["movie_id"].nunique()

    all_recommendations = []
    ild_scores = []

    for user_id in train["user_id"].unique():
        seen = set(train[train["user_id"] == user_id]["movie_id"])
        recs = model.recommend(user_id, k=k, exclude=seen)
        recs = [int(r) for r in recs]  # cast off any np.int64

        all_recommendations.append(recs)
        ild_scores.append(intra_list_diversity(recs, movie_genres))

    coverage = catalog_coverage(all_recommendations, total_catalog_size)
    avg_ild = np.mean(ild_scores)

    return {
        "catalog_coverage": coverage,
        "avg_intra_list_diversity": avg_ild,
    }


if __name__ == "__main__":
    from src.data.load_data import load_ratings, load_movies
    from src.data.split import temporal_split
    from src.models.popularity import PopularityModel
    from src.models.user_cf import UserCFModel

    ratings = load_ratings()
    movies = load_movies()
    train, val, test = temporal_split(ratings)

    print("Evaluating Popularity baseline diversity...")
    pop_model = PopularityModel().fit(train)
    pop_diversity = evaluate_diversity(pop_model, train, movies, k=10)
    print(f"Popularity — Coverage: {pop_diversity['catalog_coverage']:.4f}, "
          f"Intra-list Diversity: {pop_diversity['avg_intra_list_diversity']:.4f}")

    print("\nEvaluating User-based CF diversity (this may take a few minutes)...")
    cf_model = UserCFModel(n_neighbors=20).fit(train)
    cf_diversity = evaluate_diversity(cf_model, train, movies, k=10)
    print(f"User-based CF — Coverage: {cf_diversity['catalog_coverage']:.4f}, "
          f"Intra-list Diversity: {cf_diversity['avg_intra_list_diversity']:.4f}")