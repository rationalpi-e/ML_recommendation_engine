"""One-off script: evaluate ALS diversity metrics."""
from src.data.load_data import load_ratings, load_movies
from src.data.split import temporal_split
from src.models.als import ALSModel
from src.eval.diversity import evaluate_diversity

if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    train, val, test = temporal_split(ratings)

    model = ALSModel(factors=50, regularization=0.01, iterations=15).fit(train)
    diversity = evaluate_diversity(model, train, movies, k=10)

    print(f"ALS — Coverage: {diversity['catalog_coverage']:.4f}, "
          f"Intra-list Diversity: {diversity['avg_intra_list_diversity']:.4f}")