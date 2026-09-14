"""Popularity baseline: recommend globally most-interacted movies."""
import pandas as pd


class PopularityModel:
    def __init__(self):
        self.ranked_movies = None

    def fit(self, train: pd.DataFrame):
        """Rank movies by number of interactions in train set."""
        self.ranked_movies = (
            train["movie_id"].value_counts().index.tolist()
        )
        return self

    def recommend(self, user_id: int, k: int = 10, exclude: set = None) -> list:
        """Return top-k most popular movies, optionally excluding seen items."""
        exclude = exclude or set()
        recs = [m for m in self.ranked_movies if m not in exclude]
        return recs[:k]


if __name__ == "__main__":
    from src.data.load_data import load_ratings
    from src.data.split import temporal_split

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    model = PopularityModel().fit(train)

    sample_user = train["user_id"].iloc[0]
    seen = set(train[train["user_id"] == sample_user]["movie_id"])
    recs = model.recommend(sample_user, k=10, exclude=seen)

    print(f"Top 10 recommendations for user {sample_user}:")
    print(recs)