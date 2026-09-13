'''Load 1M MovieLens data file in pandas data fram'''
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2]/'data'/'raw'/'ml-1m'

def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "ratings.dat",sep="::",engine="python",names=["user_id", "movie_id", "rating", "timestamp"],
                     encoding="latin-1",)
    return df

def load_movies() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "movies.dat",sep="::",engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


def load_users() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "users.dat",sep="::",engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    return df


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    print("Ratings:", ratings.shape)
    print(ratings.head())
    print("\nMovies:", movies.shape)
    print(movies.head())
    print("\nUsers:", users.shape)
    print(users.head())