"""Temporal train/val/test split — last interactions per user held out."""
import pandas as pd
from src.data.load_data import load_ratings


def temporal_split(ratings: pd.DataFrame, val_n: int = 1, test_n: int = 1):
    """
    For each user, sort by timestamp and hold out the last `test_n`
    interactions as test, the `val_n` before that as validation,
    and everything else as train.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])
    ratings["rank"] = ratings.groupby("user_id").cumcount(ascending=False)
    # rank 0 = most recent interaction for that user

    test = ratings[ratings["rank"] < test_n]
    val = ratings[(ratings["rank"] >= test_n) & (ratings["rank"] < test_n + val_n)]
    train = ratings[ratings["rank"] >= test_n + val_n]

    return (
        train.drop(columns="rank"),
        val.drop(columns="rank"),
        test.drop(columns="rank"),
    )


if __name__ == "__main__":
    from pathlib import Path

    ratings = load_ratings()
    train, val, test = temporal_split(ratings)

    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")
    print(f"Total: {train.shape[0] + val.shape[0] + test.shape[0]} (should equal {len(ratings)})")

    train_ids = set(zip(train.user_id, train.movie_id, train.timestamp))
    test_ids = set(zip(test.user_id, test.movie_id, test.timestamp))
    print(f"Overlap between train/test: {len(train_ids & test_ids)} (should be 0)")

    out_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"\nSaved train/val/test CSVs to {out_dir}")