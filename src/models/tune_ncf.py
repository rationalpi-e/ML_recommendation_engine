"""Optuna hyperparameter search for NCF, optimizing validation NDCG@10."""
import optuna

from src.data.load_data import load_ratings
from src.data.split import temporal_split
from src.models.train_ncf import train_ncf
from src.models.ncf_recommender import NCFRecommender
from src.eval.metrics import evaluate_model


def objective(trial):
    # Define the search space for each hyperparameter
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    n_negatives = trial.suggest_categorical("n_negatives", [4, 6, 8])

    # NOTE: train_ncf() currently hardcodes dropout inside NCF's default (0.2).
    # We pass dropout through by calling the pieces directly here instead.
    from src.models.ncf import NCF
    from src.models.ncf_dataset import NCFDataset
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings = load_ratings()
    train_df, val_df, test_df = temporal_split(ratings)

    train_dataset = NCFDataset(train_df, n_negatives=n_negatives)
    val_dataset = NCFDataset(
        val_df, n_negatives=n_negatives,
        user_id_to_idx=train_dataset.user_id_to_idx,
        movie_id_to_idx=train_dataset.movie_id_to_idx,
    )

    train_users = train_dataset.users.to(device)
    train_movies = train_dataset.movies.to(device)
    train_labels = train_dataset.labels.to(device)
    n_train = len(train_labels)

    model = NCF(
        n_users=train_dataset.n_users,
        n_movies=train_dataset.n_movies,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    batch_size = 4096
    epochs = 5

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            users, movies, labels = train_users[idx], train_movies[idx], train_labels[idx]

            optimizer.zero_grad()
            predictions = model(users, movies)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

    # Evaluate NDCG@10 on the validation set (leave-one-out style, using val split)
    recommender = NCFRecommender(
        model,
        user_id_to_idx=train_dataset.user_id_to_idx,
        movie_id_to_idx=train_dataset.movie_id_to_idx,
        device=device,
    )
    results = evaluate_model(recommender, train_df, val_df, k=10)

    return results["ndcg@k"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=12)

    print("\nBest trial:")
    print(f"  NDCG@10: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    print("\nAll trials:")
    for t in study.trials:
        print(f"  Trial {t.number}: NDCG@10={t.value:.4f}, params={t.params}")