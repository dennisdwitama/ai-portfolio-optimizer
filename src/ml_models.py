from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TECHNICAL_FEATURES


def cluster_stocks(returns: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol.replace(0, np.nan)
    downside = returns.clip(upper=0).std() * np.sqrt(252)

    features = pd.DataFrame(
        {
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe.fillna(0),
            "downside_vol": downside.fillna(0),
        }
    ).dropna()
    if features.empty:
        raise ValueError("Not enough clean return history to cluster the selected assets.")

    cluster_count = min(n_clusters, len(features))
    if cluster_count < 2:
        features["cluster"] = 0
        features["risk_group"] = "Low"
        return features.sort_values(["risk_group", "annual_return"], ascending=[True, False])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=cluster_count, random_state=42, n_init=20)),
        ]
    )
    labels = model.fit_predict(features)
    features["cluster"] = labels

    order = features.groupby("cluster")["annual_vol"].mean().sort_values().index.tolist()
    risk_labels = ["Low", "Medium", "High"][:cluster_count]
    risk_map = {cluster: risk for cluster, risk in zip(order, risk_labels)}
    features["risk_group"] = features["cluster"].map(risk_map)
    return features.sort_values(["risk_group", "annual_return"], ascending=[True, False])


def prepare_ml_dataset(feature_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, df in feature_panel.items():
        frame = df.copy()
        frame["ticker"] = ticker
        frames.append(frame)
    dataset = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    dataset = dataset.dropna(subset=TECHNICAL_FEATURES + ["target_next_return", "target_up"])
    return dataset


def train_random_forest_models(dataset: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    x = dataset[TECHNICAL_FEATURES]
    y_reg = dataset["target_next_return"]
    y_clf = dataset["target_up"]

    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    classifier = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    regressor.fit(x, y_reg)
    classifier.fit(x, y_clf)
    return regressor, classifier


def predict_next_day_scores(
    latest_feature_rows: pd.DataFrame,
    regressor: Pipeline,
    classifier: Pipeline,
) -> pd.DataFrame:
    x_latest = latest_feature_rows[TECHNICAL_FEATURES]
    predictions = latest_feature_rows[["ticker"]].copy()
    predictions["predicted_return"] = regressor.predict(x_latest)
    predictions["prob_up"] = classifier.predict_proba(x_latest)[:, 1]
    predictions["ai_score"] = 0.5 * predictions["predicted_return"].rank(pct=True) + 0.5 * predictions["prob_up"]
    return predictions.sort_values("ai_score", ascending=False)
