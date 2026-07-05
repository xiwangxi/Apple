"""Collects out-of-sample (never-trained-on) predictions across the
TimeSeriesSplit folds used in model.py, so we have an honest day-by-day
"model said X, market did Y" series to plot -- not an in-sample fit that
would flatter the model.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from model import FEATURE_COLS


def collect_oos_predictions(feat: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    labeled = feat.dropna(subset=["target_ret_next", "target_up_next"])
    X = labeled[FEATURE_COLS].values
    y_cls = labeled["target_up_next"].values
    y_reg = labeled["target_ret_next"].values

    # labeled is `feat` with only the last (unlabeled) row removed, so row i
    # of labeled is still row i of feat -- meaning the date the prediction
    # is *for* is feat.index[i + 1], one trading day ahead of the as-of row.
    target_dates = feat.index[1 : len(labeled) + 1]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    for train_idx, test_idx in tscv.split(X):
        clf = GradientBoostingClassifier(random_state=42).fit(X[train_idx], y_cls[train_idx])
        reg = GradientBoostingRegressor(random_state=42).fit(X[train_idx], y_reg[train_idx])
        prob_up = clf.predict_proba(X[test_idx])[:, 1]
        pred_ret = reg.predict(X[test_idx])

        for pos, idx in enumerate(test_idx):
            records.append({
                "date": target_dates[idx],
                "pred_prob_up": prob_up[pos],
                "pred_ret": pred_ret[pos],
                "actual_ret": y_reg[idx],
                "actual_up": y_cls[idx],
            })

    oos = pd.DataFrame(records).set_index("date").sort_index()
    oos["pred_up"] = (oos["pred_prob_up"] > 0.5).astype(int)
    oos["correct"] = (oos["pred_up"] == oos["actual_up"]).astype(int)
    return oos


if __name__ == "__main__":
    from data_fetch import fetch_all
    from features import build_features

    raw = fetch_all(period="5y")
    feat = build_features(raw)
    oos = collect_oos_predictions(feat)
    print(oos.tail())
    print(f"\n{len(oos)} out-of-sample predictions")
    print(f"model accuracy:    {oos['correct'].mean():.3f}")
    print(f"baseline accuracy: {oos['actual_up'].mean():.3f}  (always predict up)")
