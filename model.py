"""Walk-forward evaluation + next-day prediction.

Stock direction is close to a coin flip, so the point here isn't to chase a
flashy accuracy number -- it's to (a) use a time-respecting split so we don't
fool ourselves with lookahead leakage, and (b) compare against the naive
"always predict up" baseline, since AAPL has drifted up on ~54% of days over
the last few years.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "sma5_gap", "sma20_gap",
    "macd", "macd_hist", "rsi14", "vol10", "vol20", "volume_chg", "bb_width",
    "spx_ret1", "qqq_ret1", "smh_ret1", "vix_level", "vix_chg",
    "tnx_chg", "dxy_ret1", "excess_vs_qqq",
]


def walk_forward_eval(feat: pd.DataFrame, n_splits: int = 5) -> None:
    labeled = feat.dropna(subset=["target_ret_next", "target_up_next"])
    X = labeled[FEATURE_COLS].values
    y_cls = labeled["target_up_next"].values
    y_reg = labeled["target_ret_next"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, baseline_accs, maes = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        yc_train, yc_test = y_cls[train_idx], y_cls[test_idx]
        yr_train, yr_test = y_reg[train_idx], y_reg[test_idx]

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_train, yc_train)
        pred_cls = clf.predict(X_test)
        acc = accuracy_score(yc_test, pred_cls)

        baseline_pred = np.ones_like(yc_test)  # "always up"
        baseline_acc = accuracy_score(yc_test, baseline_pred)

        reg = GradientBoostingRegressor(random_state=42)
        reg.fit(X_train, yr_train)
        pred_reg = reg.predict(X_test)
        mae = mean_absolute_error(yr_test, pred_reg)

        accs.append(acc)
        baseline_accs.append(baseline_acc)
        maes.append(mae)
        print(f"fold {fold}: model_acc={acc:.3f}  baseline_acc={baseline_acc:.3f}  "
              f"return_mae={mae:.4f} ({mae*100:.2f}%)")

    print(f"\nmean model accuracy:    {np.mean(accs):.3f}")
    print(f"mean baseline accuracy: {np.mean(baseline_accs):.3f}")
    print(f"mean return MAE:        {np.mean(maes):.4f} ({np.mean(maes)*100:.2f}%)")


def predict_next_day(feat: pd.DataFrame) -> dict:
    """Train on all available labeled history, predict the next trading day
    from the most recent row (whose own target is still unknown)."""
    labeled = feat.dropna(subset=["target_ret_next", "target_up_next"])
    X = labeled[FEATURE_COLS].values
    y_cls = labeled["target_up_next"].values
    y_reg = labeled["target_ret_next"].values

    clf = GradientBoostingClassifier(random_state=42).fit(X, y_cls)
    reg = GradientBoostingRegressor(random_state=42).fit(X, y_reg)

    latest = feat[FEATURE_COLS].iloc[[-1]].values
    prob_up = clf.predict_proba(latest)[0, 1]
    expected_ret = reg.predict(latest)[0]

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    top_features = importances.sort_values(ascending=False).head(5)

    return {
        "as_of": feat.index[-1],
        "prob_up": prob_up,
        "expected_return_pct": expected_ret * 100,
        "top_features": top_features,
    }


if __name__ == "__main__":
    from data_fetch import fetch_all
    from features import build_features

    raw = fetch_all(period="5y")
    feat = build_features(raw)

    print("=== Walk-forward backtest (5 folds, time-ordered) ===")
    walk_forward_eval(feat)

    print("\n=== Next trading day prediction ===")
    result = predict_next_day(feat)
    print(f"As of: {result['as_of'].date()}")
    print(f"P(next day up): {result['prob_up']:.1%}")
    print(f"Expected next-day return: {result['expected_return_pct']:+.2f}%")
    print("Top drivers:")
    print(result["top_features"].to_string())
