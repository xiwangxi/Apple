"""One-command entry point: fetch data -> build features -> backtest -> predict tomorrow."""
from data_fetch import fetch_all
from features import build_features
from model import predict_next_day, walk_forward_eval


def main():
    print("Fetching AAPL + macro data (5y history)...")
    raw = fetch_all(period="5y")

    print("Building features...")
    feat = build_features(raw)

    print(f"\n=== Walk-forward backtest ({len(feat)} trading days) ===")
    walk_forward_eval(feat)

    print("\n=== Next trading day prediction ===")
    result = predict_next_day(feat)
    print(f"As of: {result['as_of'].date()}")
    print(f"P(next day up): {result['prob_up']:.1%}")
    print(f"Expected next-day return: {result['expected_return_pct']:+.2f}%")
    print("Top drivers:")
    print(result["top_features"].to_string())


if __name__ == "__main__":
    main()
