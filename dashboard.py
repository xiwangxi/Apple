"""Generates a static HTML dashboard comparing the model's out-of-sample
predictions against what AAPL actually did, plus the live next-day call.

Run: python dashboard.py
Then open aapl_predictor/dashboard.html in a browser.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest import collect_oos_predictions
from data_fetch import fetch_all
from features import build_features
from model import predict_next_day

OUTPUT_FILE = "dashboard.html"


def build_dashboard():
    raw = fetch_all(period="5y")
    feat = build_features(raw)
    oos = collect_oos_predictions(feat)
    latest = predict_next_day(feat)

    oos = oos.join(raw["close"], how="left")

    # Illustrative only: long when the model calls "up", short when it calls
    # "down", no costs/slippage/sizing. This is for comparing signal quality,
    # not a trading recommendation.
    oos["strategy_ret"] = np.where(oos["pred_up"] == 1, oos["actual_ret"], -oos["actual_ret"])
    oos["bh_equity"] = (1 + oos["actual_ret"]).cumprod()
    oos["strategy_equity"] = (1 + oos["strategy_ret"]).cumprod()
    oos["rolling_acc"] = oos["correct"].rolling(20).mean()

    overall_acc = oos["correct"].mean()
    baseline_acc = oos["actual_up"].mean()
    bh_total_ret = oos["bh_equity"].iloc[-1] - 1
    strat_total_ret = oos["strategy_equity"].iloc[-1] - 1

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.3, 0.25],
        vertical_spacing=0.06,
        subplot_titles=(
            "AAPL Close — marker color = model right/wrong on that day's direction",
            "Cumulative return: model long/short signal vs buy-and-hold (illustrative, no costs)",
            "Rolling 20-day directional accuracy vs 50% baseline",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=oos.index, y=oos["close"], mode="lines", name="AAPL close",
            line=dict(color="#8888aa", width=1.5),
        ),
        row=1, col=1,
    )
    correct_mask = oos["correct"] == 1
    fig.add_trace(
        go.Scatter(
            x=oos.index[correct_mask], y=oos["close"][correct_mask], mode="markers",
            name="model correct", marker=dict(color="#2ca02c", size=5, opacity=0.7),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=oos.index[~correct_mask], y=oos["close"][~correct_mask], mode="markers",
            name="model wrong", marker=dict(color="#d62728", size=5, opacity=0.7),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=oos.index, y=oos["bh_equity"], name="buy & hold",
                    line=dict(color="#8888aa", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=oos.index, y=oos["strategy_equity"], name="model long/short",
                    line=dict(color="#1f77b4", width=2)),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=oos.index, y=oos["rolling_acc"], name="rolling accuracy",
                    line=dict(color="#ff7f0e", width=2)),
        row=3, col=1,
    )
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dash"), row=3, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="growth of $1", row=2, col=1)
    fig.update_yaxes(title_text="accuracy", tickformat=".0%", row=3, col=1)

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    direction_word = "上涨 UP" if latest["prob_up"] > 0.5 else "下跌 DOWN"
    top_feat_rows = "".join(
        f"<tr><td>{name}</td><td>{val:.4f}</td></tr>"
        for name, val in latest["top_features"].items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AAPL Prediction vs Market Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #0f1117; color: #e6e6e6; margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #9a9ab0; margin-bottom: 24px; font-size: 13px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: #1a1d29; border: 1px solid #2b2f42; border-radius: 10px; padding: 16px 20px; min-width: 200px; flex: 1; }}
  .card .label {{ color: #9a9ab0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card .value {{ font-size: 28px; font-weight: 600; margin-top: 6px; }}
  .card .sub {{ font-size: 12px; color: #7d8095; margin-top: 4px; }}
  .up {{ color: #2ca02c; }}
  .down {{ color: #d62728; }}
  .warn {{ background: #2a2016; border: 1px solid #7a5a1e; border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; font-size: 13px; color: #f0c674; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
  td {{ padding: 3px 10px 3px 0; color: #cfd2e3; }}
  .chart {{ background: #14161f; border-radius: 10px; padding: 8px; }}
</style>
</head>
<body>
  <h1>AAPL 预测 vs 市场实际走势 Dashboard</h1>
  <div class="subtitle">Out-of-sample walk-forward backtest · {len(oos)} trading days · data as of {oos.index[-1].date()}</div>

  <div class="cards">
    <div class="card">
      <div class="label">明日预测 (as of {latest['as_of'].date()})</div>
      <div class="value {'up' if latest['prob_up'] > 0.5 else 'down'}">{direction_word}</div>
      <div class="sub">P(上涨) = {latest['prob_up']:.1%} · 预期涨跌幅 {latest['expected_return_pct']:+.2f}%</div>
    </div>
    <div class="card">
      <div class="label">模型历史方向准确率 (OOS)</div>
      <div class="value">{overall_acc:.1%}</div>
      <div class="sub">baseline (无脑猜涨) = {baseline_acc:.1%}</div>
    </div>
    <div class="card">
      <div class="label">累计收益: 模型信号 vs 买入持有</div>
      <div class="value {'up' if strat_total_ret > bh_total_ret else 'down'}">{strat_total_ret:+.1%}</div>
      <div class="sub">buy & hold: {bh_total_ret:+.1%} (无手续费/滑点，仅示意)</div>
    </div>
  </div>

  <div class="warn">
    ⚠ 诚实提示：回测显示模型方向准确率（{overall_acc:.1%}）{'低于' if overall_acc < baseline_acc else '略高于'}"无脑猜涨"基线（{baseline_acc:.1%}）。
    这说明日频方向预测信噪比很低，本 dashboard 用于展示预测过程和方法论，<b>不构成投资建议</b>。
  </div>

  <div class="chart">{chart_html}</div>

  <div class="card" style="margin-top: 24px; max-width: 420px;">
    <div class="label">本次预测 Top 5 驱动因子</div>
    <table><tr><td><b>feature</b></td><td><b>importance</b></td></tr>{top_feat_rows}</table>
  </div>
</body>
</html>
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUTPUT_FILE} ({len(html)} bytes)")
    print(f"Latest prediction: prob_up={latest['prob_up']:.1%}, "
          f"expected_return={latest['expected_return_pct']:+.2f}%")
    print(f"OOS accuracy: {overall_acc:.1%} vs baseline {baseline_acc:.1%}")


if __name__ == "__main__":
    build_dashboard()
