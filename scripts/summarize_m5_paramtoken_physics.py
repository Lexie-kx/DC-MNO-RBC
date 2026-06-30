import os
import argparse
import pandas as pd


METRICS = [
    "div_pred_mae",
    "div_error_mae",
    "vorticity_rel_l2_percent",
]

MODEL_COL_CANDIDATES = ["model", "model_name"]
HORIZON_COL = "horizon"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean summary for M5-Delta-ParameterToken-H4 physics diagnostics."
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    parser.add_argument("--tag", type=str, required=True, help="e.g. unseen_pr or unseen_ra")
    return parser.parse_args()


def get_model_col(df):
    for col in MODEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find model column. Existing columns: {list(df.columns)}")


def print_table(title, table):
    print("\n" + "=" * 18 + f" {title} " + "=" * 18)
    print(table.to_string(index=False))


def make_metric_table(df, model_col, metric):
    table = df.pivot_table(
        index=model_col,
        columns=HORIZON_COL,
        values=metric,
    ).reset_index()

    table.columns = [
        "model" if c == model_col else f"h={c}" for c in table.columns
    ]

    return table


def make_diff(df, model_col, model_a, model_b, out_path):
    """
    diff = model_a - model_b

    对 div_pred_mae / div_error_mae / vorticity_rel_l2_percent：
    负数表示 model_a 更好；
    正数表示 model_a 更差。
    """
    a = df[df[model_col] == model_a].set_index(HORIZON_COL)
    b = df[df[model_col] == model_b].set_index(HORIZON_COL)

    missing = []
    if a.empty:
        missing.append(model_a)
    if b.empty:
        missing.append(model_b)

    if missing:
        print(f"⚠️ 跳过差值表，缺少模型: {missing}")
        return

    diff = a[METRICS] - b[METRICS]
    diff = diff.reset_index()

    print_table(f"{model_a} - {model_b}: Physics diagnostics", diff)
    print("说明：负数表示前者更好；正数表示前者更差。")

    diff.to_csv(out_path, index=False)
    print(f"✅ saved: {out_path}")


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    model_col = get_model_col(df)

    # 兼容不同列名：有些 summary 打印叫 div_err_mae，csv 里可能叫 div_error_mae
    if "div_err_mae" in df.columns and "div_error_mae" not in df.columns:
        df = df.rename(columns={"div_err_mae": "div_error_mae"})

    if "vort_rel_l2%" in df.columns and "vorticity_rel_l2_percent" not in df.columns:
        df = df.rename(columns={"vort_rel_l2%": "vorticity_rel_l2_percent"})

    required = [model_col, HORIZON_COL] + METRICS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}. Existing columns: {list(df.columns)}")

    model_order = [
        "M0",
        "M3-Delta",
        "M3-Delta-FiLM",
        "M5-Delta-H4",
        "M5-Delta-FiLM-H4",
        "M5-Delta-ParameterToken-H4",
    ]

    df[model_col] = pd.Categorical(df[model_col], categories=model_order, ordered=True)
    df = df.sort_values([model_col, HORIZON_COL])

    # 1. 原始指标表
    for metric in METRICS:
        table = make_metric_table(df, model_col, metric)
        print_table(f"{metric} by Horizon", table)

    # 2. 差值表
    make_diff(
        df,
        model_col,
        "M5-Delta-ParameterToken-H4",
        "M5-Delta-H4",
        os.path.join(args.out_dir, f"physics_{args.tag}_paramtoken___m5_delta_h4.csv"),
    )

    make_diff(
        df,
        model_col,
        "M5-Delta-ParameterToken-H4",
        "M5-Delta-FiLM-H4",
        os.path.join(args.out_dir, f"physics_{args.tag}_paramtoken___m5_delta_film_h4.csv"),
    )

    make_diff(
        df,
        model_col,
        "M5-Delta-ParameterToken-H4",
        "M3-Delta",
        os.path.join(args.out_dir, f"physics_{args.tag}_paramtoken___m3_delta.csv"),
    )


if __name__ == "__main__":
    main()
