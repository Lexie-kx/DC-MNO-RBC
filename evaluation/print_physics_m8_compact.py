import argparse
import pandas as pd


MODEL_ORDER = [
    "M6-FieldWiseEncoder-H4-continue",
    "M7-FieldCoupling-H4",
    "M7-ParamTokenOnly-H4",
    "M8-A-FullStatic-H4",
    "M8-B-ParamConditionedCoupling-H4",
]

COMPARISONS = [
    (
        "M7-FieldCoupling-H4",
        "M6-FieldWiseEncoder-H4-continue",
        "M7-FieldCoupling - M6",
    ),
    (
        "M7-ParamTokenOnly-H4",
        "M6-FieldWiseEncoder-H4-continue",
        "M7-ParamToken - M6",
    ),
    (
        "M8-A-FullStatic-H4",
        "M6-FieldWiseEncoder-H4-continue",
        "M8-A - M6",
    ),
    (
        "M8-B-ParamConditionedCoupling-H4",
        "M6-FieldWiseEncoder-H4-continue",
        "M8-B - M6",
    ),
    (
        "M8-A-FullStatic-H4",
        "M7-FieldCoupling-H4",
        "M8-A - M7-FieldCoupling",
    ),
    (
        "M8-A-FullStatic-H4",
        "M7-ParamTokenOnly-H4",
        "M8-A - M7-ParamToken",
    ),
    (
        "M8-B-ParamConditionedCoupling-H4",
        "M7-FieldCoupling-H4",
        "M8-B - M7-FieldCoupling",
    ),
    (
        "M8-B-ParamConditionedCoupling-H4",
        "M7-ParamTokenOnly-H4",
        "M8-B - M7-ParamToken",
    ),
    (
        "M8-B-ParamConditionedCoupling-H4",
        "M8-A-FullStatic-H4",
        "M8-B - M8-A",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def build_difference_table(df, metric):
    row_map = {
        (row.model_name, int(row.horizon)): row
        for row in df.itertuples(index=False)
    }

    horizons = sorted(df["horizon"].unique())

    rows = []

    for model_a, model_b, name in COMPARISONS:
        row = {"comparison": name}

        for horizon in horizons:
            a = row_map[(model_a, int(horizon))]
            b = row_map[(model_b, int(horizon))]

            row[f"h={int(horizon)}"] = (
                getattr(a, metric)
                - getattr(b, metric)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    missing = [
        model
        for model in MODEL_ORDER
        if model not in set(df["model_name"])
    ]

    if missing:
        raise RuntimeError(
            f"缺少模型结果: {missing}"
        )

    df["model_name"] = pd.Categorical(
        df["model_name"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        ["model_name", "horizon"]
    )

    print()
    print("=" * 100)
    print("M6–M8 PHYSICS SUMMARY")
    print("=" * 100)

    summary = df[
        [
            "model_name",
            "horizon",
            "div_error_mae",
            "vorticity_rel_l2_percent",
        ]
    ].copy()

    summary.columns = [
        "model",
        "h",
        "div_err_mae",
        "vort_rel_l2%",
    ]

    print(
        summary.to_string(
            index=False,
            formatters={
                "div_err_mae": lambda x: f"{x:.6e}",
                "vort_rel_l2%": lambda x: f"{x:.6f}",
            },
        )
    )

    div_difference = build_difference_table(
        df,
        "div_error_mae",
    )

    print()
    print("=" * 100)
    print(
        "DIV ERROR MAE DIFFERENCE"
    )
    print(
        "前者 - 后者；负数表示前者误差更低"
    )
    print("=" * 100)

    print(
        div_difference.to_string(
            index=False,
            formatters={
                column: (
                    lambda x: f"{x:+.6e}"
                )
                for column in div_difference.columns
                if column != "comparison"
            },
        )
    )

    vort_difference = build_difference_table(
        df,
        "vorticity_rel_l2_percent",
    )

    print()
    print("=" * 100)
    print(
        "VORTICITY REL-L2 (%) DIFFERENCE"
    )
    print(
        "前者 - 后者；负数表示前者误差更低"
    )
    print("=" * 100)

    print(
        vort_difference.to_string(
            index=False,
            formatters={
                column: (
                    lambda x: f"{x:+.6f}"
                )
                for column in vort_difference.columns
                if column != "comparison"
            },
        )
    )

    print()
    print("=" * 100)
    print(
        "完整 physics 指标保存在原 CSV；"
        "终端仅显示当前阶段核心结果。"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
