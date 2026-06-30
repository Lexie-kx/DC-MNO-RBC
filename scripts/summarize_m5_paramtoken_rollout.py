import os
import argparse
import pandas as pd


FIELDS = ["buoyancy", "u_x", "u_y", "pressure", "global"]

MODELS = [
    "M0",
    "M3-Delta",
    "M3-Delta-FiLM",
    "M5-Delta-H4",
    "M5-Delta-FiLM-H4",
    "M5-Delta-ParameterToken-H4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean summary for M5-Delta-ParameterToken-H4 rollout results."
    )
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/tables")
    parser.add_argument("--tag", type=str, required=True, help="e.g. unseen_pr or unseen_ra")
    return parser.parse_args()


def print_table(title, table):
    print("\n" + "=" * 18 + f" {title} " + "=" * 18)
    print(table.to_string(index=False))


def make_diff(wide, model_a, model_b, out_path):
    """
    diff = model_a - model_b
    负数表示 model_a 更好；正数表示 model_a 更差。
    """
    a = wide[wide["model"] == model_a].set_index("horizon")[FIELDS]
    b = wide[wide["model"] == model_b].set_index("horizon")[FIELDS]

    missing = []
    if a.empty:
        missing.append(model_a)
    if b.empty:
        missing.append(model_b)
    if missing:
        print(f"⚠️ 跳过差值表，缺少模型: {missing}")
        return None

    diff = (a - b).reset_index()

    print_table(f"{model_a} - {model_b}: Rel-L2 (%)", diff)
    print("说明：负数表示前者更好；正数表示前者更差。")

    diff.to_csv(out_path, index=False)
    print(f"✅ saved: {out_path}")

    return diff


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    wide = df.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    existing_models = [m for m in MODELS if m in set(wide["model"])]
    wide = wide[wide["model"].isin(existing_models)]
    wide = wide[["model", "horizon"] + FIELDS]

    # 1. global summary
    global_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="global",
    ).reset_index()
    global_table.columns = [
        "model" if c == "model" else f"h={c}" for c in global_table.columns
    ]
    print_table("Global Rel-L2 (%) by Horizon", global_table)

    # 2. u_x summary
    ux_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="u_x",
    ).reset_index()
    ux_table.columns = [
        "model" if c == "model" else f"h={c}" for c in ux_table.columns
    ]
    print_table("u_x Rel-L2 (%) by Horizon", ux_table)

    # 3. u_y summary
    uy_table = wide.pivot_table(
        index="model",
        columns="horizon",
        values="u_y",
    ).reset_index()
    uy_table.columns = [
        "model" if c == "model" else f"h={c}" for c in uy_table.columns
    ]
    print_table("u_y Rel-L2 (%) by Horizon", uy_table)

    # 4. clean diff tables
    make_diff(
        wide,
        "M5-Delta-ParameterToken-H4",
        "M5-Delta-H4",
        os.path.join(args.out_dir, f"rollout_{args.tag}_paramtoken___m5_delta_h4.csv"),
    )

    make_diff(
        wide,
        "M5-Delta-ParameterToken-H4",
        "M5-Delta-FiLM-H4",
        os.path.join(args.out_dir, f"rollout_{args.tag}_paramtoken___m5_delta_film_h4.csv"),
    )

    make_diff(
        wide,
        "M5-Delta-ParameterToken-H4",
        "M3-Delta",
        os.path.join(args.out_dir, f"rollout_{args.tag}_paramtoken___m3_delta.csv"),
    )


if __name__ == "__main__":
    main()
