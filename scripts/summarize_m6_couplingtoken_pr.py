import os
import pandas as pd


ROLLOUT_CSV = "outputs/tables/rollout_m6_couplingtoken_unseen_pr.csv"
PHYSICS_CSV = "outputs/tables/physics_m6_couplingtoken_unseen_pr.csv"
OUT_DIR = "outputs/tables/m6_couplingtoken_summary"
os.makedirs(OUT_DIR, exist_ok=True)


M6 = "M6-Delta-ParamToken-CouplingToken-H4"
PARAMTOKEN = "M5-Delta-ParameterToken-H4"
M5 = "M5-Delta-H4"
FILM_H4 = "M5-Delta-FiLM-H4"


def load_rollout_wide():
    df = pd.read_csv(ROLLOUT_CSV)

    wide = df.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()

    field_order = ["buoyancy", "u_x", "u_y", "pressure", "global"]
    wide = wide[["model", "horizon"] + field_order]
    return wide


def rollout_diff(wide, model_a, model_b):
    """
    model_a - model_b
    Negative means model_a is better.
    """
    a = wide[wide["model"] == model_a].set_index("horizon")
    b = wide[wide["model"] == model_b].set_index("horizon")

    fields = ["buoyancy", "u_x", "u_y", "pressure", "global"]
    diff = a[fields] - b[fields]
    diff = diff.reset_index()
    diff.insert(1, "comparison", f"{model_a} - {model_b}")
    return diff


def load_physics():
    df = pd.read_csv(PHYSICS_CSV)
    return df


def physics_diff(df, model_a, model_b):
    """
    model_a - model_b
    For div_error_mae and vorticity_rel_l2_percent:
    Negative means model_a is better.
    """
    metrics = [
        "div_pred_mae",
        "div_error_mae",
        "vorticity_rel_l2_percent",
        "vorticity_mse",
    ]

    a = df[df["model_name"] == model_a].set_index("horizon")
    b = df[df["model_name"] == model_b].set_index("horizon")

    diff = a[metrics] - b[metrics]
    diff = diff.reset_index()
    diff.insert(1, "comparison", f"{model_a} - {model_b}")
    return diff


def print_block(title, df):
    print("\n" + "=" * 90)
    print(title)
    print("说明：负数表示前者更好；正数表示前者更差。")
    print("=" * 90)
    print(df.to_string(index=False))


def main():
    print("📌 Loading:")
    print(f"  rollout = {ROLLOUT_CSV}")
    print(f"  physics = {PHYSICS_CSV}")

    rollout_wide = load_rollout_wide()
    physics = load_physics()

    rollout_comparisons = [
        (M6, PARAMTOKEN),
        (M6, M5),
        (M6, FILM_H4),
    ]

    physics_comparisons = [
        (M6, PARAMTOKEN),
        (M6, M5),
        (M6, FILM_H4),
    ]

    rollout_all = []
    for a, b in rollout_comparisons:
        d = rollout_diff(rollout_wide, a, b)
        rollout_all.append(d)

        name = f"rollout_diff_{a}_minus_{b}.csv"
        name = name.replace(" ", "_").replace("/", "_")
        path = os.path.join(OUT_DIR, name)
        d.to_csv(path, index=False)

        print_block(f"Rollout Difference: {a} - {b}", d)

    rollout_all = pd.concat(rollout_all, ignore_index=True)
    rollout_all_path = os.path.join(OUT_DIR, "rollout_m6_all_differences_unseen_pr.csv")
    rollout_all.to_csv(rollout_all_path, index=False)

    physics_all = []
    for a, b in physics_comparisons:
        d = physics_diff(physics, a, b)
        physics_all.append(d)

        name = f"physics_diff_{a}_minus_{b}.csv"
        name = name.replace(" ", "_").replace("/", "_")
        path = os.path.join(OUT_DIR, name)
        d.to_csv(path, index=False)

        print_block(f"Physics Difference: {a} - {b}", d)

    physics_all = pd.concat(physics_all, ignore_index=True)
    physics_all_path = os.path.join(OUT_DIR, "physics_m6_all_differences_unseen_pr.csv")
    physics_all.to_csv(physics_all_path, index=False)

    # 简洁结论表：只保留 M6 vs ParamToken
    rollout_m6_vs_param = rollout_diff(rollout_wide, M6, PARAMTOKEN)
    physics_m6_vs_param = physics_diff(physics, M6, PARAMTOKEN)

    rollout_m6_vs_param.to_csv(
        os.path.join(OUT_DIR, "rollout_m6_minus_paramtoken_unseen_pr.csv"),
        index=False,
    )

    physics_m6_vs_param.to_csv(
        os.path.join(OUT_DIR, "physics_m6_minus_paramtoken_unseen_pr.csv"),
        index=False,
    )

    print("\n✅ Summary files saved to:")
    print(f"  {OUT_DIR}")
    print(f"  {rollout_all_path}")
    print(f"  {physics_all_path}")

    print("\n🎯 Key conclusion check:")
    print("Rollout M6 - ParamToken global:")
    print(rollout_m6_vs_param[["horizon", "global"]].to_string(index=False))

    print("\nPhysics M6 - ParamToken:")
    print(physics_m6_vs_param[["horizon", "div_error_mae", "vorticity_rel_l2_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
