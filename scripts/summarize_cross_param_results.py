import os
import pandas as pd


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    table_dir = os.path.join(project_root, "outputs", "tables")

    files = [
        # ================= unseen Ra =================
        {
            "split": "unseen_ra",
            "model": "M0",
            "path": os.path.join(table_dir, "m0_unseen_ra_eval.csv"),
        },
        {
            "split": "unseen_ra",
            "model": "M3-State",
            "path": os.path.join(table_dir, "m3_state_unseen_ra_eval.csv"),
        },
        {
            "split": "unseen_ra",
            "model": "M3-Delta",
            "path": os.path.join(table_dir, "m3_delta_unseen_ra_eval.csv"),
        },
        {
            "split": "unseen_ra",
            "model": "M3-Delta-ParamConcat",
            "path": os.path.join(table_dir, "m3_delta_paramconcat_unseen_ra_eval.csv"),
        },
        {
            "split": "unseen_ra",
            "model": "M3-Delta-FiLM",
            "path": os.path.join(table_dir, "m3_delta_film_unseen_ra_eval.csv"),
        },

        # ================= unseen Pr =================
        {
            "split": "unseen_pr",
            "model": "M0",
            "path": os.path.join(table_dir, "m0_unseen_pr_eval.csv"),
        },
        {
            "split": "unseen_pr",
            "model": "M3-State",
            "path": os.path.join(table_dir, "m3_state_unseen_pr_eval.csv"),
        },
        {
            "split": "unseen_pr",
            "model": "M3-Delta",
            "path": os.path.join(table_dir, "m3_delta_unseen_pr_eval.csv"),
        },
        {
            "split": "unseen_pr",
            "model": "M3-Delta-ParamConcat",
            "path": os.path.join(table_dir, "m3_delta_paramconcat_unseen_pr_eval.csv"),
        },
        {
            "split": "unseen_pr",
            "model": "M3-Delta-FiLM",
            "path": os.path.join(table_dir, "m3_delta_film_unseen_pr_eval.csv"),
        },
    ]

    all_rows = []

    for item in files:
        if not os.path.exists(item["path"]):
            raise FileNotFoundError(f"Missing file: {item['path']}")

        df = pd.read_csv(item["path"])
        df["split_name"] = item["split"]
        df["model_name"] = item["model"]

        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    # 只保留核心列
    full_df = full_df[
        [
            "split_name",
            "model_name",
            "field",
            "rel_l2_percent",
            "mse",
        ]
    ]

    summary_long_path = os.path.join(table_dir, "cross_param_summary_long.csv")
    full_df.to_csv(summary_long_path, index=False)

    # 生成宽表：每个 field 一列
    summary_wide = full_df.pivot_table(
        index=["split_name", "model_name"],
        columns="field",
        values="rel_l2_percent"
    ).reset_index()

    field_order = ["buoyancy", "u_x", "u_y", "pressure", "global"]
    summary_wide = summary_wide[["split_name", "model_name"] + field_order]

    # 固定显示顺序
    model_order = {
        "M0": 0,
        "M3-State": 1,
        "M3-Delta": 2,
        "M3-Delta-ParamConcat": 3,
        "M3-Delta-FiLM": 4,
    }

    split_order = {
        "unseen_ra": 0,
        "unseen_pr": 1,
    }

    summary_wide["split_order"] = summary_wide["split_name"].map(split_order)
    summary_wide["model_order"] = summary_wide["model_name"].map(model_order)

    summary_wide = summary_wide.sort_values(
        by=["split_order", "model_order"]
    ).drop(columns=["split_order", "model_order"])

    summary_wide_path = os.path.join(table_dir, "cross_param_summary_wide.csv")
    summary_wide.to_csv(summary_wide_path, index=False)

    # 计算相对 M0 的 improvement
    improvement_rows = []

    compare_models = [
        "M3-State",
        "M3-Delta",
        "M3-Delta-ParamConcat",
        "M3-Delta-FiLM",
    ]

    for split_name in ["unseen_ra", "unseen_pr"]:
        split_df = summary_wide[summary_wide["split_name"] == split_name]

        m0_row = split_df[split_df["model_name"] == "M0"].iloc[0]

        for model_name in compare_models:
            model_row = split_df[split_df["model_name"] == model_name].iloc[0]

            row = {
                "split_name": split_name,
                "model_name": model_name,
            }

            for field in field_order:
                m0_error = m0_row[field]
                model_error = model_row[field]

                improvement = 1.0 - model_error / m0_error
                row[f"{field}_improvement_percent"] = improvement * 100.0

            improvement_rows.append(row)

    improvement_df = pd.DataFrame(improvement_rows)

    improvement_path = os.path.join(table_dir, "cross_param_improvement_over_m0.csv")
    improvement_df.to_csv(improvement_path, index=False)

    print("\n✅ Cross-parameter summary saved:")
    print(f"  {summary_long_path}")
    print(f"  {summary_wide_path}")
    print(f"  {improvement_path}")

    print("\n================ Cross-Parameter Summary: Rel-L2 (%) ================")
    print(summary_wide.to_string(index=False))

    print("\n================ Improvement over M0 (%) ================")
    print(improvement_df.to_string(index=False))

    print("\n================ Quick Conclusion: Compared with M3-Delta ================")

    for split_name in ["unseen_ra", "unseen_pr"]:
        split_df = summary_wide[summary_wide["split_name"] == split_name]

        m3_delta = split_df[split_df["model_name"] == "M3-Delta"].iloc[0]

        print(f"\n[{split_name}]")

        for compare_model in ["M3-Delta-ParamConcat", "M3-Delta-FiLM"]:
            compare_row = split_df[split_df["model_name"] == compare_model].iloc[0]

            print(f"\n{compare_model} vs M3-Delta")

            for field in field_order:
                base_error = m3_delta[field]
                new_error = compare_row[field]
                diff = new_error - base_error

                if diff < 0:
                    trend = "improved"
                elif diff > 0:
                    trend = "worse"
                else:
                    trend = "same"

                print(
                    f"{field:<10}: "
                    f"M3-Delta={base_error:.4f}, "
                    f"{compare_model}={new_error:.4f}, "
                    f"diff={diff:+.4f} ({trend})"
                )


if __name__ == "__main__":
    main()