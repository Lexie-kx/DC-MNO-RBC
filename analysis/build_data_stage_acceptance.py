from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/tables/data_stage_acceptance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 4, 8, 16]
FIELDS = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
    "global",
]
PHYSICAL_FIELDS = [
    "buoyancy",
    "u_x",
    "u_y",
    "pressure",
]


def load_rollout(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"❌ 文件不存在：{path}")

    df = pd.read_csv(path)

    required = {
        "model",
        "horizon",
        "field",
        "rel_l2_percent",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"❌ {path.name} 缺少列：{sorted(missing)}"
        )

    df = df.copy()
    df["horizon"] = df["horizon"].astype(int)

    return df


def load_physics(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"❌ 文件不存在：{path}")

    df = pd.read_csv(path)

    required = {
        "model_name",
        "horizon",
        "div_pred_mae",
        "vorticity_rel_l2_percent",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"❌ {path.name} 缺少列：{sorted(missing)}"
        )

    df = df.copy()
    df["horizon"] = df["horizon"].astype(int)

    return df


def global_table(
    df: pd.DataFrame,
    split_name: str,
    population: str,
    model_order: list[str],
) -> pd.DataFrame:
    sub = df[
        (df["field"] == "global")
        & (df["model"].isin(model_order))
    ].copy()

    pivot = sub.pivot(
        index="model",
        columns="horizon",
        values="rel_l2_percent",
    )

    pivot = pivot.reindex(model_order)
    pivot = pivot.reindex(columns=HORIZONS)

    pivot.columns = [
        f"h{horizon}"
        for horizon in pivot.columns
    ]

    pivot = pivot.reset_index()
    pivot.insert(0, "population", population)
    pivot.insert(0, "split", split_name)

    return pivot


def pairwise_rollout(
    df: pd.DataFrame,
    split_name: str,
    population: str,
    candidate: str,
    baseline: str,
) -> tuple[dict, pd.DataFrame]:
    available = set(df["model"].unique())

    for model in [candidate, baseline]:
        if model not in available:
            raise RuntimeError(
                f"❌ 缺少模型 {model}；"
                f"现有模型={sorted(available)}"
            )

    candidate_wide = (
        df[df["model"] == candidate]
        .pivot(
            index="horizon",
            columns="field",
            values="rel_l2_percent",
        )
        .reindex(index=HORIZONS, columns=FIELDS)
    )

    baseline_wide = (
        df[df["model"] == baseline]
        .pivot(
            index="horizon",
            columns="field",
            values="rel_l2_percent",
        )
        .reindex(index=HORIZONS, columns=FIELDS)
    )

    difference = candidate_wide - baseline_wide

    all_values = difference[FIELDS].to_numpy()
    physical_values = difference[
        PHYSICAL_FIELDS
    ].to_numpy()
    global_values = difference["global"].to_numpy()

    summary = {
        "split": split_name,
        "population": population,
        "comparison": f"{candidate} - {baseline}",
        "candidate": candidate,
        "baseline": baseline,
        "physical_field_pass": int(
            (physical_values < 0).sum()
        ),
        "physical_field_total": int(
            physical_values.size
        ),
        "all_metric_pass": int(
            (all_values < 0).sum()
        ),
        "all_metric_total": int(
            all_values.size
        ),
        "global_pass": int(
            (global_values < 0).sum()
        ),
        "global_total": int(
            global_values.size
        ),
        "global_h1_delta": float(
            difference.loc[1, "global"]
        ),
        "global_h4_delta": float(
            difference.loc[4, "global"]
        ),
        "global_h8_delta": float(
            difference.loc[8, "global"]
        ),
        "global_h16_delta": float(
            difference.loc[16, "global"]
        ),
        "mean_global_delta": float(
            difference["global"].mean()
        ),
        "worst_delta": float(
            difference.to_numpy().max()
        ),
        "best_delta": float(
            difference.to_numpy().min()
        ),
    }

    detail = difference.reset_index()
    detail.insert(0, "baseline", baseline)
    detail.insert(0, "candidate", candidate)
    detail.insert(0, "population", population)
    detail.insert(0, "split", split_name)

    return summary, detail


def resolve_physics_model(
    df: pd.DataFrame,
    label: str,
) -> str:
    values = df["model_name"].dropna().unique().tolist()

    rules = {
        "M6": lambda value: (
            "M6" in value
            and "FieldWise" in value
        ),
        "M7-Field": lambda value: (
            "M7" in value
            and "FieldCoupling" in value
        ),
        "M7-Param": lambda value: (
            "M7" in value
            and "ParamToken" in value
        ),
        "M8-A": lambda value: "M8-A" in value,
        "M8-B": lambda value: "M8-B" in value,
    }

    if label not in rules:
        raise KeyError(label)

    matches = [
        value
        for value in values
        if rules[label](value)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"❌ physics模型解析失败："
            f"label={label}, matches={matches}, "
            f"available={values}"
        )

    return matches[0]


def compact_physics(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    keep = [
        "model_name",
        "horizon",
        "div_pred_mae",
        "vorticity_rel_l2_percent",
    ]

    result = df[keep].copy()
    result.insert(0, "split", split_name)

    return result.sort_values(
        ["model_name", "horizon"]
    )


def pairwise_physics(
    df: pd.DataFrame,
    split_name: str,
    candidate: str,
    baseline: str,
) -> tuple[dict, pd.DataFrame]:
    metrics = [
        "div_pred_mae",
        "vorticity_rel_l2_percent",
    ]

    candidate_wide = (
        df[df["model_name"] == candidate]
        .set_index("horizon")[metrics]
        .reindex(HORIZONS)
    )

    baseline_wide = (
        df[df["model_name"] == baseline]
        .set_index("horizon")[metrics]
        .reindex(HORIZONS)
    )

    difference = candidate_wide - baseline_wide

    summary = {
        "split": split_name,
        "comparison": f"{candidate} - {baseline}",
        "candidate": candidate,
        "baseline": baseline,
        "div_pass": int(
            (difference["div_pred_mae"] < 0).sum()
        ),
        "div_total": 4,
        "vorticity_pass": int(
            (
                difference[
                    "vorticity_rel_l2_percent"
                ] < 0
            ).sum()
        ),
        "vorticity_total": 4,
        "div_h1_delta": float(
            difference.loc[1, "div_pred_mae"]
        ),
        "div_h4_delta": float(
            difference.loc[4, "div_pred_mae"]
        ),
        "div_h8_delta": float(
            difference.loc[8, "div_pred_mae"]
        ),
        "div_h16_delta": float(
            difference.loc[16, "div_pred_mae"]
        ),
        "vort_h1_delta": float(
            difference.loc[
                1,
                "vorticity_rel_l2_percent",
            ]
        ),
        "vort_h4_delta": float(
            difference.loc[
                4,
                "vorticity_rel_l2_percent",
            ]
        ),
        "vort_h8_delta": float(
            difference.loc[
                8,
                "vorticity_rel_l2_percent",
            ]
        ),
        "vort_h16_delta": float(
            difference.loc[
                16,
                "vorticity_rel_l2_percent",
            ]
        ),
    }

    detail = difference.reset_index()
    detail.insert(0, "baseline", baseline)
    detail.insert(0, "candidate", candidate)
    detail.insert(0, "split", split_name)

    return summary, detail


def print_section(
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    print(
        dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main() -> None:
    # -------------------------------------------------
    # 1. M5 → M6 → M7 正式主链
    # -------------------------------------------------
    main_chain_files = {
        "unseen_pr": (
            "outputs/tables/"
            "rollout_m7_paramtoken_unseen_pr.csv"
        ),
        "unseen_ra": (
            "outputs/tables/"
            "rollout_m7_paramtoken_unseen_ra.csv"
        ),
    }

    main_models = [
        "M5-Delta-H4",
        "M5-Delta-ParameterToken-H4",
        "M6-FieldWiseEncoder-H4-continue",
        "M7-FieldCoupling-H4",
        "M7-ParamTokenOnly-H4",
    ]

    main_pairs = [
        (
            "M6-FieldWiseEncoder-H4-continue",
            "M5-Delta-H4",
        ),
        (
            "M7-ParamTokenOnly-H4",
            "M6-FieldWiseEncoder-H4-continue",
        ),
        (
            "M7-FieldCoupling-H4",
            "M6-FieldWiseEncoder-H4-continue",
        ),
        (
            "M7-ParamTokenOnly-H4",
            "M7-FieldCoupling-H4",
        ),
    ]

    main_global_rows = []
    main_summary_rows = []
    main_detail_rows = []

    for split_name, relative_path in (
        main_chain_files.items()
    ):
        df = load_rollout(relative_path)

        main_global_rows.append(
            global_table(
                df=df,
                split_name=split_name,
                population="formal_full_split",
                model_order=main_models,
            )
        )

        for candidate, baseline in main_pairs:
            summary, detail = pairwise_rollout(
                df=df,
                split_name=split_name,
                population="formal_full_split",
                candidate=candidate,
                baseline=baseline,
            )

            main_summary_rows.append(summary)
            main_detail_rows.append(detail)

    main_global = pd.concat(
        main_global_rows,
        ignore_index=True,
    )

    main_summary = pd.DataFrame(
        main_summary_rows
    )

    main_details = pd.concat(
        main_detail_rows,
        ignore_index=True,
    )

    # -------------------------------------------------
    # 2. M8 trajectory 8 公平筛选
    # -------------------------------------------------
    m8_traj8_files = {
        "unseen_pr": (
            "outputs/tables/"
            "m8_ab_o5_fair_rollout_"
            "unseen_pr_traj8.csv"
        ),
        "unseen_ra": (
            "outputs/tables/"
            "m8_ab_o5_fair_rollout_"
            "unseen_ra_traj8.csv"
        ),
    }

    m8_models = [
        "M7-FieldCoupling-H4",
        "M7-ParamTokenOnly-H4",
        "M8-A-FullStatic-H4",
        "M8-B-ParamConditionedCoupling-H4",
    ]

    m8_pairs = [
        (
            "M8-A-FullStatic-H4",
            "M7-FieldCoupling-H4",
        ),
        (
            "M8-A-FullStatic-H4",
            "M7-ParamTokenOnly-H4",
        ),
        (
            "M8-B-ParamConditionedCoupling-H4",
            "M8-A-FullStatic-H4",
        ),
    ]

    traj8_global_rows = []
    traj8_summary_rows = []
    traj8_detail_rows = []

    for split_name, relative_path in (
        m8_traj8_files.items()
    ):
        df = load_rollout(relative_path)

        traj8_global_rows.append(
            global_table(
                df=df,
                split_name=split_name,
                population="trajectory_8_validation",
                model_order=m8_models,
            )
        )

        for candidate, baseline in m8_pairs:
            summary, detail = pairwise_rollout(
                df=df,
                split_name=split_name,
                population="trajectory_8_validation",
                candidate=candidate,
                baseline=baseline,
            )

            traj8_summary_rows.append(summary)
            traj8_detail_rows.append(detail)

    traj8_global = pd.concat(
        traj8_global_rows,
        ignore_index=True,
    )

    traj8_summary = pd.DataFrame(
        traj8_summary_rows
    )

    traj8_details = pd.concat(
        traj8_detail_rows,
        ignore_index=True,
    )

    # -------------------------------------------------
    # 3. M8 trajectory 9 独立确认
    # -------------------------------------------------
    m8_traj9_files = {
        "unseen_pr": (
            "outputs/tables/"
            "m8_ab_o5_confirmation_rollout_"
            "unseen_pr_traj9.csv"
        ),
        "unseen_ra": (
            "outputs/tables/"
            "m8_ab_o5_confirmation_rollout_"
            "unseen_ra_traj9.csv"
        ),
    }

    traj9_global_rows = []
    traj9_summary_rows = []
    traj9_detail_rows = []

    for split_name, relative_path in (
        m8_traj9_files.items()
    ):
        df = load_rollout(relative_path)

        traj9_global_rows.append(
            global_table(
                df=df,
                split_name=split_name,
                population="trajectory_9_confirmation",
                model_order=m8_models,
            )
        )

        for candidate, baseline in m8_pairs:
            summary, detail = pairwise_rollout(
                df=df,
                split_name=split_name,
                population="trajectory_9_confirmation",
                candidate=candidate,
                baseline=baseline,
            )

            traj9_summary_rows.append(summary)
            traj9_detail_rows.append(detail)

    traj9_global = pd.concat(
        traj9_global_rows,
        ignore_index=True,
    )

    traj9_summary = pd.DataFrame(
        traj9_summary_rows
    )

    traj9_details = pd.concat(
        traj9_detail_rows,
        ignore_index=True,
    )

    # -------------------------------------------------
    # 4. trajectory 8 物理指标
    # -------------------------------------------------
    physics_files = {
        "unseen_pr": (
            "outputs/tables/"
            "m8_tuning_baseline_physics_"
            "unseen_pr_traj8.csv"
        ),
        "unseen_ra": (
            "outputs/tables/"
            "m8_tuning_baseline_physics_"
            "unseen_ra_traj8.csv"
        ),
    }

    physics_rows = []
    physics_summary_rows = []
    physics_detail_rows = []

    for split_name, relative_path in (
        physics_files.items()
    ):
        df = load_physics(relative_path)

        physics_rows.append(
            compact_physics(
                df=df,
                split_name=split_name,
            )
        )

        resolved = {
            label: resolve_physics_model(df, label)
            for label in [
                "M6",
                "M7-Field",
                "M7-Param",
                "M8-A",
                "M8-B",
            ]
        }

        physics_pairs = [
            (
                resolved["M7-Field"],
                resolved["M6"],
            ),
            (
                resolved["M7-Param"],
                resolved["M6"],
            ),
            (
                resolved["M8-A"],
                resolved["M7-Field"],
            ),
            (
                resolved["M8-A"],
                resolved["M7-Param"],
            ),
            (
                resolved["M8-B"],
                resolved["M8-A"],
            ),
        ]

        for candidate, baseline in physics_pairs:
            summary, detail = pairwise_physics(
                df=df,
                split_name=split_name,
                candidate=candidate,
                baseline=baseline,
            )

            physics_summary_rows.append(summary)
            physics_detail_rows.append(detail)

    physics_table = pd.concat(
        physics_rows,
        ignore_index=True,
    )

    physics_summary = pd.DataFrame(
        physics_summary_rows
    )

    physics_details = pd.concat(
        physics_detail_rows,
        ignore_index=True,
    )

    # -------------------------------------------------
    # 保存
    # -------------------------------------------------
    outputs = {
        "01_main_chain_global.csv": main_global,
        "02_main_chain_pairwise.csv": main_summary,
        "03_main_chain_pairwise_details.csv": main_details,
        "04_m8_traj8_global.csv": traj8_global,
        "05_m8_traj8_pairwise.csv": traj8_summary,
        "06_m8_traj8_pairwise_details.csv": traj8_details,
        "07_m8_traj9_global.csv": traj9_global,
        "08_m8_traj9_pairwise.csv": traj9_summary,
        "09_m8_traj9_pairwise_details.csv": traj9_details,
        "10_physics_traj8.csv": physics_table,
        "11_physics_pairwise.csv": physics_summary,
        "12_physics_pairwise_details.csv": physics_details,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            OUT_DIR / filename,
            index=False,
        )

    # -------------------------------------------------
    # 终端紧凑输出
    # -------------------------------------------------
    rollout_columns = [
        "split",
        "population",
        "comparison",
        "physical_field_pass",
        "all_metric_pass",
        "global_pass",
        "global_h1_delta",
        "global_h4_delta",
        "global_h8_delta",
        "global_h16_delta",
    ]

    physics_columns = [
        "split",
        "comparison",
        "div_pass",
        "vorticity_pass",
        "div_h1_delta",
        "div_h4_delta",
        "div_h8_delta",
        "div_h16_delta",
        "vort_h1_delta",
        "vort_h4_delta",
        "vort_h8_delta",
        "vort_h16_delta",
    ]

    print_section(
        "表1：M5 → M6 → M7 正式主链 Global Rel-L2 (%)",
        main_global,
    )

    print_section(
        "表2：M5 → M6 → M7 逐层差值验收 "
        "（负数表示候选更好）",
        main_summary[rollout_columns],
    )

    print_section(
        "表3：M8-O5 trajectory 8 公平验证 "
        "（负数表示候选更好）",
        traj8_summary[rollout_columns],
    )

    print_section(
        "表4：M8-O5 trajectory 9 独立确认 "
        "（负数表示候选更好）",
        traj9_summary[rollout_columns],
    )

    print_section(
        "表5：trajectory 8 物理指标差值 "
        "（负数表示候选更好）",
        physics_summary[physics_columns],
    )

    print(
        "\n✅ 数据驱动阶段总体验收表已保存："
    )
    print(OUT_DIR)

    print(
        "\n说明："
        "\n- physical_field_pass 满分16："
        "4个物理场 × 4个horizon；"
        "\n- all_metric_pass 满分20："
        "4个物理场 + global；"
        "\n- global_pass 满分4；"
        "\n- div_pass和vorticity_pass满分均为4；"
        "\n- 所有delta均为候选减基线，负数代表候选更好。"
    )


if __name__ == "__main__":
    main()
