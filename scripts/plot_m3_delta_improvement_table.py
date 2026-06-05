import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # =========================
    # 已有评估结果：Rel-L2 (%)
    # =========================
    results = {
        "Field": ["buoyancy", "u_x", "u_y", "pressure", "global"],

        "M0 Persistence": [8.30, 22.72, 22.84, 4.98, 9.56],

        "M3-State Rel-L2": [8.67, 66.22, 38.79, 7.45, 9.21],

        "M3-Delta Rel-L2": [6.82, 15.80, 15.95, 2.92, 7.06],
    }

    df = pd.DataFrame(results)

    # =========================
    # 计算 improvement over M0
    # improvement = 1 - error_model / error_M0
    # 大于 0 表示优于 M0
    # 小于 0 表示不如 M0
    # =========================
    df["M3-State Improvement over M0 (%)"] = (
        1.0 - df["M3-State Rel-L2"] / df["M0 Persistence"]
    ) * 100.0

    df["M3-Delta Improvement over M0 (%)"] = (
        1.0 - df["M3-Delta Rel-L2"] / df["M0 Persistence"]
    ) * 100.0

    # 保留两位小数
    df_round = df.copy()
    for col in df_round.columns:
        if col != "Field":
            df_round[col] = df_round[col].round(2)

    # =========================
    # 保存目录
    # =========================
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(root_dir, "outputs", "figures", "m3_delta")
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 保存 CSV
    # =========================
    csv_path = os.path.join(save_dir, "m3_delta_m0_normalized_improvement.csv")
    df_round.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n📊 M0-normalized improvement table:")
    print(df_round.to_string(index=False))
    print(f"\n✅ CSV 已保存至: {csv_path}")

    # =========================
    # 画表格图：排版优化版
    # =========================
    table_df = df_round[
        [
            "Field",
            "M0 Persistence",
            "M3-State Rel-L2",
            "M3-Delta Rel-L2",
            "M3-State Improvement over M0 (%)",
            "M3-Delta Improvement over M0 (%)",
        ]
    ]

    col_labels = [
        "Field",
        "M0\nRel-L2 (%)",
        "M3-State\nRel-L2 (%)",
        "M3-Delta\nRel-L2 (%)",
        "State\nImp. vs M0 (%)",
        "Delta\nImp. vs M0 (%)",
    ]

    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    n_rows = len(table_df)
    n_cols = len(col_labels)

    # 表头加粗
    for col_idx in range(n_cols):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.5)

    # global 行加粗
    global_row_idx = n_rows
    for col_idx in range(n_cols):
        cell = table[(global_row_idx, col_idx)]
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.5)

    # 所有单元格边框统一
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    ax.set_title(
        "M3-Delta Achieves Consistent Improvement over Persistence",
        fontsize=18,
        fontweight="bold",
        pad=28,
    )

    fig_path = os.path.join(save_dir, "m3_delta_m0_normalized_improvement_table.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ 表格图已保存至: {fig_path}")

    # =========================
    # 画 improvement bar 图
    # =========================
    plot_df = df_round[df_round["Field"] != "global"]

    x = range(len(plot_df["Field"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        [i - width / 2 for i in x],
        plot_df["M3-State Improvement over M0 (%)"],
        width=width,
        label="M3-State vs M0",
    )

    ax.bar(
        [i + width / 2 for i in x],
        plot_df["M3-Delta Improvement over M0 (%)"],
        width=width,
        label="M3-Delta vs M0",
    )

    ax.axhline(0, linewidth=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["Field"].tolist())
    ax.set_ylabel("Improvement over M0 (%)")
    ax.set_title(
        "Field-wise Improvement over Persistence Baseline",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()

    bar_path = os.path.join(save_dir, "m3_delta_m0_normalized_improvement_bar.png")
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ 柱状图已保存至: {bar_path}")


if __name__ == "__main__":
    main()