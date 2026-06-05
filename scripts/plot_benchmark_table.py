import os
import matplotlib.pyplot as plt


def main():
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
    )
    os.makedirs(out_dir, exist_ok=True)

    columns = [
        "Model",
        "buoyancy\nRel-L2 (%)",
        "u_x\nRel-L2 (%)",
        "u_y\nRel-L2 (%)",
        "pressure\nRel-L2 (%)",
        "Global\nRel-L2 (%)",
    ]

    # ================================
    # 最新 Controlled Benchmark Results
    # ================================
    data = [
        ["M0\nPersistence", "8.30", "22.72", "22.84", "4.98", "9.56"],

        ["M1-MSE\nControlled", "4.02", "859.48", "616.94", "3.65", "4.37"],

        ["M1-RelL2\nControlled", "16.15", "67.70", "51.66", "17.40", "17.60"],

        ["M3-MSE\nControlled", "4.08", "799.49", "359.99", "2.16", "4.11"],

        ["M3-RelL2\nControlled", "8.67", "66.22", "38.79", "7.45", "9.21"],
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 2.2)

    # ================================
    # 表格样式
    # ================================
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

        # Header
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#222222")

        # M1-MSE catastrophic failure
        elif row == 2:
            cell.set_facecolor("#ffe6e6")

        # M3-RelL2 best balanced model
        elif row == 5:
            cell.set_facecolor("#e6ffe6")

        else:
            cell.set_facecolor("#f7f7f7")

    ax.set_title(
        "Phase-1 Controlled Benchmark Summary\n(M0 / M1 / M3)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    save_path = os.path.join(
        out_dir,
        "phase1_controlled_benchmark_summary.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"✅ Saved benchmark table figure to: {save_path}")


if __name__ == "__main__":
    main()