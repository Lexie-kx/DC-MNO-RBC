import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PR_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables", "m5_physics_eval_tables")
RA_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables", "m5_ra_physics_eval_tables")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "m5")
os.makedirs(OUT_DIR, exist_ok=True)


def load_table(base_dir, name):
    return pd.read_csv(os.path.join(base_dir, name))


def plot_metric(metric_file, ylabel, save_name):
    pr = load_table(PR_DIR, metric_file)
    ra = load_table(RA_DIR, metric_file)

    horizons = ["h=1", "h=4", "h=8", "h=16"]

    for split_name, df in [("unseen Pr", pr), ("unseen Ra", ra)]:
        plt.figure(figsize=(7, 5))

        for _, row in df.iterrows():
            model = row["model"]
            values = [row[h] for h in horizons]
            x = [1, 4, 8, 16]
            plt.plot(x, values, marker="o", label=model)

        plt.xlabel("Rollout horizon")
        plt.ylabel(ylabel)
        plt.title(f"{split_name}: {ylabel}")
        plt.xticks([1, 4, 8, 16])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        split_tag = "unseen_pr" if split_name == "unseen Pr" else "unseen_ra"
        out_path = os.path.join(OUT_DIR, f"{save_name}_{split_tag}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"saved: {out_path}")


def plot_m5_minus_m3_physics():
    pr = load_table(PR_DIR, "m5_minus_m3_physics.csv")
    ra = load_table(RA_DIR, "m5_minus_m3_physics.csv")

    metrics = ["div_pred_mae", "div_error_mae", "vorticity_rel_l2_percent"]

    for split_name, df in [("unseen Pr", pr), ("unseen Ra", ra)]:
        plt.figure(figsize=(8, 5))

        for metric in metrics:
            y = df[metric]
            if metric in ["div_pred_mae", "div_error_mae"]:
                y = y * 1000.0
                label = metric + " ×1000"
            else:
                label = metric

            plt.plot(df["horizon"], y, marker="o", label=label)

        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Rollout horizon")
        plt.ylabel("M5 - M3 difference")
        plt.title(f"{split_name}: M5 physics difference over M3-Delta")
        plt.xticks([1, 4, 8, 16])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        split_tag = "unseen_pr" if split_name == "unseen Pr" else "unseen_ra"
        out_path = os.path.join(OUT_DIR, f"fig_m5_minus_m3_physics_{split_tag}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"saved: {out_path}")


def main():
    plot_metric(
        metric_file="vorticity_rel_l2_percent_by_horizon.csv",
        ylabel="Vorticity Rel-L2 (%)",
        save_name="fig_m5_physics_vorticity"
    )

    plot_metric(
        metric_file="div_pred_mae_x1000_by_horizon.csv",
        ylabel="Predicted divergence MAE ×1000",
        save_name="fig_m5_physics_div_pred_x1000"
    )

    plot_metric(
        metric_file="div_error_mae_x1000_by_horizon.csv",
        ylabel="Divergence error MAE ×1000",
        save_name="fig_m5_physics_div_error_x1000"
    )

    plot_m5_minus_m3_physics()


if __name__ == "__main__":
    main()
