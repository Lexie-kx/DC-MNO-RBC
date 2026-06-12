import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PR_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables", "m5_eval_tables")
RA_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables", "m5_ra_eval_tables")
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


def plot_m5_minus_m3():
    pr = load_table(PR_DIR, "m5_minus_m3_delta.csv")
    ra = load_table(RA_DIR, "m5_minus_m3_delta.csv")

    fields = ["buoyancy", "u_x", "u_y", "pressure", "global"]

    for split_name, df in [("unseen Pr", pr), ("unseen Ra", ra)]:
        plt.figure(figsize=(8, 5))

        for field in fields:
            plt.plot(df["horizon"], df[field], marker="o", label=field)

        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Rollout horizon")
        plt.ylabel("M5 - M3-Delta Rel-L2 (%)")
        plt.title(f"{split_name}: M5 improvement over M3-Delta")
        plt.xticks([1, 4, 8, 16])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        split_tag = "unseen_pr" if split_name == "unseen Pr" else "unseen_ra"
        out_path = os.path.join(OUT_DIR, f"fig_m5_minus_m3_{split_tag}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"saved: {out_path}")


def main():
    plot_metric(
        metric_file="global_by_horizon.csv",
        ylabel="Global Rel-L2 (%)",
        save_name="fig_m5_rollout_global"
    )

    plot_metric(
        metric_file="u_x_by_horizon.csv",
        ylabel="u_x Rel-L2 (%)",
        save_name="fig_m5_rollout_ux"
    )

    plot_metric(
        metric_file="u_y_by_horizon.csv",
        ylabel="u_y Rel-L2 (%)",
        save_name="fig_m5_rollout_uy"
    )

    plot_m5_minus_m3()


if __name__ == "__main__":
    main()
