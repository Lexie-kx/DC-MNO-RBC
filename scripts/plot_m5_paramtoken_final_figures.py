import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "M0",
    "M3-Delta",
    "M3-Delta-FiLM",
    "M5-Delta-H4",
    "M5-Delta-FiLM-H4",
    "M5-Delta-ParameterToken-H4",
]

FIELDS = ["buoyancy", "u_x", "u_y", "pressure", "global"]


def load_rollout(csv_path):
    df = pd.read_csv(csv_path)
    wide = df.pivot_table(
        index=["model", "horizon"],
        columns="field",
        values="rel_l2_percent",
    ).reset_index()
    return wide


def load_physics(csv_path):
    df = pd.read_csv(csv_path)

    if "model_name" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_name": "model"})

    if "div_err_mae" in df.columns and "div_error_mae" not in df.columns:
        df = df.rename(columns={"div_err_mae": "div_error_mae"})

    if "vort_rel_l2%" in df.columns and "vorticity_rel_l2_percent" not in df.columns:
        df = df.rename(columns={"vort_rel_l2%": "vorticity_rel_l2_percent"})

    return df


def plot_metric_curve(ax, df, metric, title, ylabel):
    for model in MODEL_ORDER:
        if model not in set(df["model"]):
            continue
        sub = df[df["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub[metric], marker="o", label=model)

    ax.set_title(title)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def plot_improvement(ax, rollout_wide, title, baseline="M5-Delta-H4"):
    pt = rollout_wide[rollout_wide["model"] == "M5-Delta-ParameterToken-H4"].set_index("horizon")
    base = rollout_wide[rollout_wide["model"] == baseline].set_index("horizon")

    diff = pt[FIELDS] - base[FIELDS]

    for field in FIELDS:
        ax.plot(diff.index, diff[field], marker="o", label=field)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel(f"ParamToken - {baseline} Rel-L2 (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr_rollout", required=True)
    parser.add_argument("--ra_rollout", required=True)
    parser.add_argument("--pr_physics", required=True)
    parser.add_argument("--ra_physics", required=True)
    parser.add_argument("--out_dir", default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pr_rollout = load_rollout(args.pr_rollout)
    ra_rollout = load_rollout(args.ra_rollout)

    pr_physics = load_physics(args.pr_physics)
    ra_physics = load_physics(args.ra_physics)

    # Figure 1: rollout global + improvement
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_metric_curve(
        axes[0, 0],
        pr_rollout,
        metric="global",
        title="unseen Pr: Global Rel-L2 (%)",
        ylabel="Global Rel-L2 (%)",
    )

    plot_metric_curve(
        axes[0, 1],
        ra_rollout,
        metric="global",
        title="unseen Ra: Global Rel-L2 (%)",
        ylabel="Global Rel-L2 (%)",
    )

    plot_improvement(
        axes[1, 0],
        pr_rollout,
        title="unseen Pr: ParamToken improvement over M5-Delta-H4",
        baseline="M5-Delta-H4",
    )

    plot_improvement(
        axes[1, 1],
        ra_rollout,
        title="unseen Ra: ParamToken improvement over M5-Delta-H4",
        baseline="M5-Delta-H4",
    )

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "m5_paramtoken_rollout_summary_pr_ra.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ saved: {out_path}")

    # Figure 2: physics diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_metric_curve(
        axes[0, 0],
        pr_physics,
        metric="vorticity_rel_l2_percent",
        title="unseen Pr: Vorticity Rel-L2 (%)",
        ylabel="Vorticity Rel-L2 (%)",
    )

    plot_metric_curve(
        axes[0, 1],
        ra_physics,
        metric="vorticity_rel_l2_percent",
        title="unseen Ra: Vorticity Rel-L2 (%)",
        ylabel="Vorticity Rel-L2 (%)",
    )

    plot_metric_curve(
        axes[1, 0],
        pr_physics,
        metric="div_error_mae",
        title="unseen Pr: Divergence Error MAE",
        ylabel="Divergence Error MAE",
    )

    plot_metric_curve(
        axes[1, 1],
        ra_physics,
        metric="div_error_mae",
        title="unseen Ra: Divergence Error MAE",
        ylabel="Divergence Error MAE",
    )

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "m5_paramtoken_physics_summary_pr_ra.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ saved: {out_path}")


if __name__ == "__main__":
    main()
