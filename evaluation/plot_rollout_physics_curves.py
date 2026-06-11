import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(pr_df, ra_df, metric, ylabel, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    settings = [
        (axes[0], pr_df, "Unseen Pr"),
        (axes[1], ra_df, "Unseen Ra"),
    ]

    for ax, df, subtitle in settings:
        for model in ["M0", "M3-Delta", "M3-Delta-FiLM"]:
            sub = df[df["model_name"] == model].sort_values("horizon")
            ax.plot(
                sub["horizon"],
                sub[metric],
                marker="o",
                label=model,
            )

        ax.set_title(subtitle)
        ax.set_xlabel("Rollout horizon")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 4, 8, 16])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    pr_path = "outputs/tables/rollout_physics_unseen_pr_summary.csv"
    ra_path = "outputs/tables/rollout_physics_unseen_ra_summary.csv"

    pr_df = pd.read_csv(pr_path)
    ra_df = pd.read_csv(ra_path)

    out_dir = "outputs/figures/physics"
    os.makedirs(out_dir, exist_ok=True)

    plot_metric(
        pr_df=pr_df,
        ra_df=ra_df,
        metric="div_error_mae",
        ylabel="Divergence error MAE",
        title="Cross-parameter rollout divergence error",
        output_path=os.path.join(out_dir, "fig_rollout_divergence_error.png"),
    )

    plot_metric(
        pr_df=pr_df,
        ra_df=ra_df,
        metric="div_pred_mae",
        ylabel="Predicted divergence MAE",
        title="Cross-parameter rollout predicted divergence",
        output_path=os.path.join(out_dir, "fig_rollout_predicted_divergence.png"),
    )

    plot_metric(
        pr_df=pr_df,
        ra_df=ra_df,
        metric="vorticity_rel_l2_percent",
        ylabel="Vorticity Rel-L2 (%)",
        title="Cross-parameter rollout vorticity error",
        output_path=os.path.join(out_dir, "fig_rollout_vorticity_error.png"),
    )


if __name__ == "__main__":
    main()
