import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = ["M0", "M3-Delta", "M3-Delta-FiLM"]

SPLIT_LABELS = {
    "unseen_pr": "Unseen Pr",
    "unseen_ra": "Unseen Ra",
}

FIELD_LABELS = {
    "global": "Global",
    "u_x": r"$u_x$",
    "u_y": r"$u_y$",
}


def resolve_path(project_root, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def load_rollout_csv(csv_path, split_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing rollout csv: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"model", "horizon", "field", "rel_l2_percent", "mse"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["split_name"] = split_name
    return df


def plot_field_on_axis(ax, df, split_name, field):
    sub_df = df[
        (df["split_name"] == split_name) &
        (df["field"] == field)
    ].copy()

    for model in MODEL_ORDER:
        model_df = sub_df[sub_df["model"] == model].sort_values("horizon")

        if len(model_df) == 0:
            print(f"Skip missing model={model}, split={split_name}, field={field}")
            continue

        ax.plot(
            model_df["horizon"],
            model_df["rel_l2_percent"],
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xticks([1, 4, 8, 16])
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Rel-L2 (%)")
    ax.grid(True, linestyle="--", alpha=0.4)

    split_label = SPLIT_LABELS.get(split_name, split_name)
    field_label = FIELD_LABELS.get(field, field)

    ax.set_title(f"{split_label} - {field_label}")


def plot_global_comparison(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    plot_field_on_axis(
        ax=axes[0],
        df=df,
        split_name="unseen_pr",
        field="global",
    )

    plot_field_on_axis(
        ax=axes[1],
        df=df,
        split_name="unseen_ra",
        field="global",
    )

    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")

    fig.suptitle("Cross-parameter rollout global error", fontsize=15)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_velocity_comparison(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)

    plot_field_on_axis(
        ax=axes[0, 0],
        df=df,
        split_name="unseen_pr",
        field="u_x",
    )

    plot_field_on_axis(
        ax=axes[0, 1],
        df=df,
        split_name="unseen_pr",
        field="u_y",
    )

    plot_field_on_axis(
        ax=axes[1, 0],
        df=df,
        split_name="unseen_ra",
        field="u_x",
    )

    plot_field_on_axis(
        ax=axes[1, 1],
        df=df,
        split_name="unseen_ra",
        field="u_y",
    )

    for ax in axes.flatten():
        ax.legend(loc="upper left")

    fig.suptitle("Cross-parameter rollout velocity-field error", fontsize=15)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot clean cross-parameter rollout comparison figures."
    )

    parser.add_argument(
        "--unseen_ra_csv",
        type=str,
        default="outputs/tables/rollout_unseen_ra_summary.csv",
    )

    parser.add_argument(
        "--unseen_pr_csv",
        type=str,
        default="outputs/tables/rollout_unseen_pr_summary.csv",
    )

    parser.add_argument(
        "--figure_dir",
        type=str,
        default="outputs/figures/rollout",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    unseen_ra_csv = resolve_path(project_root, args.unseen_ra_csv)
    unseen_pr_csv = resolve_path(project_root, args.unseen_pr_csv)
    figure_dir = resolve_path(project_root, args.figure_dir)

    print("Plotting rollout comparison figures...")
    print(f"unseen Ra csv: {unseen_ra_csv}")
    print(f"unseen Pr csv: {unseen_pr_csv}")
    print(f"figure dir:    {figure_dir}")

    df_ra = load_rollout_csv(unseen_ra_csv, split_name="unseen_ra")
    df_pr = load_rollout_csv(unseen_pr_csv, split_name="unseen_pr")

    df = pd.concat([df_pr, df_ra], ignore_index=True)

    plot_global_comparison(
        df=df,
        output_path=os.path.join(
            figure_dir,
            "fig_rollout_global_comparison.png"
        ),
    )

    plot_velocity_comparison(
        df=df,
        output_path=os.path.join(
            figure_dir,
            "fig_rollout_velocity_comparison.png"
        ),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
