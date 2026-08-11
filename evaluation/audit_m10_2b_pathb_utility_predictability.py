import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)


# ============================================================
# Feature definitions
# Must match M10-2a exactly.
# ============================================================

LEGACY = [
    "b_mean",
    "b_rms",
    "ux_mean",
    "ux_rms",
    "uy_mean",
    "uy_rms",
    "log1p_path_a_rms",
    "log1p_path_b_rms",
    "logRa",
    "logPr",
]

CAP = (
    LEGACY
    +
    [
        "path_b_scale",
    ]
)

TEMPORAL = (
    CAP
    +
    [
        "d1_b_rms",
        "d1_ux_rms",
        "d1_uy_rms",
        "d2_b_rms",
        "d2_ux_rms",
        "d2_uy_rms",
        "m6_delta_b_rms",
        "m6_delta_uy_rms",
    ]
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "M10-2b offline Path-B utility "
            "predictability audit."
        )
    )

    parser.add_argument(
        "--train_csv",
        required=True,
    )

    parser.add_argument(
        "--val_csv",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/tables/"
            "m10_2b_utility"
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# Metrics
# ============================================================

def spearman_corr(x, y):

    x = pd.Series(x).rank(
        method="average"
    ).to_numpy()

    y = pd.Series(y).rank(
        method="average"
    ).to_numpy()

    if (
        np.std(x) < 1.0e-12
        or np.std(y) < 1.0e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def roc_auc_score_simple(
    y_true,
    score,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    pos = (
        y_true == 1
    )

    neg = (
        y_true == 0
    )

    n_pos = int(
        pos.sum()
    )

    n_neg = int(
        neg.sum()
    )

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return float("nan")

    ranks = pd.Series(
        score
    ).rank(
        method="average"
    ).to_numpy()

    rank_sum_pos = float(
        ranks[pos].sum()
    )

    auc = (
        rank_sum_pos
        -
        n_pos
        * (
            n_pos + 1
        )
        / 2.0
    ) / (
        n_pos
        * n_neg
    )

    return float(auc)


def regression_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    mae = float(
        np.mean(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )

    mse = float(
        np.mean(
            (
                y_true
                -
                y_pred
            )
            ** 2
        )
    )

    denominator = float(
        np.sum(
            (
                y_true
                -
                y_true.mean()
            )
            ** 2
        )
    )

    r2 = float(
        1.0
        -
        np.sum(
            (
                y_true
                -
                y_pred
            )
            ** 2
        )
        /
        (
            denominator
            +
            1.0e-30
        )
    )

    return {
        "mae":
            mae,

        "mse":
            mse,

        "r2":
            r2,

        "spearman":
            spearman_corr(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# Models
# ============================================================

class UtilityClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                32,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                16,
            ),
            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.net(x)
            .squeeze(-1)
        )


class UtilityRegressor(nn.Module):

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                32,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                16,
            ),
            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.net(x)
            .squeeze(-1)
        )


# ============================================================
# Fit helper
# ============================================================

def fit_models(
    train_x,
    train_help,
    train_gain,
    val_x,
    *,
    seed,
    epochs,
):

    set_seed(seed)

    train_x = np.asarray(
        train_x,
        dtype=np.float32,
    )

    val_x = np.asarray(
        val_x,
        dtype=np.float32,
    )

    train_help = np.asarray(
        train_help,
        dtype=np.float32,
    )

    train_gain = np.asarray(
        train_gain,
        dtype=np.float32,
    )

    mean = train_x.mean(
        axis=0,
        keepdims=True,
    )

    std = train_x.std(
        axis=0,
        keepdims=True,
    )

    std[
        std < 1.0e-6
    ] = 1.0

    train_z = (
        train_x
        -
        mean
    ) / std

    val_z = (
        val_x
        -
        mean
    ) / std

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x_train = torch.tensor(
        train_z,
        device=device,
    )

    x_val = torch.tensor(
        val_z,
        device=device,
    )

    y_help = torch.tensor(
        train_help,
        device=device,
    )

    y_gain = torch.tensor(
        train_gain,
        device=device,
    )

    classifier = (
        UtilityClassifier(
            train_z.shape[1]
        ).to(device)
    )

    regressor = (
        UtilityRegressor(
            train_z.shape[1]
        ).to(device)
    )

    optimizer_c = torch.optim.Adam(
        classifier.parameters(),
        lr=1.0e-3,
    )

    optimizer_r = torch.optim.Adam(
        regressor.parameters(),
        lr=1.0e-3,
    )

    loss_c = (
        nn.BCEWithLogitsLoss()
    )

    loss_r = nn.SmoothL1Loss()

    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        seed
    )

    batch_size = min(
        512,
        len(train_z),
    )

    for _ in range(
        epochs
    ):

        permutation = torch.randperm(
            len(train_z),
            generator=generator,
        )

        classifier.train()
        regressor.train()

        for start in range(
            0,
            len(train_z),
            batch_size,
        ):

            idx = permutation[
                start:
                start + batch_size
            ].to(device)

            # ---------------------------
            # Utility sign
            # ---------------------------

            logits = classifier(
                x_train[idx]
            )

            lc = loss_c(
                logits,
                y_help[idx],
            )

            optimizer_c.zero_grad(
                set_to_none=True
            )

            lc.backward()
            optimizer_c.step()

            # ---------------------------
            # Utility magnitude
            # ---------------------------

            pred_gain = regressor(
                x_train[idx]
            )

            lr = loss_r(
                pred_gain,
                y_gain[idx],
            )

            optimizer_r.zero_grad(
                set_to_none=True
            )

            lr.backward()
            optimizer_r.step()

    classifier.eval()
    regressor.eval()

    with torch.no_grad():

        train_probability = (
            torch.sigmoid(
                classifier(
                    x_train
                )
            )
            .cpu()
            .numpy()
        )

        val_probability = (
            torch.sigmoid(
                classifier(
                    x_val
                )
            )
            .cpu()
            .numpy()
        )

        train_gain_pred = (
            regressor(
                x_train
            )
            .cpu()
            .numpy()
        )

        val_gain_pred = (
            regressor(
                x_val
            )
            .cpu()
            .numpy()
        )

    return (
        train_probability,
        val_probability,
        train_gain_pred,
        val_gain_pred,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    train = pd.read_csv(
        args.train_csv
    )

    val = pd.read_csv(
        args.val_csv
    )

    # --------------------------------------------------------
    # M10-2b asks about rollout-contaminated states.
    # Step1 is still a pure GT initial state, so exclude it.
    # --------------------------------------------------------

    train = train[
        train["step"] >= 2
    ].copy()

    val = val[
        val["step"] >= 2
    ].copy()

    if (
        len(train) == 0
        or len(val) == 0
    ):
        raise RuntimeError(
            "Empty TRAIN/VAL rows."
        )

    # --------------------------------------------------------
    # Dimensionless utility.
    #
    # Positive:
    #     Path B improves buoyancy.
    #
    # Negative:
    #     Path B hurts buoyancy.
    # --------------------------------------------------------

    for frame in [
        train,
        val,
    ]:

        frame[
            "relative_utility_gain"
        ] = (
            frame[
                "utility_gain"
            ]
            /
            (
                frame[
                    "m6_b_error"
                ]
                +
                1.0e-12
            )
        )

        frame[
            "utility_help"
        ] = (
            frame[
                "relative_utility_gain"
            ]
            > 0.0
        ).astype(int)

    print(
        "=" * 110
    )

    print(
        "M10-2b PATH-B UTILITY PREDICTABILITY AUDIT"
    )

    print(
        "=" * 110
    )

    print(
        "📌 Stage: pre-candidate control diagnostic"
    )

    print(
        "📌 Neural-operator rollout: NO"
    )

    print(
        "📌 Uses saved M10-2a TRAIN/VAL rows only"
    )

    print(
        "📌 TEST split: NOT ACCESSED"
    )

    print(
        "📌 Target:"
    )

    print(
        "   relative utility = "
        "(M6 error - B-only error) / M6 error"
    )

    print(
        "📌 Positive = Path B helps"
    )

    print(
        "📌 Step1 excluded"
    )

    print()
    print(
        "TRAIN_ROWS:",
        len(train),
    )

    print(
        "VAL_ROWS:",
        len(val),
    )

    # --------------------------------------------------------
    # Utility behavior
    # --------------------------------------------------------

    utility_summary = (
        pd.concat(
            [
                train,
                val,
            ],
            ignore_index=True,
        )
        .groupby(
            [
                "split",
                "step",
            ],
            as_index=False,
        )
        .agg(
            count=(
                "relative_utility_gain",
                "size",
            ),
            help_fraction=(
                "utility_help",
                "mean",
            ),
            relative_gain_mean=(
                "relative_utility_gain",
                "mean",
            ),
            relative_gain_std=(
                "relative_utility_gain",
                "std",
            ),
            q_mean=(
                "q_target",
                "mean",
            ),
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "UTILITY TARGET BY STEP"
    )

    print(
        "=" * 110
    )

    print(
        utility_summary.to_string(
            index=False
        )
    )

    feature_sets = {
        "Legacy10":
            [
                f"legacy__{name}"
                for name in LEGACY
            ],

        "LegacyCap11":
            [
                f"cap__{name}"
                for name in CAP
            ],

        "Temporal19":
            [
                f"temporal__{name}"
                for name in TEMPORAL
            ],

        "TemporalAlpha20":
            (
                [
                    f"temporal__{name}"
                    for name in TEMPORAL
                ]
                +
                [
                    "alpha_b",
                ]
            ),
    }

    classification_rows = []
    regression_rows = []

    stored = {}

    for (
        feature_name,
        columns,
    ) in feature_sets.items():

        (
            train_prob,
            val_prob,
            train_gain_pred,
            val_gain_pred,
        ) = fit_models(
            train_x=(
                train[
                    columns
                ].to_numpy()
            ),
            train_help=(
                train[
                    "utility_help"
                ].to_numpy()
            ),
            train_gain=(
                train[
                    "relative_utility_gain"
                ].to_numpy()
            ),
            val_x=(
                val[
                    columns
                ].to_numpy()
            ),
            seed=args.seed,
            epochs=args.epochs,
        )

        y_help = val[
            "utility_help"
        ].to_numpy()

        y_gain = val[
            "relative_utility_gain"
        ].to_numpy()

        auc = roc_auc_score_simple(
            y_help,
            val_prob,
        )

        accuracy = float(
            np.mean(
                (
                    val_prob >= 0.5
                ).astype(
                    np.int64
                )
                ==
                y_help
            )
        )

        classification_rows.append(
            {
                "features":
                    feature_name,

                "num_features":
                    len(columns),

                "auc":
                    auc,

                "accuracy_at_0p5":
                    accuracy,

                "spearman_prob_vs_utility":
                    spearman_corr(
                        val_prob,
                        y_gain,
                    ),

                "spearman_prob_vs_q":
                    spearman_corr(
                        val_prob,
                        val[
                            "q_target"
                        ].to_numpy(),
                    ),
            }
        )

        reg_metrics = (
            regression_metrics(
                y_gain,
                val_gain_pred,
            )
        )

        regression_rows.append(
            {
                "features":
                    feature_name,

                "num_features":
                    len(columns),

                **reg_metrics,
            }
        )

        stored[
            feature_name
        ] = {
            "train_prob":
                train_prob,

            "val_prob":
                val_prob,

            "val_gain_pred":
                val_gain_pred,
        }

    classification_df = pd.DataFrame(
        classification_rows
    )

    regression_df = pd.DataFrame(
        regression_rows
    )

    print()
    print(
        "=" * 110
    )

    print(
        "VAL PATH-B HELP CLASSIFICATION"
    )

    print(
        "=" * 110
    )

    print(
        classification_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "VAL RELATIVE-UTILITY REGRESSION"
    )

    print(
        "=" * 110
    )

    print(
        regression_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Predicted-utility stratification.
    #
    # Use Temporal19 first, because that is the clean
    # proposed temporal-reliability feature set.
    #
    # Bin thresholds are derived from TRAIN predictions only.
    # --------------------------------------------------------

    best_name = "Temporal19"

    train_prob = stored[
        best_name
    ][
        "train_prob"
    ]

    val_prob = stored[
        best_name
    ][
        "val_prob"
    ]

    edges = np.quantile(
        train_prob,
        [
            0.25,
            0.50,
            0.75,
        ],
    )

    val_for_bins = (
        val[
            [
                "q_target",
                "utility_help",
                "relative_utility_gain",
            ]
        ].copy()
    )

    val_for_bins[
        "pred_help_probability"
    ] = val_prob

    val_for_bins[
        "pred_bin"
    ] = np.digitize(
        val_prob,
        edges,
        right=False,
    )

    names = {
        0: "P1-low",
        1: "P2",
        2: "P3",
        3: "P4-high",
    }

    val_for_bins[
        "pred_bin"
    ] = (
        val_for_bins[
            "pred_bin"
        ].map(names)
    )

    predicted_bins = (
        val_for_bins.groupby(
            "pred_bin",
            observed=False,
            as_index=False,
        )
        .agg(
            count=(
                "utility_help",
                "size",
            ),
            predicted_probability_mean=(
                "pred_help_probability",
                "mean",
            ),
            actual_help_fraction=(
                "utility_help",
                "mean",
            ),
            actual_relative_gain_mean=(
                "relative_utility_gain",
                "mean",
            ),
            q_mean=(
                "q_target",
                "mean",
            ),
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "TEMPORAL19 PREDICTED-UTILITY STRATIFICATION"
    )

    print(
        "Bins fixed from TRAIN predictions"
    )

    print(
        "=" * 110
    )

    print(
        predicted_bins.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Baseline context
    # --------------------------------------------------------

    base_help_fraction = float(
        train[
            "utility_help"
        ].mean()
    )

    val_help_fraction = float(
        val[
            "utility_help"
        ].mean()
    )

    print()
    print(
        "========== BASELINE =========="
    )

    print(
        "TRAIN_HELP_FRACTION:",
        f"{base_help_fraction:.6f}",
    )

    print(
        "VAL_HELP_FRACTION:",
        f"{val_help_fraction:.6f}",
    )

    print(
        "Random-ranking AUC = 0.5"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = os.path.join(
        PROJECT_ROOT,
        args.output_dir,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    prefix = os.path.join(
        output_dir,
        args.run_name,
    )

    utility_summary.to_csv(
        prefix
        +
        "_utility_summary.csv",
        index=False,
    )

    classification_df.to_csv(
        prefix
        +
        "_classification.csv",
        index=False,
    )

    regression_df.to_csv(
        prefix
        +
        "_regression.csv",
        index=False,
    )

    predicted_bins.to_csv(
        prefix
        +
        "_predicted_bins.csv",
        index=False,
    )

    print()
    print(
        "✅ Saved:"
    )

    print(
        " ",
        prefix
        +
        "_utility_summary.csv",
    )

    print(
        " ",
        prefix
        +
        "_classification.csv",
    )

    print(
        " ",
        prefix
        +
        "_regression.csv",
    )

    print(
        " ",
        prefix
        +
        "_predicted_bins.csv",
    )


if __name__ == "__main__":
    main()
