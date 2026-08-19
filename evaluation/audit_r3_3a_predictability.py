from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Frozen identities
# ============================================================

R3_3_CONTRACT = (
    "configs/r3/r3_3_utility_contract.json"
)

EXPECTED_R3_3_CONTRACT_SHA256 = (
    "426299fcce724d1e74490887c39ea583"
    "07f2180bf464c4d41ee6ee170b6c4af8"
)

DATASET_REGISTRY = (
    "configs/r3/"
    "r3_3a_formal_utility_dataset_registry.json"
)

EXPECTED_DATASET_REGISTRY_SHA256 = (
    "97171f60689c8fe3c32746640a109b1f"
    "e554a091073f68536550711b7f273d40"
)

EXPECTED_GENERATOR_FREEZE_COMMIT = (
    "7963c0934aa49aa26b410f32829bc306f8d48ed5"
)

EXPECTED_DATASET_FREEZE_COMMIT = (
    "cc4dfb12d62ea9940cb4e5ed7a32f43d7821f326"
)

SEED = 42
FORMAL_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3

ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)

ARMS = (
    "ParamUtility",
    "StateUtility",
)

FEATURE_NAMES = (
    "state_b_mean_norm",
    "state_b_rms_norm",
    "state_ux_mean_norm",
    "state_ux_rms_norm",
    "state_uy_mean_norm",
    "state_uy_rms_norm",
    (
        "log1p_canonical_"
        "buoyancy_advection_"
        "signal_rms_norm"
    ),
    (
        "log1p_canonical_"
        "buoyancy_forcing_"
        "signal_rms_norm"
    ),
    "logRa",
    "logPr",
)

STATE_FEATURE_COUNT = 8

KEY_COLUMNS = (
    "group",
    "trajectory",
    "t0",
    "step",
)

EXPECTED_CLASSIFIER_NUMEL = 897


# ============================================================
# Model
# ============================================================

class UtilityClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return (
            self.net(x)
            .squeeze(-1)
        )


def parameter_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def binary_roc_auc(
    y_true,
    y_score,
):
    """
    Binary ROC-AUC via the Mann-Whitney rank statistic.

    - Pure NumPy; no sklearn dependency.
    - Uses average ranks for exact score ties.
    - Equivalent to the standard binary ROC-AUC definition.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    ).reshape(-1)

    if (
        y_true.shape
        !=
        y_score.shape
    ):
        raise ValueError(
            "AUC y_true/y_score shape mismatch: "
            f"{y_true.shape} vs {y_score.shape}"
        )

    if not np.isfinite(
        y_score
    ).all():
        raise ValueError(
            "AUC received non-finite scores."
        )

    unique_labels = set(
        np.unique(
            y_true
        ).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            "AUC requires binary labels {0,1}."
        )

    n_pos = int(
        np.sum(
            y_true == 1
        )
    )

    n_neg = int(
        np.sum(
            y_true == 0
        )
    )

    if (
        n_pos == 0
        or
        n_neg == 0
    ):
        raise ValueError(
            "ROC-AUC is undefined when only "
            "one label class is present."
        )

    # Stable sorting makes the implementation deterministic.
    order = np.argsort(
        y_score,
        kind="mergesort",
    )

    sorted_scores = (
        y_score[
            order
        ]
    )

    ranks_sorted = np.empty(
        len(y_score),
        dtype=np.float64,
    )

    start = 0
    n = len(
        sorted_scores
    )

    while start < n:

        end = (
            start
            +
            1
        )

        while (
            end < n
            and
            sorted_scores[end]
            ==
            sorted_scores[start]
        ):
            end += 1

        # 1-indexed average rank for [start, end).
        average_rank = (
            (
                start + 1
            )
            +
            end
        ) / 2.0

        ranks_sorted[
            start:end
        ] = average_rank

        start = end

    ranks = np.empty_like(
        ranks_sorted
    )

    ranks[
        order
    ] = ranks_sorted

    positive_rank_sum = float(
        np.sum(
            ranks[
                y_true == 1
            ]
        )
    )

    auc = (
        positive_rank_sum
        -
        (
            n_pos
            *
            (
                n_pos + 1
            )
            /
            2.0
        )
    ) / (
        n_pos
        *
        n_neg
    )

    return float(
        auc
    )


# ============================================================
# Generic helpers
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "R3-3A TRAIN-only Utility classifier fit "
            "and VAL-only predictability audit."
        )
    )

    p.add_argument(
        "--split_label",
        required=True,
        choices=(
            "unseen_pr",
            "unseen_ra",
        ),
    )

    p.add_argument(
        "--audit_only",
        action="store_true",
    )

    p.add_argument(
        "--debug_epochs",
        type=int,
        default=None,
        help=(
            "DEBUG only. Formal protocol is fixed "
            "to 100 epochs."
        ),
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    return p.parse_args()


def resolve_path(path):

    p = Path(path)

    if not p.is_absolute():
        p = PROJECT_ROOT / p

    return p.resolve()


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def audit_sha(
    path,
    expected,
    label,
):

    actual = sha256_file(path)

    if actual != expected:

        raise RuntimeError(
            f"{label} SHA mismatch.\n"
            f"expected={expected}\n"
            f"actual={actual}\n"
            f"path={path}"
        )

    print(
        f"✅ {label} SHA exact match"
    )

    return actual


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def audit_frozen_git_commit(
    expected_commit,
    label,
):
    """
    Verify that the frozen provenance commit exists
    and is an ancestor of the current repository HEAD.

    This is provenance-only. It does not alter any
    training, feature, label, or evaluation semantics.
    """

    verify = subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{expected_commit}^{{commit}}",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if verify.returncode != 0:
        raise RuntimeError(
            f"{label} commit does not exist: "
            f"{expected_commit}"
        )

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            expected_commit,
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            f"{label} commit is not an ancestor "
            f"of current HEAD: {expected_commit}"
        )

    print(
        f"✅ {label} commit exists "
        "and is ancestor of HEAD"
    )


# ============================================================
# Registry audit
# ============================================================

def load_and_audit_registry(
    split_label,
):

    r3_3_path = resolve_path(
        R3_3_CONTRACT
    )

    registry_path = resolve_path(
        DATASET_REGISTRY
    )

    r3_3_sha = audit_sha(
        r3_3_path,
        EXPECTED_R3_3_CONTRACT_SHA256,
        "R3-3 contract",
    )

    registry_sha = audit_sha(
        registry_path,
        EXPECTED_DATASET_REGISTRY_SHA256,
        "R3-3A formal Utility dataset registry",
    )

    audit_frozen_git_commit(
        EXPECTED_DATASET_FREEZE_COMMIT,
        "R3-3A formal Utility dataset freeze",
    )

    with open(
        registry_path,
        "r",
        encoding="utf-8",
    ) as f:
        registry = json.load(f)

    if registry.get("stage") != "R3-3A":
        raise RuntimeError(
            "Dataset registry stage mismatch."
        )

    if (
        registry.get("status")
        !=
        "FORMAL_TRAIN_VAL_DATA_COMPLETE"
    ):
        raise RuntimeError(
            "Formal Utility datasets are not "
            "frozen COMPLETE."
        )

    generator_freeze = registry[
        "generator_freeze"
    ]

    if (
        generator_freeze["commit"]
        !=
        EXPECTED_GENERATOR_FREEZE_COMMIT
    ):
        raise RuntimeError(
            "Generator freeze commit mismatch."
        )

    formal_protocol = registry[
        "formal_protocol"
    ]

    if formal_protocol["test_accessed"]:
        raise RuntimeError(
            "Frozen Utility registry claims "
            "TEST access."
        )

    if (
        formal_protocol[
            "utility_error_accumulation_dtype"
        ]
        !=
        "float64"
    ):
        raise RuntimeError(
            "Utility label arithmetic changed."
        )

    train_key = (
        f"{split_label}_train"
    )

    val_key = (
        f"{split_label}_val"
    )

    datasets = registry[
        "datasets"
    ]

    train_entry = datasets[
        train_key
    ]

    val_entry = datasets[
        val_key
    ]

    for name, entry in (
        ("TRAIN", train_entry),
        ("VAL", val_entry),
    ):

        for kind in (
            "rows",
            "summary",
            "metadata",
        ):

            path = resolve_path(
                entry[
                    f"{kind}_path"
                ]
            )

            expected = entry[
                f"{kind}_sha256"
            ]

            audit_sha(
                path,
                expected,
                (
                    f"{split_label} "
                    f"{name} {kind}"
                ),
            )

        metadata_path = resolve_path(
            entry["metadata_path"]
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        expected_split = (
            "train"
            if name == "TRAIN"
            else "val"
        )

        if (
            metadata["data_split"]
            !=
            expected_split
        ):
            raise RuntimeError(
                f"{name} metadata split mismatch."
            )

        if metadata["test_accessed"]:
            raise RuntimeError(
                f"{name} metadata claims TEST access."
            )

        if (
            metadata[
                "utility_error_accumulation_dtype"
            ]
            !=
            "float64"
        ):
            raise RuntimeError(
                f"{name} Utility dtype changed."
            )

    return {
        "registry":
            registry,
        "registry_path":
            registry_path,
        "registry_sha":
            registry_sha,
        "r3_3_sha":
            r3_3_sha,
        "train_entry":
            train_entry,
        "val_entry":
            val_entry,
    }


# ============================================================
# Utility-row audits
# ============================================================

def load_rows(
    entry,
    split_label,
    data_split,
):

    path = resolve_path(
        entry["rows_path"]
    )

    df = pd.read_csv(path)

    if len(df) != int(
        entry["rows"]
    ):
        raise RuntimeError(
            f"{split_label}/{data_split}: "
            "row count mismatch."
        )

    required = set(
        KEY_COLUMNS
    ) | {
        "split",
        "data_split",
        "term",
        "helps",
        "gain",
        "relative_gain",
    } | set(
        FEATURE_NAMES
    )

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing Utility-row columns:\n"
            f"{sorted(missing)}"
        )

    if set(
        df["term"].unique()
    ) != set(
        ACTIVE_TERMS
    ):
        raise RuntimeError(
            "Unexpected Utility term set."
        )

    if set(
        df["step"].unique()
    ) != {
        1,
        2,
        3,
        4,
    }:
        raise RuntimeError(
            "Utility rows must contain "
            "steps 1..4 exactly."
        )

    if not (
        df["split"]
        ==
        split_label
    ).all():
        raise RuntimeError(
            "Utility split label mismatch."
        )

    if not (
        df["data_split"]
        ==
        data_split
    ).all():
        raise RuntimeError(
            "Utility data_split mismatch."
        )

    helps = set(
        df["helps"].unique()
    )

    if not helps.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            "Utility labels are not binary."
        )

    reproduced = (
        df["gain"] > 0.0
    ).astype(int)

    mismatch = int(
        (
            reproduced
            !=
            df["helps"]
        ).sum()
    )

    if mismatch != 0:
        raise RuntimeError(
            "Frozen Utility gain/help "
            f"reproduction mismatch={mismatch}"
        )

    values = (
        df[
            list(FEATURE_NAMES)
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Non-finite Utility features."
        )

    duplicate_key = list(
        KEY_COLUMNS
    ) + ["term"]

    if df.duplicated(
        duplicate_key
    ).any():
        raise RuntimeError(
            "Duplicate sample-step-term rows."
        )

    for term in ACTIVE_TERMS:

        labels = (
            df.loc[
                df["term"] == term,
                "helps",
            ]
            .to_numpy()
        )

        if len(
            np.unique(labels)
        ) != 2:
            raise RuntimeError(
                f"{split_label}/{data_split}/"
                f"{term}: only one label class."
            )

    print(
        f"✅ {split_label} {data_split.upper()} "
        f"Utility-row audit PASS | rows={len(df)}"
    )

    return df


# ============================================================
# Exact duplicated-feature audit
# ============================================================

def audit_term_feature_identity(df):

    adv = (
        df[
            df["term"]
            ==
            "buoyancy_advection"
        ]
        .sort_values(
            list(KEY_COLUMNS)
        )
        .reset_index(drop=True)
    )

    forcing = (
        df[
            df["term"]
            ==
            "buoyancy_forcing"
        ]
        .sort_values(
            list(KEY_COLUMNS)
        )
        .reset_index(drop=True)
    )

    if len(adv) != len(forcing):
        raise RuntimeError(
            "Per-term Utility row counts differ."
        )

    for key in KEY_COLUMNS:

        if not np.array_equal(
            adv[key].to_numpy(),
            forcing[key].to_numpy(),
        ):
            raise RuntimeError(
                f"Per-term row key mismatch: {key}"
            )

    adv_x = adv[
        list(FEATURE_NAMES)
    ].to_numpy(
        dtype=np.float64
    )

    forcing_x = forcing[
        list(FEATURE_NAMES)
    ].to_numpy(
        dtype=np.float64
    )

    if not np.array_equal(
        adv_x,
        forcing_x,
    ):
        max_abs = float(
            np.max(
                np.abs(
                    adv_x
                    -
                    forcing_x
                )
            )
        )

        raise RuntimeError(
            "The two term rows do not contain "
            "the exact same R3-2 10D features. "
            f"max_abs={max_abs:.12e}"
        )

    print(
        "✅ Both Utility terms share exact "
        "sample-state 10D features"
    )


# ============================================================
# TRAIN-only shared feature normalization
# ============================================================

def build_train_normalizer(
    train_df,
):

    # One row per sample-state.
    # The second term is a duplicate of the same 10D state.
    unique = (
        train_df[
            train_df["term"]
            ==
            ACTIVE_TERMS[0]
        ]
        .sort_values(
            list(KEY_COLUMNS)
        )
        .reset_index(drop=True)
    )

    x = (
        unique[
            list(FEATURE_NAMES)
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mean = x.mean(
        axis=0
    )

    std = x.std(
        axis=0,
        ddof=0,
    )

    constant = (
        std <= 1.0e-12
    )

    safe_std = std.copy()

    safe_std[
        constant
    ] = 1.0

    if not np.isfinite(
        mean
    ).all():
        raise RuntimeError(
            "Non-finite TRAIN feature mean."
        )

    if not np.isfinite(
        safe_std
    ).all():
        raise RuntimeError(
            "Non-finite TRAIN feature std."
        )

    print()
    print(
        "========== TRAIN-ONLY FEATURE NORMALIZER =========="
    )

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        suffix = (
            " [CONSTANT]"
            if constant[i]
            else ""
        )

        print(
            f"{i:2d} {name:65s} "
            f"mean={mean[i]: .8e} "
            f"std={std[i]: .8e}"
            f"{suffix}"
        )

    return (
        mean,
        safe_std,
        constant,
    )


def normalized_features(
    df,
    mean,
    std,
):

    x = (
        df[
            list(FEATURE_NAMES)
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    z = (
        x - mean
    ) / std

    if not np.isfinite(
        z
    ).all():
        raise RuntimeError(
            "Non-finite normalized features."
        )

    return z.astype(
        np.float32
    )


def arm_features(
    full_z,
    arm,
):

    x = full_z.copy()

    if arm == "ParamUtility":

        x[
            :,
            :STATE_FEATURE_COUNT,
        ] = 0.0

        if not np.array_equal(
            x[
                :,
                :STATE_FEATURE_COUNT,
            ],
            np.zeros_like(
                x[
                    :,
                    :STATE_FEATURE_COUNT,
                ]
            ),
        ):
            raise RuntimeError(
                "ParamUtility state features "
                "are not exact zero."
            )

    elif arm == "StateUtility":

        pass

    else:
        raise ValueError(
            f"Unknown Utility arm: {arm}"
        )

    return x


# ============================================================
# Matched classifier initialization
# ============================================================

def matched_initial_state(
    term,
):

    term_index = ACTIVE_TERMS.index(
        term
    )

    init_seed = (
        SEED
        +
        1000
        *
        term_index
    )

    torch.manual_seed(
        init_seed
    )

    template = UtilityClassifier()

    if (
        parameter_count(template)
        !=
        EXPECTED_CLASSIFIER_NUMEL
    ):
        raise RuntimeError(
            "Utility classifier parameter count "
            f"must be {EXPECTED_CLASSIFIER_NUMEL}, "
            f"got {parameter_count(template)}"
        )

    return (
        copy.deepcopy(
            template.state_dict()
        ),
        init_seed,
    )


def assert_same_initialization(
    model_a,
    model_b,
):

    a = model_a.state_dict()
    b = model_b.state_dict()

    if set(a) != set(b):
        raise RuntimeError(
            "Matched classifier topology differs."
        )

    for key in a:

        if not torch.equal(
            a[key],
            b[key],
        ):
            raise RuntimeError(
                "Matched classifier initialization "
                f"differs at {key}."
            )

    print(
        "✅ ParamUtility / StateUtility "
        "initialization exactly matched"
    )


# ============================================================
# Training
# ============================================================

def train_classifier(
    *,
    model,
    x_train,
    y_train,
    epochs,
    device,
    schedule_seed,
):

    model = model.to(
        device
    )

    x = torch.from_numpy(
        x_train
    ).to(
        device=device,
        dtype=torch.float32,
    )

    y = torch.from_numpy(
        y_train.astype(
            np.float32
        )
    ).to(
        device=device,
        dtype=torch.float32,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        schedule_seed
    )

    history = []

    n = x.shape[0]

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        permutation = torch.randperm(
            n,
            generator=generator,
        )

        total_loss = 0.0
        seen = 0

        for start in range(
            0,
            n,
            CLASSIFIER_BATCH_SIZE,
        ):

            idx = permutation[
                start:
                start
                +
                CLASSIFIER_BATCH_SIZE
            ].to(device)

            logits = model(
                x[idx]
            )

            loss = criterion(
                logits,
                y[idx],
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            batch_n = idx.numel()

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                *
                batch_n
            )

            seen += batch_n

        epoch_loss = (
            total_loss
            /
            seen
        )

        history.append(
            epoch_loss
        )

        if (
            epoch == 1
            or epoch == epochs
            or epoch % 20 == 0
        ):

            print(
                f"    epoch "
                f"{epoch:03d}/{epochs} "
                f"loss={epoch_loss:.8f}"
            )

    return (
        model,
        history,
    )


@torch.no_grad()
def predict_probability(
    model,
    x,
    device,
):

    model.eval()

    x_tensor = torch.from_numpy(
        x
    ).to(
        device=device,
        dtype=torch.float32,
    )

    logits = model(
        x_tensor
    )

    probability = torch.sigmoid(
        logits
    )

    return (
        probability
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(SEED)

    if (
        args.device == "cuda"
        and
        not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    if args.debug_epochs is not None:

        if args.debug_epochs <= 0:
            raise ValueError(
                "--debug_epochs must be > 0."
            )

        epochs = args.debug_epochs
        formal = False

    else:

        epochs = FORMAL_EPOCHS
        formal = True

    print("=" * 110)
    print(
        "R3-3A UTILITY PREDICTABILITY AUDIT"
    )
    print("=" * 110)

    print(
        "Split axis        :",
        args.split_label,
    )
    print(
        "Classifier arms   :",
        ARMS,
    )
    print(
        "Utility terms     :",
        ACTIVE_TERMS,
    )
    print(
        "Feature dim       : 10"
    )
    print(
        "ParamUtility mask : dims 0..7 exact zero"
    )
    print(
        "StateUtility      : all 10 normalized features"
    )
    print(
        "Normalization     : TRAIN only"
    )
    print(
        "VAL role          : diagnostic AUC only"
    )
    print(
        "VAL selection     : NO"
    )
    print(
        "VAL threshold tune: NO"
    )
    print(
        "TEST access       : NO"
    )
    print(
        "Epochs            :",
        epochs,
        (
            "(FORMAL)"
            if formal
            else "(DEBUG)"
        ),
    )

    frozen = (
        load_and_audit_registry(
            args.split_label
        )
    )

    train_df = load_rows(
        frozen["train_entry"],
        args.split_label,
        "train",
    )

    val_df = load_rows(
        frozen["val_entry"],
        args.split_label,
        "val",
    )

    audit_term_feature_identity(
        train_df
    )

    audit_term_feature_identity(
        val_df
    )

    (
        feature_mean,
        feature_std,
        constant_mask,
    ) = build_train_normalizer(
        train_df
    )

    # --------------------------------------------------------
    # Interface audit for both matched arms.
    # --------------------------------------------------------

    probe = (
        train_df[
            train_df["term"]
            ==
            ACTIVE_TERMS[0]
        ]
        .head(32)
        .reset_index(drop=True)
    )

    probe_z = normalized_features(
        probe,
        feature_mean,
        feature_std,
    )

    param_probe = arm_features(
        probe_z,
        "ParamUtility",
    )

    state_probe = arm_features(
        probe_z,
        "StateUtility",
    )

    if not np.array_equal(
        param_probe[:, 8:],
        state_probe[:, 8:],
    ):
        raise RuntimeError(
            "Param dimensions differ between "
            "matched Utility arms."
        )

    if not np.array_equal(
        param_probe[:, :8],
        np.zeros_like(
            param_probe[:, :8]
        ),
    ):
        raise RuntimeError(
            "ParamUtility state mask failed."
        )

    print()
    print(
        "✅ ParamUtility state dims 0..7 "
        "are exact zero"
    )
    print(
        "✅ ParamUtility / StateUtility "
        "parameter dims 8..9 exactly match"
    )

    for term in ACTIVE_TERMS:

        init_state, _ = (
            matched_initial_state(
                term
            )
        )

        model_param = (
            UtilityClassifier()
        )

        model_state = (
            UtilityClassifier()
        )

        model_param.load_state_dict(
            copy.deepcopy(
                init_state
            )
        )

        model_state.load_state_dict(
            copy.deepcopy(
                init_state
            )
        )

        assert_same_initialization(
            model_param,
            model_state,
        )

    print(
        "✅ Utility classifier numel = 897 "
        "per term / per arm"
    )

    if args.audit_only:

        print()
        print(
            "✅ R3-3A PREDICTABILITY "
            "AUDIT-ONLY PASS"
        )
        print(
            "✅ Frozen TRAIN/VAL datasets verified"
        )
        print(
            "✅ Exact matched feature interface verified"
        )
        print(
            "✅ Exact matched classifier initialization verified"
        )
        print(
            "✅ No classifier training performed"
        )
        print(
            "✅ TEST access = NO"
        )

        return

    # ========================================================
    # TRAIN-only classifier fit.
    # VAL is diagnostic only.
    # ========================================================

    results = []

    checkpoint_root = resolve_path(
        (
            "checkpoints/r3_3_utility/"
            f"{args.split_label}"
        )
    )

    output_root = resolve_path(
        "outputs/tables/r3_3_utility"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for term_index, term in enumerate(
        ACTIVE_TERMS
    ):

        print()
        print("=" * 110)
        print(
            f"TERM: {term}"
        )
        print("=" * 110)

        train_term = (
            train_df[
                train_df["term"]
                ==
                term
            ]
            .reset_index(drop=True)
        )

        val_term = (
            val_df[
                val_df["term"]
                ==
                term
            ]
            .reset_index(drop=True)
        )

        y_train = (
            train_term["helps"]
            .to_numpy(
                dtype=np.int64
            )
        )

        y_val = (
            val_term["helps"]
            .to_numpy(
                dtype=np.int64
            )
        )

        full_train = (
            normalized_features(
                train_term,
                feature_mean,
                feature_std,
            )
        )

        full_val = (
            normalized_features(
                val_term,
                feature_mean,
                feature_std,
            )
        )

        init_state, init_seed = (
            matched_initial_state(
                term
            )
        )

        schedule_seed = (
            SEED
            +
            10000
            +
            term_index
        )

        initialized_models = {}

        for arm in ARMS:

            model = UtilityClassifier()

            model.load_state_dict(
                copy.deepcopy(
                    init_state
                )
            )

            initialized_models[
                arm
            ] = model

        assert_same_initialization(
            initialized_models[
                "ParamUtility"
            ],
            initialized_models[
                "StateUtility"
            ],
        )

        for arm in ARMS:

            print()
            print(
                f"--- {term} / {arm} ---"
            )

            x_train = arm_features(
                full_train,
                arm,
            )

            x_val = arm_features(
                full_val,
                arm,
            )

            model = initialized_models[
                arm
            ]

            model, history = (
                train_classifier(
                    model=model,
                    x_train=x_train,
                    y_train=y_train,
                    epochs=epochs,
                    device=device,
                    schedule_seed=(
                        schedule_seed
                    ),
                )
            )

            train_probability = (
                predict_probability(
                    model,
                    x_train,
                    device,
                )
            )

            val_probability = (
                predict_probability(
                    model,
                    x_val,
                    device,
                )
            )

            train_auc = (
                binary_roc_auc(
                    y_train,
                    train_probability,
                )
            )

            val_auc = (
                binary_roc_auc(
                    y_val,
                    val_probability,
                )
            )

            print(
                f"    TRAIN AUC = {train_auc:.8f}"
            )

            print(
                f"    VAL   AUC = {val_auc:.8f}"
            )

            result = {
                "split":
                    args.split_label,
                "term":
                    term,
                "arm":
                    arm,
                "formal":
                    formal,
                "epochs":
                    epochs,
                "seed":
                    SEED,
                "init_seed":
                    init_seed,
                "schedule_seed":
                    schedule_seed,
                "classifier_numel":
                    EXPECTED_CLASSIFIER_NUMEL,
                "train_rows":
                    int(len(y_train)),
                "val_rows":
                    int(len(y_val)),
                "train_positive_fraction":
                    float(
                        y_train.mean()
                    ),
                "val_positive_fraction":
                    float(
                        y_val.mean()
                    ),
                "final_train_loss":
                    float(
                        history[-1]
                    ),
                "train_auc":
                    train_auc,
                "val_auc":
                    val_auc,
            }

            results.append(
                result
            )

            suffix = (
                ""
                if formal
                else
                f"_debug{epochs}"
            )

            checkpoint_path = (
                checkpoint_root
                /
                (
                    f"r3_3a_"
                    f"{args.split_label}_"
                    f"{term}_"
                    f"{arm.lower()}_"
                    f"seed42"
                    f"{suffix}.pth"
                )
            )

            payload = {
                "stage":
                    "R3-3A",
                "role":
                    "UTILITY_PREDICTABILITY_CLASSIFIER",
                "split":
                    args.split_label,
                "term":
                    term,
                "arm":
                    arm,
                "formal":
                    formal,
                "seed":
                    SEED,
                "epochs":
                    epochs,
                "batch_size":
                    CLASSIFIER_BATCH_SIZE,
                "learning_rate":
                    LEARNING_RATE,
                "classifier_numel":
                    EXPECTED_CLASSIFIER_NUMEL,
                "feature_names":
                    list(
                        FEATURE_NAMES
                    ),
                "state_feature_count":
                    STATE_FEATURE_COUNT,
                "feature_mean":
                    feature_mean.tolist(),
                "feature_std":
                    feature_std.tolist(),
                "constant_feature_mask":
                    constant_mask.tolist(),
                "paramutility_semantics":
                    (
                        "normalize using shared "
                        "TRAIN-only statistics, "
                        "then set dims 0..7 "
                        "to exact zero"
                    ),
                "stateutility_semantics":
                    (
                        "normalize using shared "
                        "TRAIN-only statistics; "
                        "all 10 dimensions visible"
                    ),
                "model_state_dict":
                    {
                        k:
                            v.detach().cpu()
                        for k, v
                        in model.state_dict().items()
                    },
                "train_auc":
                    train_auc,
                "val_auc":
                    val_auc,
                "final_train_loss":
                    float(
                        history[-1]
                    ),
                "dataset_registry_sha256":
                    frozen[
                        "registry_sha"
                    ],
                "r3_3_contract_sha256":
                    frozen[
                        "r3_3_sha"
                    ],
                "train_rows_sha256":
                    frozen[
                        "train_entry"
                    ][
                        "rows_sha256"
                    ],
                "val_rows_sha256":
                    frozen[
                        "val_entry"
                    ][
                        "rows_sha256"
                    ],
                "val_role":
                    "diagnostic AUC only",
                "val_model_selection":
                    False,
                "val_threshold_tuning":
                    False,
                "val_temperature_tuning":
                    False,
                "test_accessed":
                    False,
            }

            torch.save(
                payload,
                checkpoint_path,
            )

            print(
                f"    saved: {checkpoint_path}"
            )

    result_df = pd.DataFrame(
        results
    )

    suffix = (
        ""
        if formal
        else
        f"_debug{epochs}"
    )

    csv_path = (
        output_root
        /
        (
            f"r3_3a_predictability_"
            f"{args.split_label}_"
            f"seed42"
            f"{suffix}_summary.csv"
        )
    )

    json_path = (
        output_root
        /
        (
            f"r3_3a_predictability_"
            f"{args.split_label}_"
            f"seed42"
            f"{suffix}_summary.json"
        )
    )

    result_df.to_csv(
        csv_path,
        index=False,
    )

    contrasts = {}

    for term in ACTIVE_TERMS:

        sub = result_df[
            result_df["term"]
            ==
            term
        ]

        param_auc = float(
            sub[
                sub["arm"]
                ==
                "ParamUtility"
            ]["val_auc"].iloc[0]
        )

        state_auc = float(
            sub[
                sub["arm"]
                ==
                "StateUtility"
            ]["val_auc"].iloc[0]
        )

        contrasts[term] = {
            "paramutility_val_auc":
                param_auc,
            "stateutility_val_auc":
                state_auc,
            "state_minus_param_auc":
                state_auc
                -
                param_auc,
        }

    summary = {
        "stage":
            "R3-3A",
        "role":
            "UTILITY_PREDICTABILITY_AUDIT",
        "split":
            args.split_label,
        "formal":
            formal,
        "epochs":
            epochs,
        "seed":
            SEED,
        "dataset_registry_sha256":
            frozen[
                "registry_sha"
            ],
        "r3_3_contract_sha256":
            frozen[
                "r3_3_sha"
            ],
        "feature_normalization":
            "shared TRAIN-only",
        "classifier_architecture":
            "Linear(10,32)-GELU-Linear(32,16)-GELU-Linear(16,1)",
        "classifier_numel":
            EXPECTED_CLASSIFIER_NUMEL,
        "matched_initialization":
            True,
        "matched_batch_schedule":
            True,
        "paramutility_state_dims_exact_zero":
            True,
        "val_role":
            "diagnostic AUC only",
        "val_selection":
            False,
        "test_accessed":
            False,
        "contrasts":
            contrasts,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 110)
    print(
        "R3-3A PREDICTABILITY SUMMARY"
    )
    print("=" * 110)

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print(
        "========== STATE INFORMATION AUC CONTRAST =========="
    )

    for term in ACTIVE_TERMS:

        item = contrasts[
            term
        ]

        print(
            f"{term}: "
            f"ParamUtility="
            f"{item['paramutility_val_auc']:.8f} | "
            f"StateUtility="
            f"{item['stateutility_val_auc']:.8f} | "
            f"State-Param="
            f"{item['state_minus_param_auc']:+.8f}"
        )

    print()
    print(
        "NOTE: VAL AUC is diagnostic only."
    )
    print(
        "No arm/feature/threshold/model selection "
        "is permitted from this result."
    )
    print(
        "TEST access = NO"
    )

    print()
    print(
        f"Summary CSV : {csv_path}"
    )
    print(
        f"Summary JSON: {json_path}"
    )


if __name__ == "__main__":
    main()
