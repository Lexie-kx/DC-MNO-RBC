from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from training.normalization import FieldWiseNormalizer

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)

from evaluation.evaluate_cross_param_rollout_r3_1 import (
    RolloutDataset,
    assert_same_m6,
    build_m6,
    load_locked_contract,
    resolve_path,
    sha256_file,
)

from evaluation.evaluate_cross_param_rollout_r3_2 import (
    LOCKED_RESOURCES,
    R3_0_CONTRACT,
    EXPECTED_R3_0_CONTRACT_SHA256,
    audit_locked_file,
    build_r3_2,
    load_and_audit_registry,
)


# ============================================================
# Frozen R3-3 identities
# ============================================================

R3_3_CONTRACT = (
    "configs/r3/r3_3_utility_contract.json"
)

EXPECTED_R3_3_CONTRACT_SHA256 = (
    "426299fcce724d1e74490887c39ea583"
    "07f2180bf464c4d41ee6ee170b6c4af8"
)

R3_2_CLOSEOUT_REGISTRY = (
    "configs/r3/r3_2_closeout_registry.json"
)

EXPECTED_R3_2_CLOSEOUT_SHA256 = (
    "45013b8d2c960e13419b02578f10af6"
    "21b7d59917c33a95be66925a00622ca5c"
)

ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)

EXPECTED_TARGET_INDEX = {
    "buoyancy_advection": 0,
    "buoyancy_forcing": 2,
}

TARGET_FIELD = {
    "buoyancy_advection": "buoyancy",
    "buoyancy_forcing": "u_y",
}


# ============================================================
# CLI
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "R3-3A TRAIN/VAL-only per-term Utility target "
            "generator from frozen R3-2b StateParam."
        )
    )

    p.add_argument(
        "--split_label",
        required=True,
        choices=("unseen_pr", "unseen_ra"),
    )

    p.add_argument(
        "--data_split",
        required=True,
        choices=("train", "val"),
        help="TEST is intentionally forbidden.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="DEBUG only.",
    )

    p.add_argument(
        "--audit_only",
        action="store_true",
        help=(
            "Audit frozen contracts/checkpoints/interfaces only. "
            "No TRAIN/VAL rollout dataset is constructed."
        ),
    )

    p.add_argument(
        "--output_dir",
        default="outputs/tables/r3_3_utility",
    )

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class IndexedDataset(Dataset):
    """
    Preserve the exact frozen RolloutDataset semantics while
    also returning its dataset index for provenance.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        x0_phys, future_phys, param = self.base[idx]

        return (
            x0_phys,
            future_phys,
            param,
            int(idx),
        )


def rms_error_2d(pred, target):
    """
    Per-sample RMS error.
    pred/target: [B, X, Y]
    returns: [B]
    """

    return torch.sqrt(
        torch.mean(
            (pred - target) ** 2,
            dim=(-2, -1),
        )
    )


def spatial_rms_2d(x):

    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
    )


# ============================================================
# Frozen parent / provenance audit
# ============================================================

def build_and_audit_parent(
    *,
    split_label,
    device,
):

    resource = LOCKED_RESOURCES[split_label]

    split_path = resolve_path(
        resource["split"]
    )

    stats_path = resolve_path(
        resource["stats"]
    )

    m6_path = resolve_path(
        resource["m6"]
    )

    r3_0_contract_path = resolve_path(
        R3_0_CONTRACT
    )

    r3_3_contract_path = resolve_path(
        R3_3_CONTRACT
    )

    closeout_path = resolve_path(
        R3_2_CLOSEOUT_REGISTRY
    )

    # --------------------------------------------------------
    # Locked resource hashes
    # --------------------------------------------------------

    audit_locked_file(
        split_path,
        resource["split_sha256"],
        f"{split_label} split",
    )

    audit_locked_file(
        stats_path,
        resource["stats_sha256"],
        f"{split_label} stats",
    )

    audit_locked_file(
        m6_path,
        resource["m6_sha256"],
        f"{split_label} M6",
    )

    r3_0_sha = audit_locked_file(
        r3_0_contract_path,
        EXPECTED_R3_0_CONTRACT_SHA256,
        "R3-0 contract",
    )

    r3_3_sha = audit_locked_file(
        r3_3_contract_path,
        EXPECTED_R3_3_CONTRACT_SHA256,
        "R3-3 contract",
    )

    closeout_sha = audit_locked_file(
        closeout_path,
        EXPECTED_R3_2_CLOSEOUT_SHA256,
        "R3-2 closeout registry",
    )

    r3_3_contract = load_json(
        r3_3_contract_path
    )

    if r3_3_contract.get("stage") != "R3-3":
        raise RuntimeError(
            "Unexpected R3-3 contract stage."
        )

    # --------------------------------------------------------
    # Frozen R3-2 pre-TEST checkpoint registry
    # --------------------------------------------------------

    (
        registry,
        entries,
        registry_path,
        r3_2_contract_path,
        r3_2_contract_sha,
    ) = load_and_audit_registry(
        split_label
    )

    stateparam_entry = entries["R3-2b"]

    stateparam_path = resolve_path(
        stateparam_entry["checkpoint"]
    )

    stateparam_sha = sha256_file(
        stateparam_path
    )

    # --------------------------------------------------------
    # Fixed R3-0 capacities
    # --------------------------------------------------------

    _, alpha_map = load_locked_contract(
        r3_0_contract_path
    )

    metadata = build_rbc_canonical_metadata(
        stats_path
    )

    # --------------------------------------------------------
    # Bare M6 + frozen R3-2b parent
    # --------------------------------------------------------

    m6, _ = build_m6(
        m6_path,
        device,
    )

    stateparam, payload = build_r3_2(
        adaptation_mode="stateparam",
        checkpoint_path=stateparam_path,
        expected_entry=stateparam_entry,
        metadata=metadata,
        alpha_map=alpha_map,
        r3_2_contract_sha=r3_2_contract_sha,
        split_label=split_label,
        device=device,
    )

    assert_same_m6(
        m6,
        stateparam,
        "R3-3 parent / R3-2b-StateParam",
    )

    # R3-3A is audit-only with respect to the parent.
    for parameter in stateparam.parameters():
        parameter.requires_grad_(False)

    stateparam.eval()
    stateparam.m6.eval()

    # --------------------------------------------------------
    # Synthetic interface audit
    # --------------------------------------------------------

    x = torch.zeros(
        2,
        16,
        256,
        64,
        dtype=torch.float32,
        device=device,
    )

    params = torch.tensor(
        [
            [6.0, -0.3010300],
            [8.0,  0.3010300],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        output, info = stateparam(
            x,
            params=params,
            return_components=True,
        )

    required = {
        "base_delta_norm",
        "physics_residual_norm",
        "compiled_terms",
        "effective_gate_values",
        "full_conditioner_features",
        "term_corrections",
    }

    missing = required - set(info)

    if missing:
        raise RuntimeError(
            f"Missing R3-3A parent components: {missing}"
        )

    if tuple(
        info["full_conditioner_features"].shape
    ) != (2, 10):
        raise RuntimeError(
            "R3-3A requires exact [B,10] "
            "R3-2 frozen features."
        )

    for term in ACTIVE_TERMS:

        if term not in info["term_corrections"]:
            raise RuntimeError(
                f"Missing term correction: {term}"
            )

        correction = info[
            "term_corrections"
        ][term]

        if tuple(correction.shape) != (
            2,
            256,
            64,
        ):
            raise RuntimeError(
                f"{term}: correction must be [B,X,Y], "
                f"got {tuple(correction.shape)}"
            )

        actual_index = int(
            stateparam.TERM_TO_OUTPUT_INDEX[term]
        )

        expected_index = (
            EXPECTED_TARGET_INDEX[term]
        )

        if actual_index != expected_index:
            raise RuntimeError(
                f"{term}: target output index changed. "
                f"expected={expected_index}, "
                f"actual={actual_index}"
            )

        residual_channel = (
            info["physics_residual_norm"][
                :,
                actual_index,
                :,
                :,
            ]
        )

        max_abs = float(
            torch.max(
                torch.abs(
                    residual_channel - correction
                )
            ).cpu()
        )

        if max_abs > 1.0e-7:
            raise RuntimeError(
                f"{term}: correction routing mismatch "
                f"max_abs={max_abs:.12e}"
            )

    reconstructed = (
        info["base_delta_norm"]
        +
        info["physics_residual_norm"]
    )

    max_output_diff = float(
        torch.max(
            torch.abs(
                output - reconstructed
            )
        ).cpu()
    )

    if max_output_diff > 1.0e-7:
        raise RuntimeError(
            "R3-2b output reconstruction failed: "
            f"{max_output_diff:.12e}"
        )

    print(
        "✅ R3-3A frozen parent/interface audit PASS"
    )

    return {
        "split_path": split_path,
        "stats_path": stats_path,
        "m6_path": m6_path,
        "stateparam_path": stateparam_path,
        "stateparam_sha": stateparam_sha,
        "stateparam": stateparam,
        "stateparam_payload": payload,
        "registry_path": registry_path,
        "r3_2_contract_path": r3_2_contract_path,
        "r3_2_contract_sha": r3_2_contract_sha,
        "r3_0_sha": r3_0_sha,
        "r3_3_sha": r3_3_sha,
        "closeout_sha": closeout_sha,
        "alpha_map": alpha_map,
    }


# ============================================================
# Utility-row generation
# ============================================================

@torch.no_grad()
def generate_rows(
    *,
    model,
    loader,
    base_dataset,
    split_items,
    split_label,
    data_split,
    normalizer,
    device,
):

    trajectory_lookup = {}

    for item in split_items:

        group = item["group"]

        for local_idx, original_idx in enumerate(
            item["trajectories"]
        ):
            trajectory_lookup[
                (group, local_idx)
            ] = int(original_idx)

    rows = []

    feature_names = tuple(
        model.FEATURE_NAMES
    )

    if len(feature_names) != 10:
        raise RuntimeError(
            "R3-3A requires exactly 10 frozen features."
        )

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        (
            x0_phys,
            future_phys,
            param,
            sample_ids,
        ) = batch

        x0_phys = x0_phys.to(
            device,
            non_blocking=True,
        )

        future_phys = future_phys.to(
            device,
            non_blocking=True,
        )

        param = param.to(
            device,
            non_blocking=True,
        )

        x_norm = normalizer.normalize_x(
            x0_phys
        )

        for step in range(1, 5):

            current_norm = x_norm[
                :,
                -4:,
                :,
                :,
            ]

            pred_delta_norm, info = model(
                x_norm,
                params=param,
                return_components=True,
            )

            full_next_norm = (
                current_norm
                +
                pred_delta_norm
            )

            gt_next_phys = future_phys[
                :,
                step - 1,
                :,
                :,
                :,
            ]

            # Exact project normalization:
            # (y - mean) / (std + eps)
            gt_next_norm = (
                normalizer.normalize_y(
                    gt_next_phys
                )
            )

            features = (
                info[
                    "full_conditioner_features"
                ]
                .detach()
                .cpu()
            )

            alphas = {
                term: (
                    info[
                        "effective_gate_values"
                    ][term]
                    .detach()
                    .cpu()
                )
                for term in ACTIVE_TERMS
            }

            term_arrays = {}

            for term in ACTIVE_TERMS:

                output_index = int(
                    model.TERM_TO_OUTPUT_INDEX[
                        term
                    ]
                )

                correction = (
                    info[
                        "term_corrections"
                    ][term]
                )

                # ------------------------------------------------
                # Utility errors are accumulated in float64.
                #
                # The frozen R3-2b parent remains float32.
                # Only the SAME-state counterfactual arithmetic and
                # error accumulation are promoted to float64 so that
                # very small nonzero term corrections are not erased
                # by float32 subtraction.
                # ------------------------------------------------

                full_target = (
                    full_next_norm[
                        :,
                        output_index,
                        :,
                        :,
                    ]
                    .double()
                )

                gt_target = (
                    gt_next_norm[
                        :,
                        output_index,
                        :,
                        :,
                    ]
                    .double()
                )

                correction64 = (
                    correction.double()
                )

                term_off_target = (
                    full_target
                    -
                    correction64
                )

                full_error = rms_error_2d(
                    full_target,
                    gt_target,
                )

                off_error = rms_error_2d(
                    term_off_target,
                    gt_target,
                )

                gain = (
                    off_error
                    -
                    full_error
                )

                relative_gain = (
                    gain
                    /
                    (
                        off_error
                        +
                        1.0e-12
                    )
                )

                helps = (
                    gain > 0.0
                ).to(torch.int64)

                correction_rms = (
                    spatial_rms_2d(
                        correction
                    )
                )

                term_arrays[term] = {
                    "full_error":
                        full_error.cpu(),
                    "off_error":
                        off_error.cpu(),
                    "gain":
                        gain.cpu(),
                    "relative_gain":
                        relative_gain.cpu(),
                    "helps":
                        helps.cpu(),
                    "correction_rms":
                        correction_rms.cpu(),
                }

            # ----------------------------------------------------
            # Row-level provenance
            # ----------------------------------------------------

            for i, dataset_idx in enumerate(
                sample_ids.tolist()
            ):

                meta = base_dataset.index[
                    int(dataset_idx)
                ]

                group = meta["group"]

                local_traj = int(
                    meta["traj"]
                )

                original_traj = (
                    trajectory_lookup[
                        (
                            group,
                            local_traj,
                        )
                    ]
                )

                base_row = {
                    "split":
                        split_label,
                    "data_split":
                        data_split,
                    "sample_index":
                        int(dataset_idx),
                    "group":
                        group,
                    "trajectory":
                        original_traj,
                    "local_trajectory":
                        local_traj,
                    "t0":
                        int(meta["t0"]),
                    "step":
                        int(step),
                }

                for feature_idx, feature_name in enumerate(
                    feature_names
                ):
                    base_row[
                        feature_name
                    ] = float(
                        features[
                            i,
                            feature_idx,
                        ].item()
                    )

                for term in ACTIVE_TERMS:

                    arr = term_arrays[term]

                    row = dict(base_row)

                    row.update(
                        {
                            "term":
                                term,
                            "target_field":
                                TARGET_FIELD[term],
                            "alpha":
                                float(
                                    alphas[term][i].item()
                                ),
                            "term_correction_rms":
                                float(
                                    arr[
                                        "correction_rms"
                                    ][i].item()
                                ),
                            "full_error":
                                float(
                                    arr[
                                        "full_error"
                                    ][i].item()
                                ),
                            "term_off_error":
                                float(
                                    arr[
                                        "off_error"
                                    ][i].item()
                                ),
                            "gain":
                                float(
                                    arr[
                                        "gain"
                                    ][i].item()
                                ),
                            "relative_gain":
                                float(
                                    arr[
                                        "relative_gain"
                                    ][i].item()
                                ),
                            "helps":
                                int(
                                    arr[
                                        "helps"
                                    ][i].item()
                                ),
                        }
                    )

                    rows.append(row)

            # IMPORTANT:
            # only FULL frozen R3-2b prediction advances.
            x_norm = torch.cat(
                [
                    x_norm[
                        :,
                        4:,
                        :,
                        :,
                    ],
                    full_next_norm,
                ],
                dim=1,
            )

        if (
            batch_idx % 20 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"  processed batch "
                f"{batch_idx}/{len(loader)}"
            )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.seed != 42:
        raise ValueError(
            "Formal R3-3A is locked to seed=42."
        )

    if args.batch_size != 4:
        raise ValueError(
            "Formal R3-3A is locked to batch_size=4."
        )

    # Hard guard: TEST is not a legal CLI choice.
    if args.data_split not in (
        "train",
        "val",
    ):
        raise RuntimeError(
            "R3-3A TEST access is forbidden."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    print("=" * 110)
    print(
        "R3-3A PER-TERM UTILITY TARGET "
        "AND PREDICTABILITY AUDIT"
    )
    print("=" * 110)

    print("Split axis :", args.split_label)
    print("Data split :", args.data_split)
    print("Seed       :", args.seed)
    print("Device     :", device)
    print("Parent     : frozen R3-2b-StateParam")
    print("Rollout    : H4 free-autoregressive")
    print(
        "Counterfactual: same-state one-step "
        "term-off; NEVER fed back"
    )
    print(
        "GT usage   : utility label construction only"
    )
    print("TEST access: NO")

    parent = build_and_audit_parent(
        split_label=args.split_label,
        device=device,
    )

    print()
    print("========== FROZEN PROVENANCE ==========")
    print(
        "R3_0_CONTRACT_SHA256:",
        parent["r3_0_sha"],
    )
    print(
        "R3_2_CONTRACT_SHA256:",
        parent["r3_2_contract_sha"],
    )
    print(
        "R3_2_CLOSEOUT_SHA256:",
        parent["closeout_sha"],
    )
    print(
        "R3_3_CONTRACT_SHA256:",
        parent["r3_3_sha"],
    )
    print(
        "R3_2B_CHECKPOINT_SHA256:",
        parent["stateparam_sha"],
    )

    if args.audit_only:

        print()
        print(
            "✅ R3-3A AUDIT-ONLY PASS"
        )
        print(
            "✅ Frozen parent/contracts/interfaces verified"
        )
        print(
            "✅ No TRAIN/VAL rollout dataset constructed"
        )
        print(
            "✅ TEST not accessed"
        )

        return

    # ========================================================
    # TRAIN / VAL access starts here. TEST is impossible here.
    # ========================================================

    split_config = load_json(
        parent["split_path"]
    )

    split_items = split_config[
        args.data_split
    ]

    # --------------------------------------------------------
    # Reuse the frozen R3-1 RolloutDataset implementation,
    # but suppress its legacy hard-coded "TEST" print text.
    #
    # The actual data passed here are explicitly
    # split_config[train] or split_config[val].
    # --------------------------------------------------------

    legacy_dataset_stdout = io.StringIO()

    with redirect_stdout(
        legacy_dataset_stdout
    ):
        base_dataset = RolloutDataset(
            split_config=split_items,
            max_horizon=4,
            max_samples=args.max_samples,
        )

    print(
        f"📦 Loading R3-3A "
        f"{args.data_split.upper()} rollout data: "
        f"{len(split_items)} group batches"
    )

    print(
        f"✅ {args.data_split.upper()} "
        f"rollout samples: "
        f"{len(base_dataset)}"
    )

    dataset = IndexedDataset(
        base_dataset
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    normalizer = (
        FieldWiseNormalizer(
            parent["stats_path"]
        )
        .to(device)
    )

    print()
    print("========== R3-3A DATA CONTRACT ==========")
    print(
        "Dataset split:",
        args.data_split,
    )
    print(
        "H4 windows:",
        len(base_dataset),
    )
    print(
        "Temporal stride: 1 "
        "(all legal windows from frozen RolloutDataset)"
    )
    print(
        "Expected utility rows:",
        len(base_dataset) * 4 * 2,
    )

    rows_df = generate_rows(
        model=parent["stateparam"],
        loader=loader,
        base_dataset=base_dataset,
        split_items=split_items,
        split_label=args.split_label,
        data_split=args.data_split,
        normalizer=normalizer,
        device=device,
    )

    expected_rows = (
        len(base_dataset)
        *
        4
        *
        len(ACTIVE_TERMS)
    )

    if len(rows_df) != expected_rows:
        raise RuntimeError(
            "Utility row count mismatch: "
            f"expected={expected_rows}, "
            f"actual={len(rows_df)}"
        )

    # Exact binary-label reproduction.
    reproduced = (
        rows_df["gain"] > 0.0
    ).astype(int)

    mismatch = int(
        (
            reproduced
            !=
            rows_df["helps"]
        ).sum()
    )

    if mismatch != 0:
        raise RuntimeError(
            "Utility label reproduction mismatch: "
            f"{mismatch}"
        )

    output_dir = resolve_path(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    debug_suffix = (
        f"_debug{args.max_samples}"
        if args.max_samples is not None
        else
        ""
    )

    prefix = (
        f"r3_3a_utility_"
        f"{args.split_label}_"
        f"{args.data_split}_"
        f"seed{args.seed}"
        f"{debug_suffix}"
    )

    rows_path = os.path.join(
        output_dir,
        f"{prefix}_rows.csv",
    )

    summary_path = os.path.join(
        output_dir,
        f"{prefix}_summary.csv",
    )

    metadata_path = os.path.join(
        output_dir,
        f"{prefix}_metadata.json",
    )

    rows_df.to_csv(
        rows_path,
        index=False,
    )

    summary_df = (
        rows_df
        .groupby(
            [
                "term",
                "step",
            ],
            as_index=False,
        )
        .agg(
            n=("helps", "size"),
            helps_fraction=("helps", "mean"),
            gain_mean=("gain", "mean"),
            gain_median=("gain", "median"),
            relative_gain_mean=(
                "relative_gain",
                "mean",
            ),
            full_error_mean=(
                "full_error",
                "mean",
            ),
            term_off_error_mean=(
                "term_off_error",
                "mean",
            ),
            correction_rms_mean=(
                "term_correction_rms",
                "mean",
            ),
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    metadata = {
        "stage":
            "R3-3A",
        "role":
            "UTILITY_TARGET_AND_PREDICTABILITY_AUDIT",
        "split":
            args.split_label,
        "data_split":
            args.data_split,
        "seed":
            args.seed,
        "batch_size":
            args.batch_size,
        "max_horizon":
            4,
        "test_accessed":
            False,
        "parent":
            "R3-2b-StateParam",
        "parent_checkpoint":
            parent["stateparam_path"],
        "parent_checkpoint_sha256":
            parent["stateparam_sha"],
        "r3_3_contract_sha256":
            parent["r3_3_sha"],
        "r3_2_closeout_sha256":
            parent["closeout_sha"],
        "r3_2_contract_sha256":
            parent["r3_2_contract_sha"],
        "r3_0_contract_sha256":
            parent["r3_0_sha"],
        "utility_definition":
            "gain = E_term_off_target - E_full_target",
        "binary_label":
            "helps = 1 iff gain > 0",
        "relative_gain":
            "gain / (E_term_off_target + 1e-12)",
        "error_space":
            "normalized target field",
        "utility_error_accumulation_dtype":
            "float64",
        "parent_inference_dtype":
            "float32",
        "gt_usage":
            "label construction only",
        "counterfactual":
            (
                "same frozen parent state; "
                "suppress only audited term; "
                "counterfactual never fed back"
            ),
        "rollout_advance":
            "full frozen R3-2b prediction only",
        "feature_names":
            list(
                parent[
                    "stateparam"
                ].FEATURE_NAMES
            ),
        "rows":
            int(len(rows_df)),
        "label_mismatch_count":
            mismatch,
        "outputs": {
            "rows":
                rows_path,
            "summary":
                summary_path,
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("========== UTILITY SUMMARY ==========")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "✅ Utility label reproduction mismatch = 0"
    )
    print(
        f"✅ Rows saved: {rows_path}"
    )
    print(
        f"✅ Summary saved: {summary_path}"
    )
    print(
        f"✅ Metadata saved: {metadata_path}"
    )
    print(
        "✅ TEST access: NO"
    )


if __name__ == "__main__":
    main()
