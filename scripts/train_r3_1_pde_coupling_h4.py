from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from datasets.rbc_dataset import RBCDataset
from training.metrics import FieldWiseRelativeL2Loss

from models.operators.fno2d_r3_pde_coupling import (
    R3PDECouplingFNO2d,
)

from physics.canonical_metadata import (
    build_rbc_canonical_metadata,
)

from scripts.train_m6_fieldwise_encoder_h4 import (
    make_rollout_weights,
)


R3_CONTRACT_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "r3",
    "r3_0_architecture_contract.json",
)

EXPECTED_ACTIVE_TERMS = (
    "buoyancy_advection",
    "buoyancy_forcing",
)

EXPECTED_TRAINABLE_NUMEL = 2
EXPECTED_FROZEN_M6_TENSORS = 38


# ============================================================
# Dataset
# ============================================================

class R3MultiStepParamDataset(Dataset):
    """
    Shared dataset wrapper for BOTH R3-1 arms.

    Returns:
        context_norm: [4,4,H,W]
        y_seq_norm:   [S,4,H,W]
        param:        [2]

    Both naive and canonical arms receive exactly the same
    dataset representation and parameter tensor.
    """

    def __init__(
        self,
        base_dataset,
        context_length=4,
    ):
        self.base_dataset = base_dataset
        self.context_length = int(
            context_length
        )

    def __len__(self):
        return len(self.base_dataset)

    def _parse_sample(
        self,
        sample,
    ):
        x_norm = None
        y_seq_norm = None
        param = None

        if isinstance(sample, dict):

            for key in (
                "x_norm",
                "x",
                "context",
                "context_norm",
            ):
                if key in sample:
                    x_norm = sample[key]
                    break

            for key in (
                "y_seq_norm",
                "y_seq",
                "target_sequence",
                "sequence",
            ):
                if key in sample:
                    y_seq_norm = sample[key]
                    break

            for key in (
                "param",
                "params",
                "parameter",
                "parameters",
            ):
                if key in sample:
                    param = sample[key]
                    break

        elif isinstance(
            sample,
            (tuple, list),
        ):

            for obj in sample:

                if torch.is_tensor(obj):

                    if (
                        obj.ndim == 3
                        and
                        obj.shape[0]
                        ==
                        self.context_length * 4
                    ):
                        x_norm = obj

                    elif (
                        obj.ndim == 4
                        and
                        obj.shape[1] == 4
                    ):
                        y_seq_norm = obj

                    elif (
                        obj.ndim == 1
                        and
                        obj.numel() == 2
                    ):
                        param = obj

                elif (
                    isinstance(
                        obj,
                        (tuple, list),
                    )
                    and len(obj) == 2
                ):
                    try:
                        maybe_param = (
                            torch.tensor(
                                obj,
                                dtype=torch.float32,
                            )
                        )

                        if (
                            maybe_param.ndim == 1
                            and
                            maybe_param.numel() == 2
                        ):
                            param = maybe_param

                    except Exception:
                        pass

        else:
            raise TypeError(
                "Unsupported RBCDataset sample type: "
                f"{type(sample)}"
            )

        if (
            x_norm is None
            or y_seq_norm is None
            or param is None
        ):
            raise RuntimeError(
                "Could not parse "
                "x_norm / y_seq_norm / param."
            )

        if not torch.is_tensor(param):
            param = torch.tensor(
                param,
                dtype=torch.float32,
            )

        return (
            x_norm,
            y_seq_norm,
            param.float(),
        )

    def __getitem__(
        self,
        idx,
    ):
        sample = self.base_dataset[idx]

        (
            x_norm,
            y_seq_norm,
            param,
        ) = self._parse_sample(
            sample
        )

        c_total, h, w = x_norm.shape

        expected_channels = (
            self.context_length * 4
        )

        if c_total != expected_channels:
            raise RuntimeError(
                "Unexpected context channels: "
                f"expected={expected_channels}, "
                f"actual={c_total}"
            )

        context_norm = x_norm.view(
            self.context_length,
            4,
            h,
            w,
        )

        return (
            context_norm,
            y_seq_norm,
            param,
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "R3-1 shared matched trainer: "
            "naive vs canonical dimension-valid "
            "PDE sparse coupling."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
    )

    parser.add_argument(
        "--stats",
        required=True,
    )

    parser.add_argument(
        "--m6_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--split_label",
        required=True,
        choices=(
            "unseen_pr",
            "unseen_ra",
        ),
    )

    parser.add_argument(
        "--representation_mode",
        required=True,
        choices=(
            "naive",
            "canonical",
        ),
    )

    parser.add_argument(
        "--run_name",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # R3-0C per-term fixed capacity bounds.
    #
    # These are intentionally REQUIRED and have no defaults.
    #
    # Formal values will be locked only after the TRAIN-only
    # unseen-Pr + unseen-Ra support audits.
    #
    # For a matched R3-1a / R3-1b experiment, BOTH arms must
    # receive exactly the same pair of values.
    # --------------------------------------------------------

    parser.add_argument(
        "--alpha_max_buoyancy_advection",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--alpha_max_buoyancy_forcing",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/r3_1",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run model/data/interface audits only. "
            "No optimization and no checkpoint saving."
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility / provenance
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def extract_m6_state(checkpoint):

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and
        "state_dict" in checkpoint
    ):
        return checkpoint[
            "state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and
        "model" in checkpoint
        and
        isinstance(
            checkpoint["model"],
            dict,
        )
    ):
        return checkpoint["model"]

    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate M6 state_dict "
        "inside checkpoint."
    )


def verify_contract():
    if not os.path.exists(
        R3_CONTRACT_PATH
    ):
        raise FileNotFoundError(
            R3_CONTRACT_PATH
        )

    with open(
        R3_CONTRACT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        contract = json.load(f)

    if contract.get("stage") != "R3-0":
        raise RuntimeError(
            "Unexpected R3 contract stage."
        )

    active = tuple(
        name
        for name, cfg
        in contract[
            "active_routing"
        ].items()
        if cfg["active"]
    )

    if set(active) != set(
        EXPECTED_ACTIVE_TERMS
    ):
        raise RuntimeError(
            "R3 active routing differs "
            "from frozen R3-0 contract."
        )

    disabled = contract[
        "disabled_in_r3_1"
    ]

    if not all(
        disabled.values()
    ):
        raise RuntimeError(
            "R3-1 disabled-mechanism "
            "contract has changed."
        )

    return contract


# ============================================================
# H4 free-autoregressive objective
# ============================================================

def autoregressive_multistep_loss_r3(
    model,
    context_norm,
    y_seq_norm,
    param,
    criterion,
    rollout_weights,
):
    """
    Exact R3 continuation of the audited M10 H4 protocol.

    No teacher forcing after initial history.
    """

    (
        batch_size,
        context_len,
        channels,
        h,
        w,
    ) = context_norm.shape

    rollout_steps = (
        y_seq_norm.shape[1]
    )

    if context_len != 4:
        raise RuntimeError(
            f"Expected context_len=4, "
            f"got {context_len}"
        )

    if channels != 4:
        raise RuntimeError(
            f"Expected 4 fields, "
            f"got {channels}"
        )

    if rollout_steps != len(
        rollout_weights
    ):
        raise RuntimeError(
            "rollout_steps / weights mismatch."
        )

    if (
        param.ndim != 2
        or param.shape[1] != 2
    ):
        raise RuntimeError(
            "param must be [B,2] "
            "= [log10(Ra),log10(Pr)]."
        )

    context = context_norm

    loss_total = 0.0
    step_losses = []

    for step in range(
        rollout_steps
    ):

        model_input = context.reshape(
            batch_size,
            context_len * channels,
            h,
            w,
        )

        pred_delta_norm = model(
            model_input,
            params=param,
        )

        current_state_norm = (
            context[:, -1]
        )

        pred_next_norm = (
            current_state_norm
            +
            pred_delta_norm
        )

        gt_next_norm = (
            y_seq_norm[:, step]
        )

        loss_step = criterion(
            pred_next_norm,
            gt_next_norm,
        )

        loss_total = (
            loss_total
            +
            rollout_weights[step]
            *
            loss_step
        )

        step_losses.append(
            loss_step.detach()
        )

        context = torch.cat(
            [
                context[:, 1:],
                pred_next_norm.unsqueeze(1),
            ],
            dim=1,
        )

    loss_total = (
        loss_total
        /
        rollout_weights.sum()
    )

    return (
        loss_total,
        step_losses,
    )


def evaluate(
    model,
    loader,
    criterion,
    rollout_weights,
    device,
    max_batches=None,
):
    model.eval()
    model.m6.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader
        ):

            if (
                max_batches is not None
                and
                batch_idx >= max_batches
            ):
                break

            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = (
                context_norm.to(device)
            )

            y_seq_norm = (
                y_seq_norm.to(device)
            )

            param = param.to(device)

            (
                loss,
                _,
            ) = (
                autoregressive_multistep_loss_r3(
                    model=model,
                    context_norm=context_norm,
                    y_seq_norm=y_seq_norm,
                    param=param,
                    criterion=criterion,
                    rollout_weights=(
                        rollout_weights
                    ),
                )
            )

            bs = context_norm.shape[0]

            total_loss += (
                float(loss.item())
                *
                bs
            )

            total_samples += bs

    if total_samples == 0:
        raise RuntimeError(
            "Validation loader produced "
            "zero samples."
        )

    return (
        total_loss
        /
        total_samples
    )


# ============================================================
# Deterministic probe / epoch-0 audit
# ============================================================

def build_probe_batch(
    dataset,
    probe_size=12,
):
    n = len(dataset)

    if n <= 0:
        raise RuntimeError(
            "Cannot probe empty dataset."
        )

    probe_size = min(
        int(probe_size),
        n,
    )

    indices = np.linspace(
        0,
        n - 1,
        num=probe_size,
        dtype=int,
    )

    contexts = []
    y_seqs = []
    params = []

    for idx in indices:

        (
            context_norm,
            y_seq_norm,
            param,
        ) = dataset[int(idx)]

        contexts.append(
            context_norm
        )

        y_seqs.append(
            y_seq_norm
        )

        params.append(
            param
        )

    return (
        torch.stack(
            contexts,
            dim=0,
        ),
        torch.stack(
            y_seqs,
            dim=0,
        ),
        torch.stack(
            params,
            dim=0,
        ),
    )


def rms_per_sample(x):
    return torch.sqrt(
        torch.mean(
            x * x,
            dim=(-2, -1),
        )
        +
        1.0e-12
    )


def summarize_probe(
    model,
    probe_batch,
    device,
    *,
    require_m6_identity: bool,
):
    model.eval()
    model.m6.eval()

    (
        context_norm,
        _,
        param,
    ) = probe_batch

    context_norm = (
        context_norm.to(device)
    )

    param = param.to(device)

    batch_size, context_len, channels, h, w = (
        context_norm.shape
    )

    model_input = context_norm.reshape(
        batch_size,
        context_len * channels,
        h,
        w,
    )

    with torch.no_grad():

        (
            output,
            comp,
        ) = model(
            model_input,
            params=param,
            return_components=True,
        )

        m6_output = model.m6(
            model_input
        )

    output_minus_m6_max_abs = float(
        (
            output
            -
            m6_output
        )
        .abs()
        .max()
        .cpu()
    )

    # --------------------------------------------------------
    # Epoch-0 identity is a one-time initialization contract:
    #
    #     R3-1a(0) = R3-1b(0) = M6
    #
    # After optimization begins, non-zero physics gates are
    # expected and the R3 output is allowed to differ from M6.
    # --------------------------------------------------------

    if (
        require_m6_identity
        and
        output_minus_m6_max_abs
        > 1.0e-7
    ):
        raise RuntimeError(
            "R3 epoch-0 output must equal M6. "
            "max_abs_diff="
            f"{output_minus_m6_max_abs:.12e}"
        )

    summary = {
        "representation_mode":
            model.representation_mode,

        "m6_identity_required":
            bool(
                require_m6_identity
            ),

        "output_minus_m6_max_abs":
            output_minus_m6_max_abs,

        "trainable_parameter_count":
            model.trainable_parameter_count(),
    }

    for term_name in (
        model.ACTIVE_TERMS
    ):

        alpha = (
            comp["gate_values"][
                term_name
            ]
        )

        signal = (
            comp["compiled_terms"][
                term_name
            ]
        )

        signal_rms = (
            rms_per_sample(signal)
        )

        summary[
            f"{term_name}_alpha"
        ] = float(
            alpha.detach().cpu()
        )

        summary[
            f"{term_name}_signal_rms_mean"
        ] = float(
            signal_rms.mean().cpu()
        )

        summary[
            f"{term_name}_signal_rms_max"
        ] = float(
            signal_rms.max().cpu()
        )

    return summary


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch,
    val_loss,
    best_val_loss,
    best_epoch,
    args,
    contract_sha,
    split_sha,
    stats_sha,
    m6_sha,
    trainable_names,
    trainable_numel,
    probe_summary,
):
    payload = {
        "experiment":
            "R3-1-Matched-PDE-Coupling-H4",

        "stage":
            (
                "R3-1a"
                if args.representation_mode
                == "naive"
                else
                "R3-1b"
            ),

        "representation_mode":
            args.representation_mode,

        "role":
            (
                "matched_control"
                if args.representation_mode
                == "naive"
                else
                "dimension_valid_candidate"
            ),

        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            (
                None
                if optimizer is None
                else
                optimizer.state_dict()
            ),

        "val_loss":
            float(val_loss),

        "best_val_loss":
            float(best_val_loss),

        "best_epoch":
            int(best_epoch),

        "architecture_contract":
            R3_CONTRACT_PATH,

        "architecture_contract_sha256":
            contract_sha,

        "backbone":
            "audited_frozen_M6_FieldWiseEncoder_H4",

        "backbone_frozen":
            True,

        "active_terms":
            list(
                model.ACTIVE_TERMS
            ),

        "correction": {
            "phi":
                "identity",

            "gate":
                "alpha_max_by_term[j]*tanh(raw_alpha_j)",

            "alpha_max_by_term": {
                term_name:
                    float(
                        model.alpha_max_for_term(
                            term_name
                        )
                    )
                for term_name
                in model.ACTIVE_TERMS
            },

            "capacity_policy":
                "fixed_nonlearnable_per_pde_term_"
                "shared_between_naive_and_canonical",

            "zero_initialized":
                True,

            "rmscap":
                False,

            "realized_dose_matching":
                False,

            "stateparam":
                False,

            "utility_gate":
                False,

            "pde_loss":
                False,
        },

        "prediction_type":
            "normalized_finite_step_delta",

        "training_type":
            "H4_free_autoregressive",

        "rollout_steps":
            int(
                args.rollout_steps
            ),

        "rollout_weights":
            [
                float(x)
                for x
                in make_rollout_weights(
                    args.rollout_steps
                ).tolist()
            ],

        "epochs":
            int(args.epochs),

        "batch_size":
            int(args.batch_size),

        "learning_rate":
            float(args.lr),

        "optimizer":
            "Adam",

        "weight_decay":
            0.0,

        "seed":
            int(args.seed),

        "split_label":
            args.split_label,

        "split":
            args.split,

        "split_sha256":
            split_sha,

        "stats":
            args.stats,

        "stats_sha256":
            stats_sha,

        "m6_checkpoint":
            args.m6_checkpoint,

        "m6_checkpoint_sha256":
            m6_sha,

        "trainable_parameter_names":
            trainable_names,

        "trainable_parameter_count":
            int(
                trainable_numel
            ),

        "frozen_m6_parameter_tensors":
            EXPECTED_FROZEN_M6_TENSORS,

        "probe_summary":
            probe_summary,

        "test_accessed":
            False,
    }

    torch.save(
        payload,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.rollout_steps != 4:
        raise ValueError(
            "R3-1 is locked to "
            "rollout_steps=4."
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if args.lr <= 0.0:
        raise ValueError(
            "lr must be positive."
        )

    alpha_max_by_term = {
        "buoyancy_advection":
            float(
                args.alpha_max_buoyancy_advection
            ),

        "buoyancy_forcing":
            float(
                args.alpha_max_buoyancy_forcing
            ),
    }

    for term_name, value in (
        alpha_max_by_term.items()
    ):
        if value <= 0.0:
            raise ValueError(
                "Every alpha_max_by_term value "
                "must be positive. "
                f"term={term_name}, value={value}"
            )

    if (
        set(alpha_max_by_term)
        !=
        set(EXPECTED_ACTIVE_TERMS)
    ):
        raise RuntimeError(
            "alpha_max_by_term keys do not match "
            "the frozen R3 active routing."
        )

    for path in (
        args.split,
        args.stats,
        args.m6_checkpoint,
        R3_CONTRACT_PATH,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                path
            )

    contract = verify_contract()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "R3-1 MATCHED PDE COUPLING H4"
    )

    print(
        "=" * 100
    )

    print(
        "Stage:",
        (
            "R3-1a CONTROL"
            if args.representation_mode
            == "naive"
            else
            "R3-1b FORMAL CANDIDATE"
        ),
    )

    print(
        "Representation:",
        args.representation_mode,
    )

    print(
        "Active terms:",
        EXPECTED_ACTIVE_TERMS,
    )

    print(
        "Frozen M6: True"
    )

    print(
        "StateParam: OFF"
    )

    print(
        "Utility Gate: OFF"
    )

    print(
        "RMSCap: OFF"
    )

    print(
        "Dose matching: OFF"
    )

    print(
        "Learnable Phi: OFF"
    )

    print(
        "PDE loss: OFF"
    )

    print(
        "TEST access: FORBIDDEN"
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Alpha max by term:",
        alpha_max_by_term,
    )

    print(
        "Matched capacity policy: "
        "same per-term bounds required for "
        "R3-1a and R3-1b"
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    contract_sha = sha256_file(
        R3_CONTRACT_PATH
    )

    split_sha = sha256_file(
        args.split
    )

    stats_sha = sha256_file(
        args.stats
    )

    m6_sha = sha256_file(
        args.m6_checkpoint
    )

    print()
    print(
        "========== PROVENANCE =========="
    )

    print(
        "CONTRACT_SHA256:",
        contract_sha,
    )

    print(
        "SPLIT_SHA256:",
        split_sha,
    )

    print(
        "STATS_SHA256:",
        stats_sha,
    )

    print(
        "M6_CHECKPOINT_SHA256:",
        m6_sha,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    with open(
        args.split,
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_base = RBCDataset(
        split_config=split["train"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    val_base = RBCDataset(
        split_config=split["val"],
        normalize=True,
        stats_path=args.stats,
        return_sequence=True,
        target_steps=args.rollout_steps,
        return_params=True,
    )

    train_dataset = (
        R3MultiStepParamDataset(
            train_base,
            context_length=4,
        )
    )

    val_dataset = (
        R3MultiStepParamDataset(
            val_base,
            context_length=4,
        )
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print()
    print(
        "========== DATA =========="
    )

    print(
        "TRAIN_SAMPLES:",
        len(train_dataset),
    )

    print(
        "VAL_SAMPLES:",
        len(val_dataset),
    )

    (
        sample_context,
        sample_y_seq,
        sample_param,
    ) = train_dataset[0]

    print(
        "CONTEXT_SHAPE:",
        tuple(
            sample_context.shape
        ),
    )

    print(
        "Y_SEQ_SHAPE:",
        tuple(
            sample_y_seq.shape
        ),
    )

    print(
        "PARAM_SHAPE:",
        tuple(
            sample_param.shape
        ),
    )

    probe_batch = build_probe_batch(
        val_dataset,
        probe_size=12,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    metadata = (
        build_rbc_canonical_metadata(
            args.stats
        )
    )

    set_seed(
        args.seed
    )

    model = R3PDECouplingFNO2d(
        canonical_metadata=metadata,
        representation_mode=(
            args.representation_mode
        ),
        alpha_max_by_term=(
            alpha_max_by_term
        ),
        freeze_m6=True,
    ).to(device)

    m6_checkpoint = torch.load(
        args.m6_checkpoint,
        map_location=device,
    )

    model.load_m6_state_dict(
        extract_m6_state(
            m6_checkpoint
        )
    )

    # --------------------------------------------------------
    # Parameter audit
    # --------------------------------------------------------

    trainable_names = (
        model.trainable_parameter_names()
    )

    trainable_numel = (
        model.trainable_parameter_count()
    )

    if (
        trainable_numel
        !=
        EXPECTED_TRAINABLE_NUMEL
    ):
        raise RuntimeError(
            "Unexpected R3 trainable numel.\n"
            f"Expected={EXPECTED_TRAINABLE_NUMEL}\n"
            f"Actual={trainable_numel}"
        )

    expected_names = {
        "raw_alpha.buoyancy_advection",
        "raw_alpha.buoyancy_forcing",
    }

    if set(
        trainable_names
    ) != expected_names:
        raise RuntimeError(
            "Unexpected trainable names: "
            f"{trainable_names}"
        )

    frozen_m6_tensors = (
        model
        .frozen_m6_parameter_tensor_count()
    )

    if (
        frozen_m6_tensors
        !=
        EXPECTED_FROZEN_M6_TENSORS
    ):
        raise RuntimeError(
            "Unexpected frozen M6 tensor count: "
            f"{frozen_m6_tensors}"
        )

    if any(
        p.requires_grad
        for p in model.m6.parameters()
    ):
        raise RuntimeError(
            "M6 backbone is not fully frozen."
        )

    print()
    print(
        "========== PARAMETER AUDIT =========="
    )

    print(
        "TRAINABLE_NAMES:",
        trainable_names,
    )

    print(
        "TRAINABLE_NUMEL:",
        trainable_numel,
    )

    print(
        "FROZEN_M6_PARAMETER_TENSORS:",
        frozen_m6_tensors,
    )

    # --------------------------------------------------------
    # Epoch-0 interface audit
    # --------------------------------------------------------

    probe_summary = summarize_probe(
        model,
        probe_batch,
        device,
        require_m6_identity=True,
    )

    print()
    print(
        "========== EPOCH-0 / INTERFACE AUDIT =========="
    )

    print(
        json.dumps(
            probe_summary,
            indent=2,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------
    # Dry run stops here.
    # --------------------------------------------------------

    if args.dry_run:
        print()
        print(
            "✅ R3-1 DRY-RUN PASS"
        )
        print(
            "No optimization performed."
        )
        print(
            "No checkpoint written."
        )
        return

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    criterion = (
        FieldWiseRelativeL2Loss()
    )

    rollout_weights = (
        make_rollout_weights(
            args.rollout_steps
        )
        .to(device)
    )

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    best_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_best.pth",
    )

    last_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_last.pth",
    )

    initial_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        rollout_weights=rollout_weights,
        device=device,
        max_batches=args.max_val_batches,
    )

    best_val_loss = initial_val
    best_epoch = 0

    save_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        val_loss=initial_val,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        args=args,
        contract_sha=contract_sha,
        split_sha=split_sha,
        stats_sha=stats_sha,
        m6_sha=m6_sha,
        trainable_names=trainable_names,
        trainable_numel=trainable_numel,
        probe_summary=probe_summary,
    )

    print()
    print(
        "========== EPOCH 0 / PURE-M6 START =========="
    )

    print(
        f"VAL_LOSS={initial_val:.12f}"
    )

    print(
        "BEST_EPOCH=0"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()
        model.m6.eval()

        train_loss_sum = 0.0
        train_samples = 0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            if (
                args.max_train_batches
                is not None
                and
                batch_idx
                >=
                args.max_train_batches
            ):
                break

            (
                context_norm,
                y_seq_norm,
                param,
            ) = batch

            context_norm = (
                context_norm.to(device)
            )

            y_seq_norm = (
                y_seq_norm.to(device)
            )

            param = param.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                loss,
                _,
            ) = (
                autoregressive_multistep_loss_r3(
                    model=model,
                    context_norm=context_norm,
                    y_seq_norm=y_seq_norm,
                    param=param,
                    criterion=criterion,
                    rollout_weights=rollout_weights,
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite loss at "
                    f"epoch={epoch}, "
                    f"batch={batch_idx}: "
                    f"{loss}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            bs = context_norm.shape[0]

            train_loss_sum += (
                float(
                    loss.item()
                )
                *
                bs
            )

            train_samples += bs

        if train_samples == 0:
            raise RuntimeError(
                "Training loader produced "
                "zero samples."
            )

        train_loss = (
            train_loss_sum
            /
            train_samples
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            rollout_weights=rollout_weights,
            device=device,
            max_batches=args.max_val_batches,
        )

        probe_summary = summarize_probe(
            model,
            probe_batch,
            device,
            require_m6_identity=False,
        )

        improved = (
            val_loss
            <
            best_val_loss
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch

            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                args=args,
                contract_sha=contract_sha,
                split_sha=split_sha,
                stats_sha=stats_sha,
                m6_sha=m6_sha,
                trainable_names=trainable_names,
                trainable_numel=trainable_numel,
                probe_summary=probe_summary,
            )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            args=args,
            contract_sha=contract_sha,
            split_sha=split_sha,
            stats_sha=stats_sha,
            m6_sha=m6_sha,
            trainable_names=trainable_names,
            trainable_numel=trainable_numel,
            probe_summary=probe_summary,
        )

        alpha_adv = (
            probe_summary[
                "buoyancy_advection_alpha"
            ]
        )

        alpha_buoy = (
            probe_summary[
                "buoyancy_forcing_alpha"
            ]
        )

        print(
            f"[Epoch {epoch:02d}/"
            f"{args.epochs:02d}] "
            f"train={train_loss:.12f} "
            f"val={val_loss:.12f} "
            f"alpha_adv="
            f"{alpha_adv:+.6e} "
            f"alpha_buoy="
            f"{alpha_buoy:+.6e} "
            f"best_epoch="
            f"{best_epoch}"
            +
            (
                "  ✅ BEST"
                if improved
                else ""
            )
        )

    print()
    print(
        "========== R3-1 COMPLETE =========="
    )

    print(
        "REPRESENTATION_MODE:",
        args.representation_mode,
    )

    print(
        "INITIAL_VAL:",
        f"{initial_val:.12f}",
    )

    print(
        "BEST_VAL:",
        f"{best_val_loss:.12f}",
    )

    print(
        "BEST_EPOCH:",
        best_epoch,
    )

    print(
        "BEST_CHECKPOINT:",
        best_path,
    )

    print(
        "LAST_CHECKPOINT:",
        last_path,
    )


if __name__ == "__main__":
    main()
