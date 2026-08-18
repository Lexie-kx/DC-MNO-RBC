"""
Physics-side canonical interface for DC-MNO.
"""

from .canonical_metadata import (
    CanonicalMetadata,
    FieldNormalization,
    GridSpec,
    TimeSpec,
    build_rbc_canonical_metadata,
    load_field_normalization,
    rbc_transport_coefficients,
)

from .canonicalizer import (
    Canonicalizer,
)

from .derivatives import (
    divergence,
    grad_x_periodic,
    grad_y_nonperiodic,
    laplacian,
    second_x_periodic,
    second_y_nonperiodic,
)

from .rbc_terms import (
    PDETermSpec,
    RBC_TERM_REGISTRY,
    TermRole,
    TermSemantic,
    get_rbc_term_spec,
)

from .canonical_compiler import (
    CanonicalPDECompiler,
    CompiledPDETerm,
)

from .rbc_residuals import (
    CanonicalRBCResidual,
    RBCResidualResult,
)

__all__ = [
    "CanonicalMetadata",
    "FieldNormalization",
    "GridSpec",
    "TimeSpec",
    "Canonicalizer",
    "CanonicalPDECompiler",
    "CompiledPDETerm",
    "CanonicalRBCResidual",
    "RBCResidualResult",
    "PDETermSpec",
    "RBC_TERM_REGISTRY",
    "TermRole",
    "TermSemantic",
    "build_rbc_canonical_metadata",
    "load_field_normalization",
    "rbc_transport_coefficients",
    "get_rbc_term_spec",
    "grad_x_periodic",
    "grad_y_nonperiodic",
    "second_x_periodic",
    "second_y_nonperiodic",
    "laplacian",
    "divergence",
]
