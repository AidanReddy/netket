"""Custom Haldane-model helpers used by notebooks."""

from .bandstructure import (
    berry_curvature_and_metric_trace_at_k,
    bloch_hamiltonian_at_k,
    dvector_and_derivatives_at_k,
    eigenvalues_at_k,
    qgt_from_bloch_hamiltonian,
)
from .geometry import (
    ANNN_VECTORS,
    DELTA_VECTORS,
    PRIMITIVE_A1,
    PRIMITIVE_A2,
    SUBLATTICE_A_OFFSET,
    SUBLATTICE_B_OFFSET,
    first_bz_hexagon_vertices,
    fold_to_shortest_k,
    high_symmetry_points,
    k_path_gamma_k_m_kp_gamma,
    reciprocal_from_basis,
    reciprocal_vectors,
    sample_shortest_representative_bz,
)
from .haldane_model import build_haldane_hamiltonian
from .models import (
    ComplexLogCoshOutputHead,
    LogSlaterDeterminant,
    LogSlaterJastrow,
    LogSlaterSpatialViT,
    LogSpatialViT,
    SiteOccupancyEmbedding,
    SpatialEncoder,
    SpatialEncoderBlock,
    SpatialFactoredMultiHeadAttention,
    make_translation_equivariant_pair_data,
    make_translation_equivariant_pair_data_from_graph,
)
from .observables import (
    map_radial_g_to_minimum_image_2d,
    pair_correlation_cartesian,
    radial_average_structure_factor,
    static_structure_factor,
)
from .optimization import (
    exact_manybody_ground_state_energy,
    exact_noninteracting_ground_state_energy_bloch,
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
    tree_l2_norm,
)
