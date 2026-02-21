"""Custom Haldane-model helpers used by notebooks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ANNN_VECTORS",
    "BoseFormerEncoder",
    "BoseFormerEncoderBlock",
    "ComplexLogCoshOutputHead",
    "ComplexParticleOrbitalProductHead",
    "DELTA_VECTORS",
    "LogBoseFormerProduct",
    "LogSlaterBoseFormer",
    "LogSlaterDeterminant",
    "LogSlaterJastrow",
    "LogSlaterSpatialViT",
    "LogSpatialViT",
    "OccupiedPeriodicFeatureEmbedding",
    "ParticleQKVSelfAttention",
    "PRIMITIVE_A1",
    "PRIMITIVE_A2",
    "SiteOccupancyEmbedding",
    "SpatialEncoder",
    "SpatialEncoderBlock",
    "SpatialFactoredMultiHeadAttention",
    "SUBLATTICE_A_OFFSET",
    "SUBLATTICE_B_OFFSET",
    "berry_curvature_and_metric_trace_at_k",
    "bloch_ed",
    "bloch_hamiltonian_at_k",
    "build_haldane_hamiltonian",
    "build_mote2_three_orbital_hamiltonian",
    "dvector_and_derivatives_at_k",
    "eigenvalues_at_k",
    "exact_manybody_ground_state_energy",
    "exact_noninteracting_ground_state_energy_bloch",
    "exact_projected_reference_ground_state_energy",
    "exact_reference_ground_state_energy",
    "first_bz_hexagon_vertices",
    "fold_to_shortest_k",
    "high_symmetry_points",
    "k_path_gamma_k_m_kp_gamma",
    "log_optimization_diagnostics",
    "make_supercell_reciprocal_vectors",
    "make_supercell_reciprocal_vectors_from_graph",
    "make_translation_equivariant_pair_data",
    "make_translation_equivariant_pair_data_from_graph",
    "map_radial_g_to_minimum_image_2d",
    "MOTE2_A1",
    "MOTE2_A2",
    "mote2_three_orbital_berry_and_metric_trace",
    "mote2_three_orbital_bloch_hamiltonian",
    "mote2_three_orbital_bloch_hamiltonian_t123",
    "mote2_three_orbital_chern_numbers",
    "mote2_three_orbital_default_parameters",
    "mote2_three_orbital_eigenvalues",
    "mote2_three_orbital_eigenvalues_t123",
    "mote2_three_orbital_parameters_from_t123",
    "mote2_three_orbital_qgt",
    "mote2_three_orbital_reciprocal_vectors",
    "noninteracting_slater_orbitals_mote2_three_orbital",
    "pair_correlation_cartesian",
    "qgt_from_bloch_hamiltonian",
    "radial_average_structure_factor",
    "reciprocal_from_basis",
    "reciprocal_vectors",
    "sample_shortest_representative_bz",
    "static_structure_factor",
    "tree_l2_norm",
]

_NAME_TO_MODULE_ATTR: dict[str, tuple[str, str]] = {
    "berry_curvature_and_metric_trace_at_k": (
        ".bandstructure",
        "berry_curvature_and_metric_trace_at_k",
    ),
    "bloch_hamiltonian_at_k": (".bandstructure", "bloch_hamiltonian_at_k"),
    "dvector_and_derivatives_at_k": (".bandstructure", "dvector_and_derivatives_at_k"),
    "eigenvalues_at_k": (".bandstructure", "eigenvalues_at_k"),
    "qgt_from_bloch_hamiltonian": (".bandstructure", "qgt_from_bloch_hamiltonian"),
    "ANNN_VECTORS": (".geometry", "ANNN_VECTORS"),
    "DELTA_VECTORS": (".geometry", "DELTA_VECTORS"),
    "PRIMITIVE_A1": (".geometry", "PRIMITIVE_A1"),
    "PRIMITIVE_A2": (".geometry", "PRIMITIVE_A2"),
    "SUBLATTICE_A_OFFSET": (".geometry", "SUBLATTICE_A_OFFSET"),
    "SUBLATTICE_B_OFFSET": (".geometry", "SUBLATTICE_B_OFFSET"),
    "first_bz_hexagon_vertices": (".geometry", "first_bz_hexagon_vertices"),
    "fold_to_shortest_k": (".geometry", "fold_to_shortest_k"),
    "high_symmetry_points": (".geometry", "high_symmetry_points"),
    "k_path_gamma_k_m_kp_gamma": (".geometry", "k_path_gamma_k_m_kp_gamma"),
    "reciprocal_from_basis": (".geometry", "reciprocal_from_basis"),
    "reciprocal_vectors": (".geometry", "reciprocal_vectors"),
    "sample_shortest_representative_bz": (".geometry", "sample_shortest_representative_bz"),
    "build_haldane_hamiltonian": (".haldane_model", "build_haldane_hamiltonian"),
    "build_mote2_three_orbital_hamiltonian": (
        ".mote2_three_orbital_model",
        "build_mote2_three_orbital_hamiltonian",
    ),
    "BoseFormerEncoder": (".models", "BoseFormerEncoder"),
    "BoseFormerEncoderBlock": (".models", "BoseFormerEncoderBlock"),
    "ComplexLogCoshOutputHead": (".models", "ComplexLogCoshOutputHead"),
    "ComplexParticleOrbitalProductHead": (".models", "ComplexParticleOrbitalProductHead"),
    "LogBoseFormerProduct": (".models", "LogBoseFormerProduct"),
    "LogSlaterBoseFormer": (".models", "LogSlaterBoseFormer"),
    "LogSlaterDeterminant": (".models", "LogSlaterDeterminant"),
    "LogSlaterJastrow": (".models", "LogSlaterJastrow"),
    "LogSlaterSpatialViT": (".models", "LogSlaterSpatialViT"),
    "LogSpatialViT": (".models", "LogSpatialViT"),
    "OccupiedPeriodicFeatureEmbedding": (".models", "OccupiedPeriodicFeatureEmbedding"),
    "ParticleQKVSelfAttention": (".models", "ParticleQKVSelfAttention"),
    "SiteOccupancyEmbedding": (".models", "SiteOccupancyEmbedding"),
    "SpatialEncoder": (".models", "SpatialEncoder"),
    "SpatialEncoderBlock": (".models", "SpatialEncoderBlock"),
    "SpatialFactoredMultiHeadAttention": (".models", "SpatialFactoredMultiHeadAttention"),
    "make_supercell_reciprocal_vectors": (
        ".models",
        "make_supercell_reciprocal_vectors",
    ),
    "make_supercell_reciprocal_vectors_from_graph": (
        ".models",
        "make_supercell_reciprocal_vectors_from_graph",
    ),
    "make_translation_equivariant_pair_data": (
        ".models",
        "make_translation_equivariant_pair_data",
    ),
    "make_translation_equivariant_pair_data_from_graph": (
        ".models",
        "make_translation_equivariant_pair_data_from_graph",
    ),
    "MOTE2_A1": (".mote2_three_orbital", "MOTE2_A1"),
    "MOTE2_A2": (".mote2_three_orbital", "MOTE2_A2"),
    "mote2_three_orbital_bloch_hamiltonian": (
        ".mote2_three_orbital",
        "mote2_three_orbital_bloch_hamiltonian",
    ),
    "mote2_three_orbital_bloch_hamiltonian_t123": (
        ".mote2_three_orbital",
        "mote2_three_orbital_bloch_hamiltonian_t123",
    ),
    "mote2_three_orbital_eigenvalues": (
        ".mote2_three_orbital",
        "mote2_three_orbital_eigenvalues",
    ),
    "mote2_three_orbital_eigenvalues_t123": (
        ".mote2_three_orbital",
        "mote2_three_orbital_eigenvalues_t123",
    ),
    "mote2_three_orbital_qgt": (
        ".mote2_three_orbital",
        "mote2_three_orbital_qgt",
    ),
    "mote2_three_orbital_berry_and_metric_trace": (
        ".mote2_three_orbital",
        "mote2_three_orbital_berry_and_metric_trace",
    ),
    "mote2_three_orbital_reciprocal_vectors": (
        ".mote2_three_orbital",
        "mote2_three_orbital_reciprocal_vectors",
    ),
    "mote2_three_orbital_default_parameters": (
        ".mote2_three_orbital",
        "mote2_three_orbital_default_parameters",
    ),
    "mote2_three_orbital_parameters_from_t123": (
        ".mote2_three_orbital",
        "mote2_three_orbital_parameters_from_t123",
    ),
    "mote2_three_orbital_chern_numbers": (
        ".mote2_three_orbital",
        "mote2_three_orbital_chern_numbers",
    ),
    "noninteracting_slater_orbitals_mote2_three_orbital": (
        ".mote2_three_orbital_model",
        "noninteracting_slater_orbitals_mote2_three_orbital",
    ),
    "map_radial_g_to_minimum_image_2d": (".observables", "map_radial_g_to_minimum_image_2d"),
    "pair_correlation_cartesian": (".observables", "pair_correlation_cartesian"),
    "radial_average_structure_factor": (".observables", "radial_average_structure_factor"),
    "static_structure_factor": (".observables", "static_structure_factor"),
    "exact_manybody_ground_state_energy": (".optimization", "exact_manybody_ground_state_energy"),
    "exact_noninteracting_ground_state_energy_bloch": (
        ".optimization",
        "exact_noninteracting_ground_state_energy_bloch",
    ),
    "exact_projected_reference_ground_state_energy": (
        ".optimization",
        "exact_projected_reference_ground_state_energy",
    ),
    "exact_reference_ground_state_energy": (".optimization", "exact_reference_ground_state_energy"),
    "log_optimization_diagnostics": (".optimization", "log_optimization_diagnostics"),
    "tree_l2_norm": (".optimization", "tree_l2_norm"),
}


def __getattr__(name: str) -> Any:
    # Written with Codex 02-19-26.
    if name == "bloch_ed":
        module = import_module(".bloch_ed", __name__)
        globals()[name] = module
        return module

    target = _NAME_TO_MODULE_ATTR.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Written with Codex 02-19-26.
    return sorted(set(globals().keys()) | set(__all__))
