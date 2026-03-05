# Post-training observables from optimized wavefunction.
# Supports both MC samples and exact FullSum weighted basis-state evaluation.
import numpy as np
import matplotlib.pyplot as plt

from aidan_custom.observables import (
    map_radial_g_to_minimum_image_2d,
    minimum_image_translations,
    pair_correlation_cartesian,
    radial_average_structure_factor,
    static_structure_factor,
)

if "samples_obs" not in globals():
    raise RuntimeError("Run the post-training observables-input cell before this plotting cell.")

samples = np.asarray(samples_obs, dtype=float)
weights = globals().get("sample_weights_obs", None)
if weights is not None:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != samples.shape[0]:
        raise ValueError("sample_weights_obs length does not match samples_obs length.")
    wsum = float(weights.sum())
    if wsum <= 0.0:
        raise ValueError("sample_weights_obs must have positive total weight.")
    weights = weights / wsum

n_samples = int(n_samples_obs)
n_sites = int(n_sites_obs)
positions = positions_obs
basis_vectors = np.asarray(graph.basis_vectors, dtype=float)

print(
    f"Using post-training observables input from run {obs_sampling_run}: "
    f"source={obs_source}, n_configs={n_samples}"
)

if weights is None:
    pc_data = pair_correlation_cartesian(
        samples=samples,
        positions=positions,
        basis_vectors=basis_vectors,
        pbc=np.asarray(graph.pbc, dtype=bool),
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
    )

    charge_density = pc_data["charge_density"]
    corr_matrix = pc_data["corr_matrix"]
    dist_matrix = pc_data["dist_matrix"]
    translations = pc_data["translations"]
    r_values_plot = pc_data["r_values_plot"]
    g_r_plot = pc_data["g_r_plot"]

    q_list, s_q = static_structure_factor(
        samples=samples,
        positions=positions,
        basis_vectors=basis_vectors,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
    )

else:
    charge_density = np.einsum("s,si->i", weights, samples, optimize=True)
    corr_matrix = np.einsum("s,si,sj->ij", weights, samples, samples, optimize=True)

    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    translations = minimum_image_translations(
        basis_vectors,
        Lx=Lx,
        Ly=Ly,
        pbc=np.asarray(graph.pbc, dtype=bool),
    )
    dist_matrix = np.full((n_sites, n_sites), np.inf, dtype=float)
    for t in translations:
        dist_matrix = np.minimum(dist_matrix, np.linalg.norm(disp + t, axis=-1))

    offdiag = ~np.eye(n_sites, dtype=bool)
    dist_shell = np.round(dist_matrix, 12)
    r_values = np.unique(dist_shell[offdiag])

    g_r = np.empty(r_values.size, dtype=float)
    norm_density = (n_fermions / n_sites) ** 2
    for i, r in enumerate(r_values):
        mask = (dist_shell == r) & offdiag
        g_r[i] = corr_matrix[mask].mean().real / norm_density

    if r_values.size == 0 or not np.isclose(r_values[0], 0.0):
        r_values_plot = np.concatenate(([0.0], r_values))
        g_r_plot = np.concatenate(([0.0], g_r))
    else:
        r_values_plot = r_values.copy()
        g_r_plot = g_r.copy()
        g_r_plot[0] = 0.0

    q_list, _ = static_structure_factor(
        samples=samples,
        positions=positions,
        basis_vectors=basis_vectors,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
    )
    phases = np.exp(-1j * (positions @ q_list.T))
    rho_q = samples @ phases
    s_q = np.einsum("s,sq->q", weights, np.abs(rho_q) ** 2, optimize=True) / n_fermions


rel_min, g_site, origin_idx, r_site = map_radial_g_to_minimum_image_2d(
    positions=positions,
    basis_coords=np.asarray(graph.basis_coords),
    translations=translations,
    r_values_plot=r_values_plot,
    g_r_plot=g_r_plot,
)

q_shell_unique, s_q_abs, s_q_abs_err, q_abs, s_q_plot = radial_average_structure_factor(
    q_list,
    s_q,
    set_q0_to_zero=True,
)

fig_charge, ax_charge = plt.subplots(figsize=(5.4, 4.8))
sc0 = ax_charge.scatter(
    positions[:, 0],
    positions[:, 1],
    c=charge_density,
    s=400,
    cmap="viridis",
    marker="o",
    clim=(0, np.amax(charge_density)),
)
ax_charge.set_title(r"Charge density $\langle n_i \rangle$")
ax_charge.set_xlabel("x")
ax_charge.set_ylabel("y")
ax_charge.set_aspect("equal", "box")
fig_charge.colorbar(sc0, ax=ax_charge, label=r"$\langle n_i \rangle$")
plt.tight_layout()
plt.show()

fig_pair, ax_pair = plt.subplots(figsize=(6.2, 4.8))
ax_pair.plot(r_values_plot, g_r_plot, "o-", label=r"$g(r)$", color="blue")
ax_pair.set_title("Pair correlation")
ax_pair.set_xlabel("Cartesian distance r")
ax_pair.set_ylabel("g(r)")
ax_pair.grid(alpha=0.25)
ax_pair.legend(fontsize=9)
plt.tight_layout()
plt.show()

fig_pair2d, ax_pair2d = plt.subplots(figsize=(5.6, 5.0))
sc_pair2d = ax_pair2d.scatter(
    rel_min[:, 0],
    rel_min[:, 1],
    c=g_site,
    s=400,
    cmap="viridis",
    marker="o",
)
ax_pair2d.scatter(
    [0.0],
    [0.0],
    s=450,
    facecolors="none",
    edgecolors="black",
    linewidths=1.2,
)
ax_pair2d.set_title("Pair correlation mapped to 2D")
ax_pair2d.set_xlabel("x (minimum image from A origin)")
ax_pair2d.set_ylabel("y (minimum image from A origin)")
ax_pair2d.set_aspect("equal", "box")
fig_pair2d.colorbar(sc_pair2d, ax=ax_pair2d, label="g(r)")
plt.tight_layout()
plt.show()

fig_abs, ax_abs = plt.subplots(figsize=(6.2, 4.0))
ax_abs.errorbar(
    q_shell_unique,
    s_q_abs,
    yerr=s_q_abs_err,
    fmt="o-",
    ms=4,
    lw=1.2,
    capsize=2,
    color="blue",
)
ax_abs.set_title(r"Static structure factor")
ax_abs.set_xlabel(r"$q$")
ax_abs.set_ylabel(r"$S(q)$")
ax_abs.grid(alpha=0.25)
plt.tight_layout()
plt.show()
