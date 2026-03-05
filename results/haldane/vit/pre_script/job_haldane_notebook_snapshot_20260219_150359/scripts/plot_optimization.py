#plot optimization data

import math

energy_history = log.data["Energy"]
iters = np.asarray(energy_history.iters)
energy_mean = np.asarray(energy_history["Mean"])
energy_sigma = np.asarray(energy_history["Sigma"])
energy_variance = np.asarray(energy_history["Variance"])

energy_std_local = np.sqrt(np.maximum(np.real(energy_variance), 0.0))
energy_tau = np.asarray(energy_history["TauCorr"])
energy_rhat = np.asarray(energy_history["R_hat"])

# acceptance_history = log.data.get("acceptance", None)
# acceptance_iters = np.asarray(acceptance_history.iters)
# acceptance_values = np.asarray(acceptance_history)

update_norm_history = log.data.get("UpdateNormL2", None)
update_norm_iters = np.asarray(update_norm_history.iters) if update_norm_history is not None else None
update_norm_values = np.asarray(update_norm_history) if update_norm_history is not None else None

# Exact reference: skip if C(n_sites, n_fermions) exceeds cutoff.
reference_dim_cutoff = 100_000
hilbert_dim_comb = math.comb(int(hi.n_orbitals), int(hi.n_fermions))
compute_reference = hilbert_dim_comb <= reference_dim_cutoff

if compute_reference:
    e_ref, e_ref_plot_label, e_ref_method = exact_reference_ground_state_energy(
        hamiltonian=ham,
        Lx=Lx,
        Ly=Ly,
        n_fermions=int(hi.n_fermions),
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
    )
else:
    e_ref = None
    e_ref_plot_label = None
    e_ref_method = (
        f"skipped (C(n_sites,n_fermions)={hilbert_dim_comb} > {reference_dim_cutoff})"
    )

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, energy_mean.real, lw=1.8, color="tab:blue", label="VMC_SR")
ax.fill_between(
    iters,
    energy_mean.real - energy_sigma,
    energy_mean.real + energy_sigma,
    color="tab:blue",
    alpha=0.2,
    linewidth=0,
)
if e_ref is not None:
    ax.axhline(e_ref, color="black", ls="--", lw=1.5, label=e_ref_plot_label)
    ax.set_title("Haldane model: VMC_SR vs exact ground-state energy")
else:
    ax.set_title("Haldane model: VMC_SR energy")
ax.set_xlabel("Optimization step")
ax.set_ylabel("Energy")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, np.log10(energy_std_local), color="tab:purple", lw=1.5)
ax.set_xlabel("Optimization step")
ax.set_ylabel("log10 Std(E_loc)")
ax.set_title("Local energy standard deviation")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(figsize=(7, 4))
# if acceptance_values is not None:
#     ax.plot(acceptance_iters, acceptance_values, color="tab:green", lw=1.5)
#     ax.set_ylim(0.0, 1.0)
# else:
#     ax.text(0.5, 0.5, "Acceptance not available", ha="center", va="center", transform=ax.transAxes)
# ax.set_xlabel("Optimization step")
# ax.set_ylabel("Acceptance")
# ax.set_title("MC acceptance")
# ax.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
if update_norm_values is not None:
    ax.plot(update_norm_iters, update_norm_values, color="tab:red", lw=1.5)
    ax.set_yscale("log")
else:
    ax.text(0.5, 0.5, "Update norm not available", ha="center", va="center", transform=ax.transAxes)
ax.set_xlabel("Optimization step")
ax.set_ylabel(r"||$\Delta\theta$||$_2$")
ax.set_title("Update norm")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
if energy_tau is not None:
    ax.plot(iters, energy_tau, color="tab:orange", lw=1.5, label="TauCorr")
    if energy_rhat is not None:
        ax.plot(iters, energy_rhat, color="tab:brown", lw=1.5, label="R_hat")
        ax.legend()
    ax.set_title("Sampling diagnostics")
else:
    ax.text(0.5, 0.5, "No extra diagnostics available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Sampling diagnostics")
ax.set_xlabel("Optimization step")
ax.set_ylabel("Value")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

if e_ref is not None:
    print(f"Exact ground-state reference [{e_ref_method}] = {e_ref:.10f}")
    print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
    print(f"Final error (VMC - reference) = {energy_mean.real[-1] - e_ref:.10f}")
else:
    print(f"Exact ground-state reference [{e_ref_method}]")
    print(f"Final VMC_SR energy (real part) = {energy_mean.real[-1]:.10f}")
# if acceptance_values is not None:
#     print(f"Final MC acceptance = {float(acceptance_values[-1]):.6f}")
if update_norm_values is not None:
    print(f"Final update norm ||dtheta||_2 = {float(update_norm_values[-1]):.6e}")
print(f"Final local-energy std dev = {float(energy_std_local[-1]):.10f}")
