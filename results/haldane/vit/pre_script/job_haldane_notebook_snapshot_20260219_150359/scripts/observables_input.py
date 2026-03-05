# Post-training observables input data from the optimized state.
# Works for both MCState (fresh samples) and FullSumState (exact weighted basis sum).
import netket as nk

if isinstance(vstate, nk.vqs.FullSumState):
    states_obs = np.asarray(hi.all_states(), dtype=float).reshape(-1, hi.size)
    psi_obs = np.asarray(vstate.to_array())
    prob_obs = np.abs(psi_obs) ** 2
    prob_sum = float(prob_obs.sum())
    if prob_sum <= 0.0:
        raise RuntimeError("Wavefunction probability norm is non-positive.")

    sample_weights_obs = (prob_obs / prob_sum).astype(float)
    samples_obs = states_obs
    n_samples_obs = int(samples_obs.shape[0])
    n_sites_obs = int(samples_obs.shape[1])
    positions_obs = np.asarray(graph.positions, dtype=float)
    obs_sampling_run = int(globals().get("obs_sampling_run", 0)) + 1
    obs_source = "fullsum_exact"

    n_eff = 1.0 / float(np.sum(sample_weights_obs**2))
    print(
        f"Post-training observables run {obs_sampling_run}: FullSum exact evaluation over "
        f"{n_samples_obs} basis states (effective sample size={n_eff:.1f})."
    )

else:
    obs_n_samples = max(4096 * 10, getattr(vstate, "n_samples", 4096 * 10))
    obs_n_discard_per_chain = max(64, getattr(vstate, "n_discard_per_chain", 64))

    # vstate.reset()
    samples_obs = np.asarray(
        vstate.sample(
            n_samples=obs_n_samples,
            n_discard_per_chain=obs_n_discard_per_chain,
        )
    ).reshape(-1, hi.size).astype(float)

    sample_weights_obs = None
    n_samples_obs = int(samples_obs.shape[0])
    n_sites_obs = int(samples_obs.shape[1])
    positions_obs = np.asarray(graph.positions, dtype=float)
    obs_sampling_run = int(globals().get("obs_sampling_run", 0)) + 1
    obs_source = "mc_samples"

    print(
        f"Post-training observables run {obs_sampling_run}: {n_samples_obs} MC samples "
        f"(single batch of {obs_n_samples}), discard_per_chain={obs_n_discard_per_chain}."
    )
