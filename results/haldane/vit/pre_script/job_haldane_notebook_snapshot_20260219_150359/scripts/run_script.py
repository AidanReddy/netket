import optax

model_type = "slater_spatial_vit"  # options: "slater", "slater_jastrow", "slater_spatial_vit"


# Optimization controls
n_iter = 1_000
n_samples = 1024 * 4
n_discard_per_chain = 64
sweep_size = 64  # number of MC move proposals between sample measurements
n_chains = 16
sample_type = "FullSum" #"FullSum" "MC"

# Spatial-ViT controls (used only for model_type == "slater_spatial_vit")
vit_num_layers = 2 #Number of stacked spatial transformer encoder blocks.
vit_d_model = 16 #Token feature dimension d throughout the ViT.
vit_n_heads = 2 #Number of attention heads; each head has size d_head = d/n_heads
vit_mlp_hidden_factor = 2 #Expansion factor r in each block's FFN, so hidden is size r * d
vit_output_hidden_dim = 16 #Hidden dimension d_h in the final real/imag output dense layers
vit_xi_epsilon = 1.0e-6 #small positive stabilizer added to the learned \xi in the spatial envelope \xi
slater_init_mode = "noninteracting"  # options: "random", "noninteracting"

from aidan_custom.haldane_model import noninteracting_slater_orbitals_haldane
from aidan_custom.models import (
    LogSlaterDeterminant,
    LogSlaterJastrow,
    LogSlaterSpatialViT,
    make_translation_equivariant_pair_data_from_graph,
)
from aidan_custom.optimization import (
    exact_reference_ground_state_energy,
    log_optimization_diagnostics,
)

sampler = nk.sampler.MetropolisFermionHop(
    hi,
    graph=graph,
    n_chains=n_chains,
    sweep_size=sweep_size,
)

if model_type == "slater":
    model = LogSlaterDeterminant(hi, param_dtype=complex)
elif model_type == "slater_jastrow":
    model = LogSlaterJastrow(hi, param_dtype=complex, jastrow_param_dtype=complex)
elif model_type == "slater_spatial_vit":
    pair_classes, pair_distances, pair_vectors = make_translation_equivariant_pair_data_from_graph(graph)
    # print(
    #     f"Spatial pair classes: {pair_distances.size}; "
    #     f"pair_classes shape={pair_classes.shape}; pair_vectors shape={pair_vectors.shape}"
    # )

    # Flax modules are hashed in NetKet internals, so static attributes must be hashable.
    pair_classes_hashable = tuple(tuple(int(v) for v in row) for row in pair_classes)
    pair_distances_hashable = tuple(float(v) for v in pair_distances)

    slater_initial_m_orbitals = None
    if slater_init_mode == "noninteracting":
        slater_initial_m_orbitals = noninteracting_slater_orbitals_haldane(
            graph=graph,
            Lx=Lx,
            Ly=Ly,
            n_fermions=n_fermions,
            t1=t1,
            t2=t2,
            phi=phi,
            m=m,
        )
        print("Initializing Slater determinant from non-interacting ground-state orbitals.")
    elif slater_init_mode != "random":
        raise ValueError(f"Unknown slater_init_mode={slater_init_mode!r}")

    model = LogSlaterSpatialViT(
        hilbert=hi,
        num_layers=vit_num_layers,
        d_model=vit_d_model,
        n_heads=vit_n_heads,
        pair_classes=pair_classes_hashable,
        pair_distances=pair_distances_hashable,
        # VMC_SR currently requires homogeneous parameter dtypes across the model tree.
        slater_param_dtype=jnp.float64,
        slater_initial_m_orbitals=slater_initial_m_orbitals,
        mlp_hidden_factor=vit_mlp_hidden_factor,
        output_hidden_dim=vit_output_hidden_dim,
        xi_epsilon=vit_xi_epsilon,
    )
else:
    raise ValueError(f"Unknown model_type={model_type!r}")

# if sample_type == "MC":
#     vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
# elif sample_type == "FullSum":
#     if not isinstance(ham, nk.operator.FermionOperator2nd):
#         ham = ham.to_fermionoperator2nd()
#     vstate = nk.vqs.FullSumState(hi, model)


n_wavefunction_params = nk.jax.tree_size(vstate.parameters) #sum(int(p.size) for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f"total # of wavefunction parameters: {n_wavefunction_params}")

optimizer = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.05, 0.05, n_iter))
driver = nk.driver.VMC_SR(
    ham,
    optimizer,
    variational_state=vstate,
    diag_shift=0.01,
    mode="complex",
)

log = nk.logging.RuntimeLog()
driver.run(n_iter=n_iter, out=log, callback=log_optimization_diagnostics)
