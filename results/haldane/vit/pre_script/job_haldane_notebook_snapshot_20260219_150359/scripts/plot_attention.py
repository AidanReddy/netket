# Visualize Spatial-ViT attention as functions of relative lattice displacement (scatter by pair class)
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

if "model" not in globals() or "vstate" not in globals():
    raise RuntimeError("Run the model construction/optimization cell before this visualization cell.")

if not isinstance(model, LogSlaterSpatialViT):
    raise RuntimeError("This cell currently expects model = LogSlaterSpatialViT.")


def _spatial_attention_class_maps_from_params(
    attn_params,
    pair_distances,
    class_counts_by_row,
    class_counts_uniform,
    xi_epsilon,
    is_uniform_row_class_count,
    reference_row=0,
):
    # Written with Codex 02-19-26.
    alpha = np.asarray(attn_params["alpha"], dtype=np.float64)
    raw_xi = np.asarray(attn_params["raw_xi"], dtype=np.float64)

    xi = np.asarray(jax.nn.softplus(jnp.asarray(raw_xi)) + xi_epsilon, dtype=np.float64)
    envelope_unnorm = np.exp(-pair_distances[None, :] / xi[:, None])

    if is_uniform_row_class_count:
        norm_denom = np.sum(
            envelope_unnorm * class_counts_uniform[None, :],
            axis=-1,
            keepdims=True,
        )
        envelope_norm = envelope_unnorm / norm_denom
        normalization_note = "uniform normalization (independent of i)"
    else:
        # Exact implementation has row-dependent normalization in this branch.
        # For class-wise scatter plotting we show one representative target row i=reference_row.
        norm_denom_hi = np.einsum(
            "hc,ic->hi",
            envelope_unnorm,
            class_counts_by_row,
            optimize=True,
        )
        envelope_norm = envelope_unnorm / norm_denom_hi[:, [reference_row]]
        normalization_note = f"row-dependent normalization (showing i={reference_row})"

    # A_{a,c} = E_{a,c} * alpha_{a,c}
    attention_amplitude = envelope_norm * alpha
    return xi, alpha, attention_amplitude, normalization_note


params = vstate.parameters
vit_params = params["vit"] if "vit" in params else params
encoder_params = vit_params["encoder"]

pair_classes = np.asarray(model.pair_classes, dtype=np.int32)
pair_distances = np.asarray(model.pair_distances, dtype=np.float64).reshape(-1)

if "pair_vectors" in globals():
    pair_vectors_arr = np.asarray(pair_vectors, dtype=np.float64)
else:
    _, _, pair_vectors_arr = make_translation_equivariant_pair_data_from_graph(graph)
    pair_vectors_arr = np.asarray(pair_vectors_arr, dtype=np.float64)

if pair_vectors_arr.shape[0] != pair_distances.size:
    raise ValueError(
        f"pair_vectors count ({pair_vectors_arr.shape[0]}) does not match n_pair_classes ({pair_distances.size})."
    )

n_pair_classes = pair_distances.size
one_hot_classes = jax.nn.one_hot(
    jnp.asarray(pair_classes),
    n_pair_classes,
    dtype=jnp.float64,
)
class_counts_by_row = np.asarray(jnp.sum(one_hot_classes, axis=1), dtype=np.float64)
class_counts_uniform = class_counts_by_row[0]
is_uniform_row_class_count = bool(
    np.all(class_counts_by_row == class_counts_uniform[None, :])
)

print(
    f"Plotting Spatial-ViT class-wise maps for {model.num_layers} layers x {model.n_heads} heads. "
    f"Uniform row class-counts: {is_uniform_row_class_count}"
)

x = pair_vectors_arr[:, 0]
y = pair_vectors_arr[:, 1]

for layer_idx in range(int(model.num_layers)):
    block_name = f"block_{layer_idx}"
    if block_name not in encoder_params:
        raise KeyError(f"Missing encoder params for {block_name}")

    attn_params = encoder_params[block_name]["attn"]
    xi, alpha_by_class, attention_amplitude, normalization_note = (
        _spatial_attention_class_maps_from_params(
            attn_params=attn_params,
            pair_distances=pair_distances,
            class_counts_by_row=class_counts_by_row,
            class_counts_uniform=class_counts_uniform,
            xi_epsilon=float(model.xi_epsilon),
            is_uniform_row_class_count=is_uniform_row_class_count,
            reference_row=0,
        )
    )

    n_heads = int(model.n_heads)
    fig, axes = plt.subplots(n_heads, 2, figsize=(11, 3.8 * n_heads), squeeze=False)

    for head_idx in range(n_heads):
        ax_alpha = axes[head_idx, 0]
        ax_a = axes[head_idx, 1]

        alpha_vals = np.abs(alpha_by_class[head_idx])
        a_vals = np.abs(attention_amplitude[head_idx])

        s = 160.0

        vmax_alpha = float(np.max(alpha_vals))
        if vmax_alpha == 0.0:
            vmax_alpha = 1.0
        sc_alpha = ax_alpha.scatter(
            x,
            y,
            c=alpha_vals,
            s=s,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_alpha,
            edgecolors="k",
            linewidths=0.4,
        )
        fig.colorbar(sc_alpha, ax=ax_alpha, fraction=0.046, pad=0.04)
        ax_alpha.set_title(
            rf"$|\alpha_{{a,c}}|$  (layer {layer_idx}, head {head_idx})" + "\n"
            + rf"$\xi_a={xi[head_idx]:.4f}$"
        )
        ax_alpha.set_xlabel(r"$\Delta x$")
        ax_alpha.set_ylabel(r"$\Delta y$")
        ax_alpha.set_aspect("equal")
        ax_alpha.set_box_aspect(1)

        vmax_a = float(np.max(a_vals))
        if vmax_a == 0.0:
            vmax_a = 1.0
        sc_a = ax_a.scatter(
            x,
            y,
            c=a_vals,
            s=s,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_a,
            edgecolors="k",
            linewidths=0.4,
        )
        fig.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.04)
        ax_a.set_title(
            rf"$|A_{{a,c}}| = |E_{{a,c}}\alpha_{{a,c}}|$  (layer {layer_idx}, head {head_idx})"
        )
        ax_a.set_xlabel(r"$\Delta x$")
        ax_a.set_ylabel(r"$\Delta y$")
        ax_a.set_aspect("equal")
        ax_a.set_box_aspect(1)

    fig.suptitle(
        f"Spatial factored attention by relative displacement (layer {layer_idx})\n{normalization_note}",
        y=1.01,
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
