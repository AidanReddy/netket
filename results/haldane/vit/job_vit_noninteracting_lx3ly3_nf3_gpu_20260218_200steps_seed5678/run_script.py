import os
import json
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import jax
import netket as nk
import optax

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import LogSlaterSpatialViT, make_translation_equivariant_pair_data_from_graph
from aidan_custom.optimization import (
    exact_reference_ground_state_energy,
    exact_manybody_ground_state_energy,
    save_optimization_plots_from_jsonlog,
)


def _to_float(x):
    return float(np.real(np.asarray(x)))


# Written with Codex 02-18-26.
def main():
    job_dir = Path('results/job_vit_noninteracting_lx3ly3_nf3_gpu_20260218_200steps_seed5678')
    job_dir.mkdir(parents=True, exist_ok=True)

    print('devices:', jax.devices())
    print('default_backend:', jax.default_backend())

    # Test case configuration
    Lx, Ly = 3, 3
    n_fermions = 3
    V1 = 0.0

    t1 = 1.0
    t2 = -1 / (4 * np.cos(0.65))
    phi = 0.65
    m = 0.0

    seed = 5678
    n_iter = 200

    graph, hi, ham = build_haldane_hamiltonian(
        Lx=Lx,
        Ly=Ly,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
        n_fermions=n_fermions,
    )

    # Hashable operator for this NetKet FullSum+SR path
    ham_sr = ham.to_fermionoperator2nd()

    pair_classes, pair_distances, _ = make_translation_equivariant_pair_data_from_graph(graph)
    pair_classes_hashable = tuple(tuple(int(v) for v in row) for row in pair_classes)
    pair_distances_hashable = tuple(float(v) for v in pair_distances)

    # VMC_SR currently requires homogeneous parameter dtypes in tree.
    model = LogSlaterSpatialViT(
        hilbert=hi,
        num_layers=2,
        d_model=32,
        n_heads=4,
        pair_classes=pair_classes_hashable,
        pair_distances=pair_distances_hashable,
        slater_param_dtype=np.float64,
        mlp_hidden_factor=4,
        output_hidden_dim=32,
        xi_epsilon=1.0e-6,
    )

    vstate = nk.vqs.FullSumState(hi, model, seed=seed)

    optimizer = nk.optimizer.Sgd(
        learning_rate=optax.linear_schedule(
            init_value=0.05,
            end_value=0.005,
            transition_steps=n_iter,
        )
    )
    driver = nk.driver.VMC_SR(
        ham_sr,
        optimizer,
        variational_state=vstate,
        diag_shift=0.01,
        mode='complex',
    )

    json_log = nk.logging.JsonLog(
        str(job_dir / 'train'),
        mode='write',
        save_params_every=10,
        write_every=10,
        save_params=True,
    )
    state_log = nk.logging.StateLog(
        str(job_dir / 'states'),
        mode='write',
        save_every=10,
        tar=False,
    )

    driver.run(
        n_iter=n_iter,
        out=[json_log, state_log],
        show_progress=True,
    )

    energy_history = json_log.data['Energy']
    energy_mean = np.asarray(energy_history['Mean'])
    energy_var = np.asarray(energy_history['Variance'])

    E_init = _to_float(energy_mean[0])
    E_final = _to_float(energy_mean[-1])
    E_std_local_final = _to_float(np.sqrt(np.maximum(np.real(energy_var[-1]), 0.0)))

    E_ref_bloch, ref_plot_label, ref_method = exact_reference_ground_state_energy(
        hamiltonian=ham,
        Lx=Lx,
        Ly=Ly,
        n_fermions=n_fermions,
        t1=t1,
        t2=t2,
        phi=phi,
        m=m,
        V1=V1,
    )
    E_ref_ed = exact_manybody_ground_state_energy(ham)

    err_bloch = E_final - E_ref_bloch
    err_ed = E_final - E_ref_ed
    plot_files = save_optimization_plots_from_jsonlog(
        job_dir=job_dir,
        train_log_path=job_dir / 'train.log',
        reference_energy=float(E_ref_bloch),
        reference_label=ref_plot_label,
    )

    summary = {
        'device': str(jax.devices()),
        'backend': jax.default_backend(),
        'system': {
            'Lx': Lx,
            'Ly': Ly,
            'n_fermions': n_fermions,
            'V1': V1,
            't1': t1,
            't2': float(t2),
            'phi': phi,
            'm': m,
        },
        'optimization': {
            'n_iter': n_iter,
            'optimizer': 'Sgd(lr=linear_schedule(0.05->0.005))',
            'driver': 'VMC_SR',
            'state': 'FullSumState',
            'diag_shift': 0.01,
            'mode': 'complex',
            'seed': seed,
            'json_log_prefix': str(job_dir / 'train'),
            'state_log_dir': str(job_dir / 'states'),
        },
        'results': {
            'energy_init_real': E_init,
            'energy_final_real': E_final,
            'energy_std_local_final': E_std_local_final,
            'ref_method': ref_method,
            'ref_bloch': float(E_ref_bloch),
            'ref_ed': float(E_ref_ed),
            'error_vs_bloch': float(err_bloch),
            'error_vs_ed': float(err_ed),
            'abs_error_vs_bloch': float(abs(err_bloch)),
        },
        'notes': [
            'ham_sr = ham.to_fermionoperator2nd() used for hashability in this NetKet build.',
            'slater_param_dtype set to float64 to satisfy current VMC_SR homogeneous-dtype constraint.',
        ],
        'artifacts': plot_files,
    }

    with open(job_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('RESULTS_START')
    print(f"E_init={E_init:.12f}")
    print(f"E_final={E_final:.12f}")
    print(f"E_ref_bloch={E_ref_bloch:.12f}")
    print(f"E_ref_ed={E_ref_ed:.12f}")
    print(f"err_vs_bloch={err_bloch:.12f}")
    print(f"abs_err_vs_bloch={abs(err_bloch):.12f}")
    print(f"energy_plot_file={plot_files['energy_plot']}")
    print(f"summary_file={job_dir / 'summary.json'}")
    print('RESULTS_END')


if __name__ == '__main__':
    main()
