# Reproduction script for the 50+50 GPU run recorded in summary.json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import netket as nk
import optax

from aidan_custom.haldane_model import build_haldane_hamiltonian
from aidan_custom.models import LogSlaterSpatialViT, make_translation_equivariant_pair_data_from_graph
from aidan_custom.optimization import save_optimization_plots_from_jsonlog

Lx, Ly = 3, 3
n_fermions = 3
V1 = 0.0

t1 = 1.0
t2 = -1 / (4 * np.cos(0.65))
phi = 0.65
m = 0.0
seed = 1234

graph, hi, ham = build_haldane_hamiltonian(
    Lx=Lx, Ly=Ly, t1=t1, t2=t2, phi=phi, m=m, V1=V1, n_fermions=n_fermions
)
ham_sr = ham.to_fermionoperator2nd()
job_dir = Path('results/job_vit_noninteracting_lx3ly3_nf3_gpu_20260218')
job_dir.mkdir(parents=True, exist_ok=True)

pair_classes, pair_distances, _ = make_translation_equivariant_pair_data_from_graph(graph)
pair_classes_hashable = tuple(tuple(int(v) for v in row) for row in pair_classes)
pair_distances_hashable = tuple(float(v) for v in pair_distances)

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
optimizer = nk.optimizer.Adam(learning_rate=1.0e-2)
driver = nk.driver.VMC_SR(
    ham_sr, optimizer, variational_state=vstate, diag_shift=0.01, mode='complex'
)

json_log = nk.logging.JsonLog(
    str(job_dir / 'train_repro_50plus50'),
    mode='write',
    save_params_every=10,
    write_every=10,
    save_params=True,
)
log1 = nk.logging.RuntimeLog()
driver.run(n_iter=50, out=[log1, json_log])
log2 = nk.logging.RuntimeLog()
driver.run(n_iter=50, out=[log2, json_log])
plot_files = save_optimization_plots_from_jsonlog(
    job_dir=job_dir,
    train_log_path=job_dir / 'train_repro_50plus50.log',
)

print('E50=', float(np.real(np.asarray(log1.data['Energy']['Mean'])[-1])))
print('E100=', float(np.real(np.asarray(log2.data['Energy']['Mean'])[-1])))
print('energy_plot_file=', plot_files['energy_plot'])
