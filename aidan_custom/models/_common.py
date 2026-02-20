from __future__ import annotations

from typing import Any

import netket as nk
from jax.nn.initializers import lecun_normal

DType = Any
default_kernel_init = lecun_normal()
log_cosh = nk.nn.activation.log_cosh
