"""Array typing aliases with documented shape conventions."""

from __future__ import annotations

import numpy as np
from jax import Array

FloatArray = Array
IntArray = Array
KeyArray = Array

ShapeLike = tuple | list | int
DTypeLike = str | np.dtype | type
