"""rsnum.random - Random number generation."""

import rsnum._core as _core

_random = _core.random

seed = _random.seed
rand = _random.rand
randn = _random.randn
randint = _random.randint
uniform = _random.uniform