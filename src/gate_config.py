import numpy as np

rng = np.random.default_rng()

DEFAULT_SETTING_ARGS = {
    "SNOT": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "X": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "CNOT": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "SWAP": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "CPHASE": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "RZ": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    },
    "GLOBALPHASE": {
        "num_tslots": rng.integers(1, 10),
        "evo_time": rng.uniform(0.1, 3),
    }
}
