import numpy as np

rng = np.random.default_rng()

DEFAULT_SETTING_ARGS = {
    "SNOT": {"num_tslots": 5, "evo_time": 1.0},
    "X": {"num_tslots": 3, "evo_time": 0.5},
    "CNOT": {"num_tslots": 10, "evo_time": 2.0},
}
