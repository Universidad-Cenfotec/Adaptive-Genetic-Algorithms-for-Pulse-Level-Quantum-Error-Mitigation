"""
Utility script to generate pulse CSVs/plots for selected algorithms
without running any GA optimization or noise.

Run with:
    python pcs_experiment.py

Set PCS_MAX_WORKERS=1 to disable parallel execution or override the default
worker count (cpu_count - 1).
"""

from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from math import exp, log
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
try:
    from matplotlib import checkdep_usetex as _checkdep_usetex
except ImportError:  # fallback if API unavailable
    _checkdep_usetex = None

from circuits.bell_circuit import BellCircuit
from circuits.bernstein_vaizirani_circuit import BernsteinVaziraniCircuit
from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.ghz_circuit import GHZCircuit
from circuits.grover_circuit import GroverCircuit
from circuits.inverse_quantum_fourier_transformation import (
    InverseQuantumFourierCircuit,
)
from circuits.layered_entangling_circuit import LayeredEntanglingCircuit
from circuits.quantum_fourier_transformation import QuantumFourierCircuit
from circuits.random_universal_circuit import RandomUniversalCircuit
from circuits.single_qubit_pi_circuit import SingleQubitPiCircuit
from circuits.teleportation_pre_measurement_circuit import (
    TeleportationPreMeasurementCircuit,
)
from main import run_algorithm_without_optimization
from src.csv_logger import CSVLogger
from src.noise_model import NoiseModel

@dataclass(frozen=True)
class ScenarioSpec:
    display: str
    slug: str
    num_qubits: int
    circuit_cls: type
    builder_kwargs: Dict[str, object] = field(default_factory=dict)
    use_fixed: bool = False

    def build_circuit(self):
        if self.builder_kwargs:
            return self.circuit_cls(self.num_qubits, **self.builder_kwargs)
        return self.circuit_cls(self.num_qubits)


def _scenario(display, slug, num_qubits, circuit_cls, *, builder_kwargs=None, use_fixed_settings=False):
    return ScenarioSpec(
        display=display,
        slug=slug,
        num_qubits=num_qubits,
        circuit_cls=circuit_cls,
        builder_kwargs=dict(builder_kwargs or {}),
        use_fixed=use_fixed_settings,
    )


# (display_name, slug, num_qubits, builder)
BASE_SCENARIOS = [
    _scenario("Bell (2q)", "Bell_2Q", 2, BellCircuit),
    _scenario(
        "Teleportation pre-meas (3q)",
        "TeleportationPreMeas_3Q",
        3,
        TeleportationPreMeasurementCircuit,
    ),
    _scenario("GHZ (3q)", "GHZ_3Q", 3, GHZCircuit),
    _scenario("Single-qubit pi (1q)", "SingleQubitPi_1Q", 1, SingleQubitPiCircuit),
    # _scenario("Deutsch-Jozsa (4q)", "DeutschJozsa_4Q", 4, DeutschJozsaCircuit),
    # _scenario("Grover (4q)", "Grover_4Q", 4, GroverCircuit),
    # _scenario(
    #     "Bernstein-Vazirani (4q)",
    #     "BernsteinVazirani_4Q",
    #     4,
    #     BernsteinVaziraniCircuit,
    # ),
    # _scenario("QFT (4q)", "QFT_4Q", 4, QuantumFourierCircuit),
    # _scenario("IQFT (4q)", "IQFT_4Q", 4, InverseQuantumFourierCircuit),
    # _scenario(
    #     "Layered Entangling (4q)",
    #     "LayeredEntangling_4Q",
    #     4,
    #     LayeredEntanglingCircuit,
    #     builder_kwargs={"num_layers": 5},
    # ),
    # _scenario("Random Universal (4q)", "RandomUniversal_4Q", 4, RandomUniversalCircuit),
]

# PCS hyperparameters (matching the paper)
ALPHA = 0.5
BETA = 0.5
GAMMA1 = 1.0
GAMMA2 = 1.0
TIME_UNIT_SECONDS = 0.1e-9  # match Δt = 0.1 ns from the paper
OMEGA_C = 2.0 * np.pi * 250e6  # rad/s cutoff
EPS = 1e-12
USE_FIXED_GATE_SETTINGS = False
ADD_FIXED_SETTING_VARIANTS = False
REPETITIONS = 500
DPI_LATEX = 600

def _build_latex_plot_style():
    style = {
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": DPI_LATEX,
        "figure.dpi": DPI_LATEX,
        "mathtext.fontset": "cm",
    }
    use_tex = False
    if _checkdep_usetex is not None:
        try:
            _checkdep_usetex(True)
            use_tex = True
        except Exception:
            use_tex = False
    if use_tex:
        style["text.usetex"] = True
    else:
        style["text.usetex"] = False
        style["font.serif"] = ["DejaVu Serif", "Computer Modern"]
    return style


_LATEX_PLOT_STYLE = _build_latex_plot_style()

FIXED_SETTING_ARGS = {
    "SNOT": {"num_tslots": 4, "evo_time": 1.2},
    "X": {"num_tslots": 4, "evo_time": 1.0},
    "CNOT": {"num_tslots": 5, "evo_time": 1.6},
    "SWAP": {"num_tslots": 6, "evo_time": 2.0},
    "CPHASE": {"num_tslots": 5, "evo_time": 1.4},
    "RZ": {"num_tslots": 3, "evo_time": 0.8},
    "GLOBALPHASE": {"num_tslots": 2, "evo_time": 0.5},
}

SCENARIOS = list(BASE_SCENARIOS)

if ADD_FIXED_SETTING_VARIANTS:
    for base in BASE_SCENARIOS:
        SCENARIOS.append(
            _scenario(
                f"{base.display} (fixed settings)",
                f"{base.slug}_Fixed",
                base.num_qubits,
                base.circuit_cls,
                builder_kwargs=base.builder_kwargs,
                use_fixed_settings=True,
            )
        )

_RANDOM_CONFIG = {
    "SNOT": (1, 10, 0.1, 3.0),
    "X": (1, 10, 0.1, 3.0),
    "CNOT": (1, 10, 0.1, 3.0),
    "SWAP": (1, 10, 0.1, 3.0),
    "CPHASE": (1, 10, 0.1, 3.0),
    "RZ": (1, 10, 0.1, 3.0),
    "GLOBALPHASE": (1, 10, 0.1, 3.0),
}


def _generate_random_setting_args(seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    args = {}
    for gate, (slot_low, slot_high, time_low, time_high) in _RANDOM_CONFIG.items():
        args[gate] = {
            "num_tslots": int(rng.integers(slot_low, slot_high)),
            "evo_time": float(rng.uniform(time_low, time_high)),
        }
    return args


def _load_pulse_csv(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, 0]
    channels = header[1:]
    signals = data[:, 1:]
    return time, channels, signals


def _compute_tdc(signal: np.ndarray, dt_samples: float) -> float:
    first = np.gradient(signal, dt_samples)
    second = np.gradient(first, dt_samples)
    denom = np.std(first)
    if denom <= EPS:
        return 0.0
    return np.std(second) / (denom + EPS)


def _compute_fdc(signal: np.ndarray, dt_seconds: float) -> float:
    freq_spectrum = np.fft.fft(signal)
    freqs = 2.0 * np.pi * np.fft.fftfreq(signal.size, d=dt_seconds)
    spec_energy = np.abs(freq_spectrum) ** 2
    total_energy = spec_energy.sum()
    if total_energy <= EPS:
        return 0.0

    b_rms = np.sqrt(np.sum((freqs ** 2) * spec_energy) / total_energy)
    oob_mask = np.abs(freqs) > OMEGA_C
    e_oob = spec_energy[oob_mask].sum() / total_energy

    return GAMMA1 * (b_rms / OMEGA_C) + GAMMA2 * e_oob


def _compute_channel_pcs(time: np.ndarray, signal: np.ndarray):
    if time.size < 2:
        return 0.0, 0.0, 0.0
    dt_raw = float(np.mean(np.diff(time)))
    if dt_raw <= EPS:
        return 0.0, 0.0, 0.0
    tdc = _compute_tdc(signal, dt_raw)
    dt_seconds = dt_raw * TIME_UNIT_SECONDS
    fdc = _compute_fdc(signal, dt_seconds)
    pcs = ALPHA * tdc + BETA * fdc
    return tdc, fdc, pcs


def _geometric_mean(values):
    filtered = [max(v, EPS) for v in values]
    log_avg = sum(log(v) for v in filtered) / len(filtered)
    return exp(log_avg)


def _write_pcs_summary(scenario_dir: Path, circuit_name: str, rows: list):
    summary_path = scenario_dir / f"{circuit_name}_pcs_summary.csv"
    header = ["scenario", "channel", "tdc", "fdc", "pcs"]
    table = [[row[h] for h in header] for row in rows]
    np.savetxt(
        summary_path,
        table,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%s",
    )


def _write_aggregate_summary(
    scenario_dir: Path,
    display_name: str,
    circuit_name: str,
    channel_stats,
    pcs_values: list,
    fidelity_values: list,
):
    summary_path = scenario_dir / f"{circuit_name}_pcs_summary_aggregate.csv"
    header = ["scope", "name", "metric", "mean", "std", "min", "max"]
    rows = []

    if pcs_values:
        g_arr = np.array(pcs_values, dtype=float)
        rows.append([
            "global",
            display_name,
            "PCS",
            f"{np.mean(g_arr):.6f}",
            f"{np.std(g_arr, ddof=0):.6f}",
            f"{np.min(g_arr):.6f}",
            f"{np.max(g_arr):.6f}",
        ])
    if fidelity_values:
        f_arr = np.array(fidelity_values, dtype=float)
        rows.append([
            "global",
            display_name,
            "Fidelity",
            f"{np.mean(f_arr):.6f}",
            f"{np.std(f_arr, ddof=0):.6f}",
            f"{np.min(f_arr):.6f}",
            f"{np.max(f_arr):.6f}",
        ])

    for channel, metrics in channel_stats.items():
        for metric_name, values in metrics.items():
            if not values:
                continue
            arr = np.array(values, dtype=float)
            rows.append([
                "channel",
                channel,
                metric_name.upper(),
                f"{np.mean(arr):.6f}",
                f"{np.std(arr, ddof=0):.6f}",
                f"{np.min(arr):.6f}",
                f"{np.max(arr):.6f}",
            ])

    if rows:
        np.savetxt(
            summary_path,
            rows,
            delimiter=",",
            header=",".join(header),
            comments="",
            fmt="%s",
        )


def _append_global_results(aggregate_rows: list, family_label: str, pcs_values: list, fidelity_values: list):
    if not pcs_values:
        return
    pcs_arr = np.array(pcs_values, dtype=float)
    fidelity_arr = np.array(fidelity_values, dtype=float) if fidelity_values else None
    pcs_std = float(np.std(pcs_arr, ddof=0))
    sample_count = pcs_arr.size
    pcs_sem = float(pcs_std / np.sqrt(sample_count)) if sample_count > 0 else 0.0
    aggregate_rows.append({
        "family": family_label,
        "pcs_mean": float(np.mean(pcs_arr)),
        "pcs_std": pcs_std,
        "pcs_sem": pcs_sem,
        "pcs_count": int(sample_count),
        "pcs_min": float(np.min(pcs_arr)),
        "pcs_max": float(np.max(pcs_arr)),
        "fidelity_mean": float(np.mean(fidelity_arr)) if fidelity_arr is not None else None,
        "fidelity_std": float(np.std(fidelity_arr, ddof=0)) if fidelity_arr is not None else None,
    })


def _plot_pcs_results(aggregate_rows, output_dir: Path, root_dir_plot: Path):
    if not aggregate_rows:
        return

    parsed = [
        (
            row["family"],
            float(row["pcs_mean"]),
            float(row.get("pcs_sem", row["pcs_std"])),
        )
        for row in aggregate_rows
    ]
    parsed.sort(key=lambda x: x[1])  # ascending for readability
    families = [p[0] for p in parsed]
    pcs_values = [p[1] for p in parsed]
    pcs_sem = [p[2] for p in parsed]
    max_pcs = max(pcs_values) if pcs_values else 1.0

    with plt.rc_context(_LATEX_PLOT_STYLE):
        plt.figure(figsize=(6.2, 4.2))
        bars = plt.barh(
            families,
            pcs_values,
            xerr=pcs_sem,
            color="#1f77b4",
            alpha=0.9,
            error_kw={"capsize": 3, "elinewidth": 0.8},
        )
        plt.xlabel("PCS Global")
        plt.title("Pulse Complexity Score by Algorithm")
        plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

        for bar, value in zip(bars, pcs_values):
            plt.text(
                bar.get_width() + 0.02 * max_pcs,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
            )

        figure_path = output_dir / "pcs_sampling_results.png"
        plt.savefig(figure_path, dpi=DPI_LATEX, bbox_inches="tight")
        plt.close()

    with plt.rc_context(_LATEX_PLOT_STYLE):
        plt.figure(figsize=(6.2, 4.2))
        bars = plt.barh(
            families,
            pcs_values,
            xerr=pcs_sem,
            color="#1f77b4",
            alpha=0.9,
            error_kw={"capsize": 3, "elinewidth": 0.8},
        )
        plt.xlabel("PCS Global")
        plt.title("Pulse Complexity Score by Algorithm")
        plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

        for bar, value in zip(bars, pcs_values):
            plt.text(
                bar.get_width() + 0.02 * max_pcs,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
            )

        plt.savefig(root_dir_plot, dpi=DPI_LATEX, bbox_inches="tight")
        plt.close()


def _resolve_max_workers(override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    env_value = os.environ.get("PCS_MAX_WORKERS")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            print(f"[WARN] Invalid PCS_MAX_WORKERS value '{env_value}', falling back to auto-detected workers.")
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _execute_scenario_run(
    scenario: ScenarioSpec,
    scenario_dir: Path,
    run_idx: int,
    use_fixed_settings: bool,
    fixed_setting_args: Optional[Dict[str, Dict[str, float]]],
    repetitions: int,
    random_seed: Optional[int] = None,
) -> Dict[str, object]:
    display_name = scenario.display
    circuit_name = scenario.slug

    print(f"\n=== Running scenario: {display_name} (run {run_idx}/{repetitions}) ===")
    run_dir = scenario_dir / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    circuit = scenario.build_circuit()
    noise_model = NoiseModel(scenario.num_qubits, enabled=False)
    logger = CSVLogger(circuit_name, output_dir=run_dir)

    if use_fixed_settings and fixed_setting_args is not None:
        run_setting_args = deepcopy(fixed_setting_args)
    else:
        run_setting_args = _generate_random_setting_args(seed=random_seed)

    fidelity_no_opt = run_algorithm_without_optimization(
        circuit,
        scenario.num_qubits,
        f"{circuit_name}_No_Opt",
        noise_model,
        logger,
        output_dir=run_dir,
        setting_args=run_setting_args,
    )
    pulses_csv = run_dir / f"{circuit_name}_pulses_no_optimization_noise_free.csv"
    if not pulses_csv.exists():
        message = f"Pulses CSV not found at {pulses_csv}, skipping PCS for this run."
        print(f"[WARN] {message}")
        return {"run_idx": run_idx, "success": False, "message": message}

    time, channels, signals = _load_pulse_csv(pulses_csv)
    channel_rows = []
    pcs_values = []
    channel_metrics: List[Dict[str, float]] = []

    for idx, channel_name in enumerate(channels):
        signal = signals[:, idx]
        tdc, fdc, pcs = _compute_channel_pcs(time, signal)
        pcs_values.append(pcs)
        channel_metrics.append({
            "channel": channel_name,
            "tdc": tdc,
            "fdc": fdc,
            "pcs": pcs,
        })
        channel_rows.append({
            "scenario": f"{display_name} (run {run_idx})",
            "channel": channel_name,
            "tdc": f"{tdc:.6f}",
            "fdc": f"{fdc:.6f}",
            "pcs": f"{pcs:.6f}",
        })
        print(f"  Channel {channel_name}: TDC={tdc:.4f}, FDC={fdc:.4f}, PCS={pcs:.4f}")

    if not pcs_values:
        message = "No PCS values computed for this run."
        print(f"[WARN] {message}")
        return {"run_idx": run_idx, "success": False, "message": message}

    global_pcs = _geometric_mean(pcs_values)
    print(f"  Global PCS (geometric mean): {global_pcs:.4f}")

    channel_rows.insert(0, {
        "scenario": f"{display_name} (run {run_idx})",
        "channel": "global",
        "tdc": "",
        "fdc": "",
        "pcs": f"{global_pcs:.6f}",
    })
    _write_pcs_summary(run_dir, circuit_name, channel_rows)

    return {
        "run_idx": run_idx,
        "success": True,
        "global_pcs": global_pcs,
        "fidelity": fidelity_no_opt,
        "channel_metrics": channel_metrics,
    }


def run_scenarios(max_workers: Optional[int] = None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output = Path("output_circuits") / f"PCS_All_Scenarios_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    effective_workers = _resolve_max_workers(max_workers)
    print(f"[INFO] Running PCS experiment with up to {effective_workers} worker(s).")

    for scenario in SCENARIOS:
        display_name = scenario.display
        circuit_name = scenario.slug
        scenario_dir = base_output / circuit_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        use_fixed_settings = USE_FIXED_GATE_SETTINGS or scenario.use_fixed
        fixed_setting_args = deepcopy(FIXED_SETTING_ARGS) if use_fixed_settings else None
        if use_fixed_settings:
            print(f"[INFO] Using fixed gate settings for pulse construction in {display_name}.")

        channel_stats = defaultdict(lambda: {"tdc": [], "fdc": [], "pcs": []})
        pcs_runs = []
        fidelity_runs = []
        run_results: List[Dict[str, object]] = []

        if effective_workers == 1:
            for run_idx in range(1, REPETITIONS + 1):
                result = _execute_scenario_run(
                    scenario,
                    scenario_dir,
                    run_idx,
                    use_fixed_settings,
                    fixed_setting_args,
                    REPETITIONS,
                )
                run_results.append(result)
        else:
            future_to_idx = {}
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                for run_idx in range(1, REPETITIONS + 1):
                    future = executor.submit(
                        _execute_scenario_run,
                        scenario,
                        scenario_dir,
                        run_idx,
                        use_fixed_settings,
                        fixed_setting_args,
                        REPETITIONS,
                    )
                    future_to_idx[future] = run_idx
                for future in as_completed(future_to_idx):
                    run_idx = future_to_idx[future]
                    try:
                        run_results.append(future.result())
                    except Exception as exc:
                        print(f"[ERROR] Scenario {display_name} run {run_idx} failed: {exc}")

        for result in run_results:
            if not isinstance(result, dict):
                print(f"[WARN] Unexpected result type {type(result)}; skipping.")
                continue
            if not result.get("success"):
                message = result.get("message", "Unknown issue")
                print(f"[WARN] Scenario {display_name} run {result.get('run_idx')} skipped: {message}")
                continue

            pcs_runs.append(float(result["global_pcs"]))
            fidelity_runs.append(float(result["fidelity"]))
            for metric in result["channel_metrics"]:
                stats_entry = channel_stats[metric["channel"]]
                stats_entry["tdc"].append(float(metric["tdc"]))
                stats_entry["fdc"].append(float(metric["fdc"]))
                stats_entry["pcs"].append(float(metric["pcs"]))

        if pcs_runs:
            pcs_arr = np.array(pcs_runs, dtype=float)
            fid_arr = np.array(fidelity_runs, dtype=float) if fidelity_runs else None
            _append_global_results(aggregate_rows, display_name, pcs_runs, fidelity_runs)
            _write_aggregate_summary(
                scenario_dir,
                display_name,
                circuit_name,
                channel_stats,
                pcs_runs,
                fidelity_runs,
            )
            fidelity_summary = (
                f"  Fidelity mean: {fid_arr.mean():.4f} ± {fid_arr.std(ddof=0):.4f}"
                if fid_arr is not None and fid_arr.size > 0
                else "  Fidelity mean: N/A"
            )
            print(
                f"\nCompleted scenario: {display_name}\n"
                f"  PCS mean: {pcs_arr.mean():.4f} ± {pcs_arr.std(ddof=0):.4f} "
                f"(min={pcs_arr.min():.4f}, max={pcs_arr.max():.4f})\n"
                f"{fidelity_summary}"
            )
            print(f"  Aggregated summary CSV: {scenario_dir / f'{circuit_name}_pcs_summary_aggregate.csv'}")
            print(f"  Run directories: {scenario_dir}")
        else:
            print(f"\n[WARN] No PCS runs completed successfully for {display_name}.")

    if aggregate_rows:
        aggregate_path = base_output / "pcs_sampling_results.csv"
        header = [
            "family",
            "pcs_mean",
            "pcs_std",
            "pcs_min",
            "pcs_max",
            "fidelity_mean",
            "fidelity_std",
        ]
        table = []
        for row in aggregate_rows:
            table.append([
                row["family"],
                f"{row['pcs_mean']:.6f}",
                f"{row['pcs_std']:.6f}",
                f"{row['pcs_min']:.6f}",
                f"{row['pcs_max']:.6f}",
                "" if row["fidelity_mean"] is None else f"{row['fidelity_mean']:.6f}",
                "" if row["fidelity_std"] is None else f"{row['fidelity_std']:.6f}",
            ])
        np.savetxt(
            aggregate_path,
            table,
            delimiter=",",
            header=",".join(header),
            comments="",
            fmt="%s",
        )
        root_table_path = Path("pcs_sampling_results.csv")
        np.savetxt(
            root_table_path,
            table,
            delimiter=",",
            header=",".join(header),
            comments="",
            fmt="%s",
        )
        print(f"\nAggregate PCS table written to: {aggregate_path}")
        print(f"Latest aggregate PCS table available at repository root: {root_table_path}")
        root_plot_path = Path("pcs_sampling_results.png")
        _plot_pcs_results(aggregate_rows, base_output, root_plot_path)
        print(f"PCS visualization saved to: {base_output / 'pcs_sampling_results.png'}")
        print(f"Latest PCS visualization available at repository root: {root_plot_path}")


if __name__ == "__main__":
    run_scenarios()
