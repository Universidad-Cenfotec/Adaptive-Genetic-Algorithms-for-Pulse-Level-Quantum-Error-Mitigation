"""
Utility script to generate pulse CSVs/plots for the baby algorithms
without running any GA optimization or noise.

Run with:
    python pcs_experiment.py
"""

from __future__ import annotations

from datetime import datetime
from math import exp, log
from pathlib import Path

import numpy as np

from circuits.bell_circuit import BellCircuit
from circuits.ghz_circuit import GHZCircuit
from circuits.single_qubit_pi_circuit import SingleQubitPiCircuit
from circuits.teleportation_pre_measurement_circuit import (
    TeleportationPreMeasurementCircuit,
)
from main import run_algorithm_without_optimization
from src.csv_logger import CSVLogger
from src.noise_model import NoiseModel

# (display_name, slug, circuit_cls, num_qubits)
SCENARIOS = [
    ("Bell (2q)", "Bell_2Q", BellCircuit, 2),
    ("Teleportation pre-meas (3q)", "TeleportationPreMeas_3Q",
     TeleportationPreMeasurementCircuit, 3),
    ("GHZ (3q)", "GHZ_3Q", GHZCircuit, 3),
    ("Single-qubit pi (1q)", "SingleQubitPi_1Q", SingleQubitPiCircuit, 1),
]

# PCS hyperparameters (matching the paper)
ALPHA = 0.5
BETA = 0.5
GAMMA1 = 1.0
GAMMA2 = 1.0
OMEGA_C = 2.0 * np.pi * 250e6  # rad/s cutoff
EPS = 1e-12


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


def _compute_tdc(signal: np.ndarray, dt: float) -> float:
    first = np.gradient(signal, dt)
    second = np.gradient(first, dt)
    denom = np.std(first)
    if denom <= EPS:
        return 0.0
    return np.std(second) / (denom + EPS)


def _compute_fdc(signal: np.ndarray, dt: float) -> float:
    freq_spectrum = np.fft.fft(signal)
    freqs = 2.0 * np.pi * np.fft.fftfreq(signal.size, d=dt)
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
    dt = float(np.mean(np.diff(time)))
    tdc = _compute_tdc(signal, dt)
    fdc = _compute_fdc(signal, dt)
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


def _append_global_results(aggregate_rows: list, family_label: str, channel_rows: list, global_pcs: float):
    aggregate_rows.append({
        "family": family_label,
        "channel": "global",
        "tdc": "",
        "fdc": "",
        "pcs": f"{global_pcs:.6f}",
    })
    aggregate_rows.extend(channel_rows)


def run_scenarios():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output = Path("output_circuits") / f"PCS_Baby_Algorithms_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []

    for display_name, slug, circuit_cls, num_qubits in SCENARIOS:
        print(f"\n=== Running scenario: {display_name} ===")
        circuit = circuit_cls(num_qubits)
        circuit_name = slug
        scenario_dir = base_output / slug
        scenario_dir.mkdir(parents=True, exist_ok=True)

        noise_model = NoiseModel(num_qubits, enabled=False)
        logger = CSVLogger(circuit_name, output_dir=scenario_dir)

        run_algorithm_without_optimization(
            circuit,
            num_qubits,
            f"{circuit_name}_No_Opt",
            noise_model,
            logger,
            output_dir=scenario_dir,
        )

        pulses_csv = scenario_dir / f"{circuit_name}_No_Opt_pulses_no_optimization_noise_free.csv"
        if not pulses_csv.exists():
            print(f"[WARN] Pulses CSV not found at {pulses_csv}, skipping PCS.")
            continue

        time, channels, signals = _load_pulse_csv(pulses_csv)
        channel_rows = []
        aggregate_channel_rows = []
        pcs_values = []

        for idx, channel_name in enumerate(channels):
            signal = signals[:, idx]
            tdc, fdc, pcs = _compute_channel_pcs(time, signal)
            pcs_values.append(pcs)
            channel_rows.append({
                "scenario": display_name,
                "channel": channel_name,
                "tdc": f"{tdc:.6f}",
                "fdc": f"{fdc:.6f}",
                "pcs": f"{pcs:.6f}",
            })
            aggregate_channel_rows.append({
                "family": display_name,
                "channel": channel_name,
                "tdc": f"{tdc:.6f}",
                "fdc": f"{fdc:.6f}",
                "pcs": f"{pcs:.6f}",
            })
            print(f"  Channel {channel_name}: TDC={tdc:.4f}, FDC={fdc:.4f}, PCS={pcs:.4f}")

        if pcs_values:
            global_pcs = _geometric_mean(pcs_values)
            print(f"  Global PCS (geometric mean): {global_pcs:.4f}")
            channel_rows.insert(0, {
                "scenario": display_name,
                "channel": "global",
                "tdc": "",
                "fdc": "",
                "pcs": f"{global_pcs:.6f}",
            })
            _write_pcs_summary(scenario_dir, circuit_name, channel_rows)
            _append_global_results(aggregate_rows, display_name, aggregate_channel_rows, global_pcs)

        print(
            f"Results stored in: {scenario_dir}\n"
            f"  • Pulses CSV: {circuit_name}_No_Opt_pulses_no_optimization_noise_free.csv\n"
            f"  • Pulses plot: pulseswithoutoptimization_noise_free.png\n"
            f"  • Visualizer plot: {circuit_name}_No_Opt_non_optimized_pulses_noise_free.jpg\n"
            f"  • PCS summary: {circuit_name}_pcs_summary.csv"
        )

    if aggregate_rows:
        aggregate_path = base_output / "pcs_sampling_results.csv"
        header = ["family", "channel", "tdc", "fdc", "pcs"]
        table = [[row[h] for h in header] for row in aggregate_rows]
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


if __name__ == "__main__":
    run_scenarios()
