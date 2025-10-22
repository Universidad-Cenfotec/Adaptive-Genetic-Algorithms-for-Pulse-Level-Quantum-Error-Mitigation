import argparse
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.bell_circuit import BellCircuit
from circuits.bernstein_vaizirani_circuit import BernsteinVaziraniCircuit
from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.ghz_circuit import GHZCircuit
from circuits.grover_circuit import GroverCircuit
from circuits.layered_entangling_circuit import LayeredEntanglingCircuit
from circuits.quantum_fourier_transformation import QuantumFourierCircuit
from circuits.random_universal_circuit import RandomUniversalCircuit
from circuits.single_qubit_pi_circuit import SingleQubitPiCircuit
from circuits.teleportation_pre_measurement_circuit import (
    TeleportationPreMeasurementCircuit,
)
from src.csv_logger import CSVLogger
from src.evaluator import Evaluator
from src.gate_config import DEFAULT_SETTING_ARGS
from src.genetic_optimizer import GeneticOptimizer
from src.noise_model import NoiseModel
from src.visualizer import Visualizer

warnings.filterwarnings("ignore")

UNSUPPORTED_ALGORITHM_SPECIFIED = "Unsupported algorithm specified."

def run_algorithm_without_optimization(
    quantum_circuit,
    num_qubits,
    circuit_name,
    noise_model,
    logger,
    output_dir,
    setting_args=None,
):
    """
    Runs the circuit under noise WITHOUT GA optimization, returns fidelity.
    """
    # Solver options
    solver_options = Options(nsteps=100000, store_states=True)

    processor_no_opt = OptPulseProcessor(
        num_qubits=num_qubits,
        model=SpinChainModel(num_qubits, setup="linear"),
    )
    processor_no_opt.load_circuit(
        quantum_circuit.circuit,
        setting_args=setting_args or DEFAULT_SETTING_ARGS,
        merge_gates=False
    )

    result_no_opt = processor_no_opt.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops,
    )

    fidelity_no_opt = fidelity(result_no_opt.states[-1], quantum_circuit.target_state)
    print(f"\nFidelity WITHOUT optimization for {circuit_name}: {fidelity_no_opt:.4f}")

    # Write CSV summary
    logger.write_summary_no_optimization(noise_model, fidelity_no_opt)

    noise_free = not getattr(noise_model, "enabled", True)
    pulses_title = f"Pulses without optimization {circuit_name}"
    pulses_path = output_dir / "pulseswithoutoptimization"
    if noise_free:
        pulses_title += " (noise-free)"
        pulses_path = output_dir / "pulseswithoutoptimization_noise_free"
        logger.write_pulses(
            processor_no_opt,
            filename_suffix="_pulses_no_optimization_noise_free.csv"
        )

    processor_no_opt.plot_pulses(title=pulses_title, dpi=600)[0].savefig(pulses_path)
    # Visualization
    vis_filename = output_dir / f"{circuit_name}_non_optimized_pulses.jpg"
    if noise_free:
        vis_filename = output_dir / f"{circuit_name}_non_optimized_pulses_noise_free.jpg"

    vis_title = f"Pulses without optimization {circuit_name}"
    if noise_free:
        vis_title += " (noise-free)"

    Visualizer.plot_pulses(
        processor_no_opt,
        vis_title,
        filename=vis_filename
    )
    return fidelity_no_opt

def run_algorithm(
    quantum_circuit,
    num_qubits,
    circuit_name,
    population_size,
    num_generations,
    noise_model,
    logger,
    output_dir,
):
    """
    Runs the genetic optimization on the circuit under noise, returns fidelity.
    """
    solver_options = Options(nsteps=100000, store_states=True)

    # Example evaluator that must implement "evaluate(individual)"
    evaluator = Evaluator(quantum_circuit, noise_model, solver_options)

    # GA optimizer
    optimizer = GeneticOptimizer(
        evaluator=evaluator,
        population_size=population_size,
        num_generations=num_generations,
        use_default=True,
    )

    # Run GA
    pop, logbook = optimizer.run(csv_logger=logger, csv_filename=output_dir / f"{circuit_name}_log.csv")
    best_individual = optimizer.hall_of_fame[0]
    evaluator.evaluate(best_individual, plot_pulses=True, outputdir=output_dir, circuit_name=circuit_name)
    best_fidelity = max(logbook.select("avg"))
    print(f"\nBest individual for {circuit_name}: {best_individual}")
    print(f"Fidelity (best_individual): {best_fidelity:.4f}")

    # Use best individual to set pulses
    processor_optimized = OptPulseProcessor(
        num_qubits=num_qubits, model=SpinChainModel(num_qubits, setup="linear")
    )
    processor_optimized.load_circuit(
        quantum_circuit.circuit, setting_args=best_individual, merge_gates=False
    )

    # Write summary CSV
    logger.write_summary_optimization(
        noise_model,
        best_individual,
        best_fidelity,
        population_size,
        num_generations
    )
    # Write pulses CSV
    logger.write_pulses(processor_optimized)

    # Visualization
    Visualizer.plot_pulses(
        processor_optimized,
        f"Optimized Pulses for {circuit_name}",
        filename=output_dir / f"{circuit_name}_optimized_pulses.jpg"
    )
    Visualizer.plot_fidelity_evolution(
        logbook,
        filename=output_dir / f"{circuit_name}_fidelity_evolution.jpg"
    )
    Visualizer.plot_histogram_fidelities(
        pop,
        filename=output_dir / f"{circuit_name}_histogram_fidelities.jpg"
    )

    # Parameter plots
    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(
        pop, parameters,
        filename_prefix=str(output_dir / f"{circuit_name}_parameter_evolution")
    )
    Visualizer.plot_correlation(
        pop, parameters,
        output_dir=output_dir,
        filename=output_dir / f"{circuit_name}_correlation_matrix.jpg"
    )
    Visualizer.plot_histogram_parameters(
        pop, parameters,
        filename_prefix=str(output_dir / f"{circuit_name}_histogram_parameters")
    )

    return best_fidelity

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Run quantum algorithms with or without GA optimization under noise."
        )

        parser.add_argument(
            "--algorithm",
            type=str,
            choices=[
                "grover",
                "deutsch-jozsa",
                "bernstein-vazirani",
                "qft",
                "iqft",
                "random-universal",
                "layered-entangling",
                "bell",
                "teleportation-pre-meas",
                "ghz",
                "single-qubit-pi",
            ],
            required=True,
            help="Specify which algorithm to run."
        )
        parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits to use in the circuit.")
        parser.add_argument("--num_generations", type=int, default=100, help="Generations for GA.")
        parser.add_argument("--population_size", type=int, default=50, help="Population size for GA.")
        parser.add_argument("--t1", type=float, default=50.0, help="T1 relaxation time.")
        parser.add_argument("--t2", type=float, default=30.0, help="T2 dephasing time.")
        parser.add_argument("--bit_flip_prob", type=float, default=0.02, help="Bit-flip probability.")
        parser.add_argument("--phase_flip_prob", type=float, default=0.02, help="Phase-flip probability.")
        parser.add_argument(
            "--disable-noise",
            action="store_true",
            help="Disable all noise channels (noise-free simulation)."
        )
        args = parser.parse_args()
        default_num_qubits = parser.get_default("num_qubits")

        # Choose circuit
        if args.algorithm == "grover":
            circuit_name = f"Grover_{args.num_qubits}Q"
            quantum_circuit = GroverCircuit(args.num_qubits)
        elif args.algorithm == "deutsch-jozsa":
            circuit_name = f"DeutschJozsa_{args.num_qubits}Q"
            quantum_circuit = DeutschJozsaCircuit(args.num_qubits)
        elif args.algorithm == "bernstein-vazirani":
            circuit_name = f"BernsteinVazirani_{args.num_qubits}Q"
            quantum_circuit = BernsteinVaziraniCircuit(args.num_qubits, None)
        elif args.algorithm == "qft":
            circuit_name = f"QFT_{args.num_qubits}Q"
            quantum_circuit = QuantumFourierCircuit(args.num_qubits)
        elif args.algorithm == "iqft":
            circuit_name = f"IQFT_{args.num_qubits}Q"
        elif args.algorithm == "bell":
            required_qubits = 2
            if args.num_qubits != required_qubits:
                if args.num_qubits != default_num_qubits:
                    msg = "Bell circuit requires exactly 2 qubits."
                    raise ValueError(msg)
                args.num_qubits = required_qubits
            circuit_name = f"Bell_{args.num_qubits}Q"
            quantum_circuit = BellCircuit(args.num_qubits)
        elif args.algorithm == "teleportation-pre-meas":
            required_qubits = 3
            if args.num_qubits != required_qubits:
                if args.num_qubits != default_num_qubits:
                    msg = "Teleportation pre-measurement circuit requires exactly 3 qubits."
                    raise ValueError(msg)
                args.num_qubits = required_qubits
            circuit_name = f"TeleportationPreMeas_{args.num_qubits}Q"
            quantum_circuit = TeleportationPreMeasurementCircuit(args.num_qubits)
        elif args.algorithm == "ghz":
            required_qubits = 3
            if args.num_qubits != required_qubits:
                if args.num_qubits != default_num_qubits:
                    msg = "GHZ circuit requires exactly 3 qubits."
                    raise ValueError(msg)
                args.num_qubits = required_qubits
            circuit_name = f"GHZ_{args.num_qubits}Q"
            quantum_circuit = GHZCircuit(args.num_qubits)
        elif args.algorithm == "single-qubit-pi":
            required_qubits = 1
            if args.num_qubits != required_qubits:
                if args.num_qubits != default_num_qubits:
                    msg = "Single-qubit π circuit requires exactly 1 qubit."
                    raise ValueError(msg)
                args.num_qubits = required_qubits
            circuit_name = f"SingleQubitPi_{args.num_qubits}Q"
            quantum_circuit = SingleQubitPiCircuit(args.num_qubits)
        elif args.algorithm == "random-universal":
            circuit_name = f"RandomUniversal_{args.num_qubits}Q"
            quantum_circuit = RandomUniversalCircuit(args.num_qubits)
        elif args.algorithm == "layered-entangling":
            circuit_name = f"LayeredEntangling_{args.num_qubits}Q"
            quantum_circuit = LayeredEntanglingCircuit(args.num_qubits, num_layers=5)

        else:
            raise ValueError(UNSUPPORTED_ALGORITHM_SPECIFIED)  # noqa: TRY301

        # Create output directory
        timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("output_circuits") / f"{circuit_name}_{timestamp_folder}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create noise model
        noise_model = NoiseModel(
            args.num_qubits,
            t1=args.t1,
            t2=args.t2,
            bit_flip_prob=args.bit_flip_prob,
            phase_flip_prob=args.phase_flip_prob,
            enabled=not args.disable_noise,
        )

        # Create CSV logger
        logger = CSVLogger(circuit_name, output_dir=output_dir)

        # Track total experiment time
        start_time = time.time()

        # 1) Run WITHOUT optimization
        fidelity_no_opt = run_algorithm_without_optimization(
            quantum_circuit,
            args.num_qubits,
            circuit_name + "_No_Opt",
            noise_model,
            logger,
            output_dir=output_dir
        )

        # 2) Run WITH optimization
        fidelity_opt = run_algorithm(
            quantum_circuit,
            args.num_qubits,
            circuit_name + "_With_Opt",
            population_size=args.population_size,
            num_generations=args.num_generations,
            noise_model=noise_model,
            logger=logger,
            output_dir=output_dir,
        )

        # Calculate total time elapsed
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")

        # 3) Compare using Visualizer
        Visualizer.plot_fidelity_comparison(fidelity_no_opt, fidelity_opt, circuit_name, output_dir)

        # Log total time
        logger.write_experiment_time(total_time)

    except Exception as e:
        print("\n[ERROR] An error occurred during execution!", file=sys.stderr)
        print(f"[DETAILS] {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)  # 🔴 Exit with an error code

if __name__ == "__main__":
    main()
