import argparse
import cProfile
import warnings
from datetime import datetime
from pathlib import Path

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import GroverCircuit
from src.csv_logger import CSVLogger
from src.evaluator import Evaluator
from src.gate_config import DEFAULT_SETTING_ARGS
from src.genetic_optimizer import GeneticOptimizer
from src.noise_model import NoiseModel
from src.visualizer import Visualizer

warnings.filterwarnings("ignore")

UNSUPPORTED_ALGORITHM_SPECIFIED = "Unsupported algorithm specified."


def run_algorithm_without_optimization(
    quantum_circuit, num_qubits, circuit_name, noise_model, logger
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
        quantum_circuit.circuit, setting_args=DEFAULT_SETTING_ARGS, merge_gates=False
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

    # Run GA with profiling
    pr = cProfile.Profile()
    pr.enable()

    pop, logbook = optimizer.run(csv_logger=logger, csv_filename=output_dir / f"{circuit_name}_log.csv")

    pr.disable()
    pr.dump_stats(output_dir / "profile_results_ga.prof")
    print("\nGA performance profile saved to 'profile_results_ga.prof'.")

    best_individual = optimizer.hall_of_fame[0]
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
    parser = argparse.ArgumentParser(
        description="Run quantum algorithms with or without GA optimization under noise."
    )
    parser.add_argument("--algorithm", type=str, choices=["grover", "deutsch-jozsa"], required=True,
                        help="Specify which algorithm to run: 'grover' or 'deutsch-jozsa'.")
    parser.add_argument("--num_generations", type=int, default=100, help="Generations for GA.")
    parser.add_argument("--population_size", type=int, default=50, help="Population size for GA.")
    parser.add_argument("--t1", type=float, default=50.0, help="T1 relaxation time.")
    parser.add_argument("--t2", type=float, default=30.0, help="T2 dephasing time.")
    parser.add_argument("--bit_flip_prob", type=float, default=0.02, help="Bit-flip probability.")
    parser.add_argument("--phase_flip_prob", type=float, default=0.02, help="Phase-flip probability.")
    args = parser.parse_args()

    if args.algorithm == "grover":
        num_qubits = 4
        circuit_name = "Grover_4Q"
        quantum_circuit = GroverCircuit(num_qubits)
    elif args.algorithm == "deutsch-jozsa":
        num_qubits = 4
        circuit_name = "DeutschJozsa_4Q"
        quantum_circuit = DeutschJozsaCircuit(num_qubits)
    else:
        raise ValueError(UNSUPPORTED_ALGORITHM_SPECIFIED)

    # Create a timestamped output directory
    timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("output_circuits") / f"{circuit_name}_{timestamp_folder}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create noise model
    noise_model = NoiseModel(
        num_qubits,
        t1=args.t1,
        t2=args.t2,
        bit_flip_prob=args.bit_flip_prob,
        phase_flip_prob=args.phase_flip_prob,
    )

    # Create CSV logger with an output_dir
    logger = CSVLogger(circuit_name, output_dir=output_dir)

    # 1) Run WITHOUT optimization
    fidelity_no_opt = run_algorithm_without_optimization(
        quantum_circuit,
        num_qubits,
        circuit_name + "_No_Opt",
        noise_model,
        logger
    )

    # 2) Run WITH optimization (Profiled)
    fidelity_opt = run_algorithm(
        quantum_circuit,
        num_qubits,
        circuit_name + "_With_Opt",
        population_size=args.population_size,
        num_generations=args.num_generations,
        noise_model=noise_model,
        logger=logger,
        output_dir=output_dir,
    )

    # 3) Compare using Visualizer (also writes CSV with comparison)
    Visualizer.plot_fidelity_comparison(fidelity_no_opt, fidelity_opt, circuit_name, output_dir)


if __name__ == "__main__":
    main()
