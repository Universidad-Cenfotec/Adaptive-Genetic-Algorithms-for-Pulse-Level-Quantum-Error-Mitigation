import argparse
import warnings
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
    quantum_circuit, num_qubits, circuit_name, noise_model
):
    """
    Runs the circuit under noise WITHOUT GA optimization, returns fidelity.
    """
    logger = CSVLogger(circuit_name)
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
    noise_model
):
    """
    Runs the genetic optimization on the circuit under noise, returns fidelity.
    """
    logger = CSVLogger(circuit_name)
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
    pop, logbook = optimizer.run(csv_logger=logger, csv_filename=f"{circuit_name}_log.csv")
    best_individual = optimizer.hall_of_fame[0]
    best_fidelity = evaluator.evaluate(best_individual)
    if isinstance(best_fidelity, list | tuple):
        best_fidelity = best_fidelity[0]

    print(f"\nBest individual for {circuit_name}: {best_individual}")
    print(f"Fidelity (best_individual): {best_fidelity:.4f}")

    # Use best individual to set pulses
    processor_optimized = OptPulseProcessor(
        num_qubits=num_qubits, model=SpinChainModel(num_qubits, setup="linear")
    )
    processor_optimized.load_circuit(
        quantum_circuit.circuit, setting_args=best_individual, merge_gates=False
    )
    result_opt = processor_optimized.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops,
    )
    final_fidelity_with_noise = fidelity(result_opt.states[-1], quantum_circuit.target_state)
    print(f"Fidelity with noise for {circuit_name} best_individual: {final_fidelity_with_noise:.4f}")

    # Write summary CSV
    logger.write_summary_optimization(
        noise_model,
        best_individual,
        best_fidelity,
        final_fidelity_with_noise,
        population_size,
        num_generations
    )
    # Write pulses CSV
    logger.write_pulses(processor_optimized)

    # Create an output dir for images
    output_dir = Path("output_circuits") / circuit_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualization
    Visualizer.plot_pulses(
        processor_optimized,
        f"Optimized Pulses for {circuit_name}",
        filename=output_dir / f"{circuit_name}_optimized_pulses.jpg"
    )
    Visualizer.plot_fidelity_evolution(logbook, filename=output_dir / f"{circuit_name}_fidelity_evolution.jpg")
    Visualizer.plot_histogram_fidelities(pop, filename=output_dir / f"{circuit_name}_histogram_fidelities.jpg")

    # Parameter plots
    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(pop, parameters, filename_prefix=str(output_dir / f"{circuit_name}_parameter_evolution"))
    Visualizer.plot_correlation(pop, parameters, filename=output_dir / f"{circuit_name}_correlation_matrix.jpg", output_dir=output_dir)
    Visualizer.plot_histogram_parameters(pop, parameters, filename_prefix=str(output_dir / f"{circuit_name}_histogram_parameters"))

    return final_fidelity_with_noise


def main():
    parser = argparse.ArgumentParser(description="Run quantum algorithms with or without GA optimization under noise.")
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

    # Create noise model
    noise_model = NoiseModel(
        num_qubits,
        t1=args.t1,
        t2=args.t2,
        bit_flip_prob=args.bit_flip_prob,
        phase_flip_prob=args.phase_flip_prob,
    )

    # 1) Run WITHOUT optimization
    fidelity_no_opt = run_algorithm_without_optimization(
        quantum_circuit, num_qubits, circuit_name + "_No_Opt", noise_model
    )

    # 2) Run WITH optimization
    fidelity_opt = run_algorithm(
        quantum_circuit, num_qubits, circuit_name + "_With_Opt",
        population_size=args.population_size,
        num_generations=args.num_generations,
        noise_model=noise_model
    )

    # 3) Compare using Visualizer
    Visualizer.plot_fidelity_comparison(fidelity_no_opt, fidelity_opt, circuit_name)


if __name__ == "__main__":
    main()
