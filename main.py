import argparse
import csv
import warnings
from pathlib import Path

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import GroverCircuit
from src import Evaluator, GeneticOptimizer, NoiseModel, Visualizer
from src.gate_config import DEFAULT_SETTING_ARGS

warnings.filterwarnings("ignore")

UNSUPORTED_ALGORITHM_SPECIFIED = "Unsupported algorithm specified."

def run_algorithm_without_optimization(quantum_circuit, num_qubits, circuit_name, t1, t2, bit_flip_prob, phase_flip_prob):
    """
    Runs the quantum circuit under noise WITHOUT any genetic optimization to obtain baseline fidelity.
    
    Args:
        quantum_circuit: The quantum circuit object (e.g., GroverCircuit, DeutschJozsaCircuit).
        num_qubits (int): Number of qubits in the circuit.
        circuit_name (str): Name identifier for the circuit (used in filenames).
        t1 (float): T1 relaxation time for the noise model.
        t2 (float): T2 dephasing time for the noise model.
        bit_flip_prob (float): Bit-flip probability for the noise model.
        phase_flip_prob (float): Phase-flip probability for the noise model.
    
    Returns:
        float: Fidelity of the circuit without optimization.

    """  # noqa: W293
    # Define solver options
    solver_options = Options(nsteps=100_000, store_states=True)

    # Create the noise model with specified parameters
    noise_model = NoiseModel(
        num_qubits,
        t1=t1,
        t2=t2,
        bit_flip_prob=bit_flip_prob,
        phase_flip_prob=phase_flip_prob,
    )

    # Initialize the pulse processor without optimization
    processor_no_opt = OptPulseProcessor(
        num_qubits=num_qubits,
        model=SpinChainModel(num_qubits, setup="linear"),
    )
    # Load the circuit with default settings
    processor_no_opt.load_circuit(
        quantum_circuit.circuit,
        setting_args=DEFAULT_SETTING_ARGS,
        merge_gates=False,
    )

    # Run the circuit with noise
    result_no_opt = processor_no_opt.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops,
    )

    # Calculate fidelity against the target state
    fidelity_no_opt = fidelity(result_no_opt.states[-1], quantum_circuit.target_state)

    print(f"\nFidelity WITHOUT optimization for {circuit_name}: {fidelity_no_opt}")

    # Save the results to a CSV file
    summary_file = Path(f"{circuit_name}_summary_no_optimization.csv")
    fieldnames = [
        "circuit_name", "t1", "t2",
        "bit_flip_prob", "phase_flip_prob", "fidelity"
    ]

    write_header = not summary_file.exists()
    with summary_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        writer.writerow({
            "circuit_name": circuit_name,
            "t1": t1,
            "t2": t2,
            "bit_flip_prob": bit_flip_prob,
            "phase_flip_prob": phase_flip_prob,
            "fidelity": fidelity_no_opt
        })

    return fidelity_no_opt


def run_algorithm(quantum_circuit, num_qubits, circuit_name, population_size, num_generations, t1, t2, bit_flip_prob, phase_flip_prob):
    """
    Runs the genetic optimization on the quantum circuit under noise and returns the optimized fidelity.
    
    Args:
        quantum_circuit: The quantum circuit object (e.g., GroverCircuit, DeutschJozsaCircuit).
        num_qubits (int): Numbesr of qubits in the circuit.
        circuit_name (str): Name identifier for the circuit (used in filenames).
        population_size (int): Population size for the genetic algorithm.
        num_generations (int): Number of generations for the genetic algorithm.
        t1 (float): T1 relaxation time for the noise model.
        t2 (float): T2 dephasing time for the noise model.
        bit_flip_prob (float): Bit-flip probability for the noise model.
        phase_flip_prob (float): Phase-flip probability for the noise model.
    
    Returns:
        float: Optimized fidelity of the circuit under noise.

    """  # noqa: W293
    # Define solver options
    solver_options = Options(nsteps=100_000, store_states=True)

    # Create the noise model with specified parameters
    noise_model = NoiseModel(
        num_qubits,
        t1=t1,
        t2=t2,
        bit_flip_prob=bit_flip_prob,
        phase_flip_prob=phase_flip_prob,
    )

    # Create the evaluator
    evaluator = Evaluator(quantum_circuit, noise_model, solver_options)

    # Run the genetic optimizer
    optimizer = GeneticOptimizer(
        evaluator=evaluator,
        population_size=population_size,
        num_generations=num_generations,
        use_default=True
    )
    pop, logbook = optimizer.run(csv_filename=f"{circuit_name}_log.csv")
    best_individual = optimizer.hall_of_fame[0]

    # Evaluate the best individual
    best_fidelity = evaluator.evaluate(best_individual)
    if isinstance(best_fidelity, tuple | list):
        best_fidelity = best_fidelity[0]  # Assuming the first element is fidelity

    print(f"\nBest individual found for {circuit_name}: {best_individual}")
    print(f"Fidelity of the best individual for {circuit_name}: {best_fidelity}")

    # Run the best individual with noise to get optimized pulses
    processor_optimized = OptPulseProcessor(
        num_qubits=num_qubits,
        model=SpinChainModel(num_qubits, setup="linear"),
    )
    processor_optimized.load_circuit(
        quantum_circuit.circuit,
        setting_args=best_individual,
        merge_gates=False,
    )

    result = processor_optimized.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops,
    )
    final_fidelity = fidelity(result.states[-1], quantum_circuit.target_state)
    print(f"Fidelity with the best individual (with noise) for {circuit_name}: {final_fidelity}")

    # Save summary data to a CSV file
    summary_file = Path(f"{circuit_name}_summary_optimization.csv")
    fieldnames = [
        "circuit_name", "population_size", "num_generations", "t1", "t2",
        "bit_flip_prob", "phase_flip_prob", "best_individual", "best_fidelity", "final_fidelity_with_noise"
    ]

    write_header = not summary_file.exists()
    with summary_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        writer.writerow({
            "circuit_name": circuit_name,
            "population_size": population_size,
            "num_generations": num_generations,
            "t1": t1,
            "t2": t2,
            "bit_flip_prob": bit_flip_prob,
            "phase_flip_prob": phase_flip_prob,
            "best_individual": str(best_individual),
            "best_fidelity": best_fidelity,
            "final_fidelity_with_noise": final_fidelity
        })

    # Record the pulses used in the optimized processor
    pulses_file = Path(f"{circuit_name}_pulses_optimization.csv")
    tlist = processor_optimized.get_full_tlist()
    coeffs = processor_optimized.get_full_coeffs()  # List of arrays, one per control channel

    # Prepare headers for the pulses file: time + one column per channel
    pulse_headers = ["time"] + [f"pulse_channel_{i+1}" for i in range(len(coeffs))]

    # Write pulses data
    with pulses_file.open(mode="w", newline="") as pf:
        writer = csv.writer(pf)
        writer.writerow(pulse_headers)
        num_points = len(tlist) - 1
        for idx in range(num_points):
            row = [tlist[idx]] + [c[idx] for c in coeffs]
            writer.writerow(row)

    # Create output directory if it doesn't exist
    output_dir = Path("output_circuits") / circuit_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualizations and saving as .jpg files in the output directory
    Visualizer.plot_pulses(
        processor_optimized,
        f"Optimized Pulses for {circuit_name} with Noise",
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

    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(
        pop,
        parameters,
        filename_prefix=output_dir / f"{circuit_name}_parameter_evolution"
    )
    Visualizer.plot_correlation(
        pop,
        parameters,
        filename=output_dir / f"{circuit_name}_correlation_matrix.jpg"
    )
    Visualizer.plot_histogram_parameters(
        pop,
        parameters,
        filename_prefix=output_dir / f"{circuit_name}_histogram_parameters"
    )

    return final_fidelity


def compare_fidelities(fidelity_no_opt, fidelity_opt, circuit_name):
    """
    Compares fidelities obtained without optimization and with genetic optimization.
    Generates a bar chart to visualize the comparison.

    Args:
        fidelity_no_opt (float): Fidelity without optimization.
        fidelity_opt (float): Fidelity with optimization.
        circuit_name (str): Name identifier for the circuit (used in the plot title).

    """
    import matplotlib.pyplot as plt

    labels = ["No Optimization", "Genetic Optimization"]
    fidelities = [fidelity_no_opt, fidelity_opt]
    colors = ["red", "green"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, fidelities, color=colors)
    plt.ylim(0, 1)
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity Comparison for {circuit_name}")

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha="center", va="bottom")

    # Save the plot
    output_dir = Path("output_circuits") / circuit_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = output_dir / f"{circuit_name}_fidelities_comparison.jpg"
    plt.savefig(plot_filename)
    plt.close()

    print(f"\nFidelity comparison plot saved at {plot_filename}")


def main():
    """
    Main function to run either Grover or Deutsch-Jozsa algorithms.
    It runs each algorithm twice:
      1. Without genetic optimization (baseline fidelity under noise).
      2. With genetic optimization (optimized fidelity under noise).
    Then, it compares and visualizes the fidelities.
    """
    parser = argparse.ArgumentParser(description="Run quantum algorithms with and without genetic optimization under noise.")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["grover", "deutsch-jozsa"],
        required=True,
        help="Specify which algorithm to run: 'grover' or 'deutsch-jozsa'."
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=100,
        help="Number of generations for the genetic algorithm."
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="Population size for the genetic algorithm."
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=50.0,
        help="T1 relaxation time for the noise model."
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=30.0,
        help="T2 dephasing time for the noise model."
    )
    parser.add_argument(
        "--bit_flip_prob",
        type=float,
        default=0.02,
        help="Bit-flip probability for the noise model."
    )
    parser.add_argument(
        "--phase_flip_prob",
        type=float,
        default=0.02,
        help="Phase-flip probability for the noise model."
    )
    args = parser.parse_args()

    # Create an output directory for circuits
    base_output_dir = Path("output_circuits")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Decide which circuit to run based on the algorithm argument
    if args.algorithm == "grover":
        num_qubits = 4  # Grover uses 4 qubits
        circuit_name = "Grover"
        print("\n--- Running Grover's Algorithm ---")
        quantum_circuit = GroverCircuit(num_qubits)
    elif args.algorithm == "deutsch-jozsa":
        num_qubits = 3  # Deutsch-Jozsa uses 3 qubits
        circuit_name = "Deutsch-Jozsa"
        print("\n--- Running Deutsch-Jozsa Algorithm ---")
        quantum_circuit = DeutschJozsaCircuit(num_qubits)
    else:
        raise ValueError(UNSUPORTED_ALGORITHM_SPECIFIED)

    # 1. Run WITHOUT optimization (baseline fidelity under noise)
    print("\n--- Running without Genetic Optimization (Baseline Fidelity) ---")
    fidelity_no_opt = run_algorithm_without_optimization(
        quantum_circuit,
        num_qubits,
        f"{circuit_name}_No_Optimization",
        t1=args.t1,
        t2=args.t2,
        bit_flip_prob=args.bit_flip_prob,
        phase_flip_prob=args.phase_flip_prob
    )

    # 2. Run WITH genetic optimization (optimized fidelity under noise)
    print("\n--- Running with Genetic Optimization (Optimized Fidelity) ---")
    fidelity_opt = run_algorithm(
        quantum_circuit,
        num_qubits,
        f"{circuit_name}_With_Optimization",
        population_size=args.population_size,
        num_generations=args.num_generations,
        t1=args.t1,
        t2=args.t2,
        bit_flip_prob=args.bit_flip_prob,
        phase_flip_prob=args.phase_flip_prob
    )

    # 3. Compare the fidelities
    print("\n--- Comparing Fidelities ---")
    compare_fidelities(fidelity_no_opt, fidelity_opt, circuit_name=circuit_name)

    # Optional: Save comparison results to a CSV
    comparison_file = Path(f"{circuit_name}_fidelity_comparison.csv")
    fieldnames = ["circuit_name", "fidelity_no_optimization", "fidelity_with_optimization", "fidelity_improvement"]

    write_header = not comparison_file.exists()
    with comparison_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        writer.writerow({
            "circuit_name": circuit_name,
            "fidelity_no_optimization": fidelity_no_opt,
            "fidelity_with_optimization": fidelity_opt,
            "fidelity_improvement": fidelity_opt - fidelity_no_opt
        })

    print(f"\nFidelity comparison data saved at {comparison_file}")


if __name__ == "__main__":
    main()
