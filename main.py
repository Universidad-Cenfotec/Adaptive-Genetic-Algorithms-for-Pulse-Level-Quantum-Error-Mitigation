import argparse
import warnings

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import GroverCircuit
from src import Evaluator, GeneticOptimizer, NoiseModel, Visualizer

warnings.filterwarnings("ignore")

def run_algorithm(quantum_circuit, num_qubits, circuit_name, population_size, num_generations, t1, t2, bit_flip_prob, phase_flip_prob):
    """
    Function to run the optimization and analysis for a given quantum circuit.
    """
    # Define solver options
    solver_options = Options(nsteps=100000, store_states=True)

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
    optimizer = GeneticOptimizer(evaluator, population_size=population_size, num_generations=num_generations)
    pop, logbook = optimizer.run(csv_filename=f"{circuit_name}_log.csv")
    best_individual = optimizer.hall_of_fame[0]
    best_fidelity = evaluator.evaluate(best_individual)[0]
    print(f"\nBest individual found for {circuit_name}:", best_individual)
    print(f"Fidelity of the best individual for {circuit_name}:", best_fidelity)

    # Run the best individual with noise to get the optimized pulses
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

    # Visualization and saving to .jpg files
    Visualizer.plot_pulses(processor_optimized, f"Optimized Pulses for {circuit_name} with Noise", filename=f"{circuit_name}_optimized_pulses.jpg")
    Visualizer.plot_fidelity_evolution(logbook, filename=f"{circuit_name}_fidelity_evolution.jpg")
    Visualizer.plot_histogram_fidelities(pop, filename=f"{circuit_name}_histogram_fidelities.jpg")

    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(pop, parameters, filename_prefix=f"{circuit_name}_parameter_evolution")
    Visualizer.plot_correlation(pop, parameters, filename=f"{circuit_name}_correlation_matrix.jpg")
    Visualizer.plot_histogram_parameters(pop, parameters, filename_prefix=f"{circuit_name}_histogram_parameters")

def main():
    """
    Main function to run either Grover or Deutsch-Jozsa based on command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run quantum algorithms.")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["grover", "deutsch-jozsa"],
        required=True,
        help="Specify which algorithm to run: 'grover' or 'deutsch-jozsa'"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=100,
        help="Specify the number of generations for the genetic algorithm"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="Specify the population size for the genetic algorithm"
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=50.0,
        help="Specify the T1 relaxation time for the noise model"
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=30.0,
        help="Specify the T2 dephasing time for the noise model"
    )
    parser.add_argument(
        "--bit_flip_prob",
        type=float,
        default=0.02,
        help="Specify the bit flip probability for the noise model"
    )
    parser.add_argument(
        "--phase_flip_prob",
        type=float,
        default=0.02,
        help="Specify the phase flip probability for the noise model"
    )
    args = parser.parse_args()

    if args.algorithm == "grover":
        num_qubits = 4  # Grover uses 4 qubits
        print("\n--- Running Grover's Algorithm ---")
        quantum_circuit = GroverCircuit(num_qubits)
        run_algorithm(
            quantum_circuit, num_qubits, "Grover",
            population_size=args.population_size,
            num_generations=args.num_generations,
            t1=args.t1,
            t2=args.t2,
            bit_flip_prob=args.bit_flip_prob,
            phase_flip_prob=args.phase_flip_prob
        )
    elif args.algorithm == "deutsch-jozsa":
        num_qubits = 3  # Deutsch-Jozsa uses 3 qubits
        print("\n--- Running Deutsch-Jozsa Algorithm ---")
        quantum_circuit = DeutschJozsaCircuit(num_qubits)
        run_algorithm(
            quantum_circuit, num_qubits, "Deutsch-Jozsa",
            population_size=args.population_size,
            num_generations=args.num_generations,
            t1=args.t1,
            t2=args.t2,
            bit_flip_prob=args.bit_flip_prob,
            phase_flip_prob=args.phase_flip_prob
        )

if __name__ == "__main__":
    main()
