import argparse

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import GroverCircuit
from src import Evaluator, GeneticOptimizer, NoiseModel, Visualizer


def run_algorithm(quantum_circuit, num_qubits, circuit_name):
    """
    Function to run the optimization and analysis for a given quantum circuit.
    """
    # Define solver options
    solver_options = Options(nsteps=100000, store_states=True)

    # Create the noise model
    noise_model = NoiseModel(
        num_qubits,
        t1=50.0,
        t2=30.0,
        bit_flip_prob=0.02,
        phase_flip_prob=0.02,
    )

    # Create the evaluator
    evaluator = Evaluator(quantum_circuit, noise_model, solver_options)

    # Run the genetic optimizer
    optimizer = GeneticOptimizer(evaluator)
    pop = optimizer.run()
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

    # Visualization
    Visualizer.plot_pulses(processor_optimized, f"Optimized Pulses for {circuit_name} with Noise")
    Visualizer.plot_fidelity_evolution(optimizer.logbook)
    Visualizer.plot_histogram_fidelities(pop)

    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(pop, parameters)
    Visualizer.plot_correlation(pop, parameters)
    Visualizer.plot_histogram_parameters(pop, parameters)


def main():
    """
    Main function to run either Grover or Deutsch-Jozsa based on command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run quantum algorithms.")
    parser.add_argument("--algorithm", type=str, choices=["grover", "deutsch-jozsa"], required=True,
                        help="Specify which algorithm to run: 'grover' or 'deutsch-jozsa'")
    args = parser.parse_args()

    if args.algorithm == "grover":
        num_qubits = 4  # Grover uses 4 qubits
        print("\n--- Running Grover's Algorithm ---")
        quantum_circuit = GroverCircuit(num_qubits)
        run_algorithm(quantum_circuit, num_qubits, "Grover")
    elif args.algorithm == "deutsch-jozsa":
        num_qubits = 3  # Deutsch-Jozsa uses 3 qubits
        print("\n--- Running Deutsch-Jozsa Algorithm ---")
        quantum_circuit = DeutschJozsaCircuit(num_qubits)
        run_algorithm(quantum_circuit, num_qubits, "Deutsch-Jozsa")


if __name__ == "__main__":
    main()
