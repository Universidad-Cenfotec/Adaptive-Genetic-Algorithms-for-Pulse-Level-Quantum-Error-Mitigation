# src/main.py

from qutip import fidelity, Options
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from src import QuantumCircuit
from src import NoiseModel
from src import Evaluator
from src import GeneticOptimizer
from src import Visualizer


def main():
    """
    Main function to run the optimization and analysis.
    """
    NUM_QUBITS = 3
    # Create the quantum circuit
    quantum_circuit = QuantumCircuit(NUM_QUBITS)
    # Define solver options
    solver_options = Options(nsteps=100000, store_states=True)
    # Create the noise model
    noise_model = NoiseModel(
        NUM_QUBITS,
        T1=30.0,
        T2=15.0,
        bit_flip_prob=0.05,
        phase_flip_prob=0.05
    )
    # Create the evaluator
    evaluator = Evaluator(quantum_circuit, noise_model, solver_options)
    # Run the genetic optimizer
    optimizer = GeneticOptimizer(evaluator)
    pop = optimizer.run()
    best_individual = optimizer.hall_of_fame[0]
    best_fidelity = evaluator.evaluate(best_individual)[0]
    print("\nBest individual found:", best_individual)
    print("Fidelity of the best individual:", best_fidelity)
    
    # Run the best individual with noise to get the optimized pulses
    processor_optimized = OptPulseProcessor(
        num_qubits=NUM_QUBITS,
        model=SpinChainModel(NUM_QUBITS, setup="linear")
    )
    processor_optimized.load_circuit(
        quantum_circuit.circuit,
        setting_args=best_individual,
        merge_gates=False
    )
    result = processor_optimized.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops
    )
    final_fidelity = fidelity(result.states[-1], quantum_circuit.target_state)
    print("Fidelity with the best individual (with noise):", final_fidelity)
    
    # Visualization
    Visualizer.plot_pulses(processor_optimized, 'Optimized Pulses for Deutsch-Jozsa with Noise')
    Visualizer.plot_fidelity_evolution(optimizer.logbook)
    Visualizer.plot_histogram_fidelities(pop)
    
    parameters = ['SNOT', 'X', 'CNOT']
    Visualizer.plot_parameter_evolution(pop, parameters)
    Visualizer.plot_correlation(pop, parameters)
    Visualizer.plot_histogram_parameters(pop, parameters)
    
    # Comparing with initial parameters
    initial_individual = {
        "SNOT": {
            "num_tslots": 5,
            "evo_time": 1.5
        },
        "X": {
            "num_tslots": 3,
            "evo_time": 0.5
        },
        "CNOT": {
            "num_tslots": 10,
            "evo_time": 5.0
        }
    }
    processor_initial = OptPulseProcessor(
        num_qubits=NUM_QUBITS,
        model=SpinChainModel(NUM_QUBITS, setup="linear")
    )
    processor_initial.load_circuit(
        quantum_circuit.circuit,
        setting_args=initial_individual,
        merge_gates=False
    )
    result_initial = processor_initial.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops
    )
    final_fidelity_initial = fidelity(result_initial.states[-1], quantum_circuit.target_state)
    print("Fidelity with initial parameters (with noise):", final_fidelity_initial)
    
    # Comparing Pulses
    Visualizer.plot_pulses(processor_initial, 'Initial Pulses for Deutsch-Jozsa with Noise')
    # Comparing Pulses Side by Side
    Visualizer.compare_pulses(processor_initial, processor_optimized)


if __name__ == "__main__":
    main()