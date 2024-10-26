import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    """
    Handles plotting and visualization of genetic algorithm optimization results.
    """

    @staticmethod
    def plot_pulses(processor, title):
        """
        Plots the optimized pulses for each control in the quantum circuit.

        Args:
            processor (OptPulseProcessor): The pulse processor with optimized pulses.
            title (str): The title of the plot.

        """
        coeffs = processor.coeffs
        tlist = processor.get_full_tlist()
        control_labels = processor.get_control_labels()
        plt.figure(figsize=(12, 6))
        for i, coeff in enumerate(coeffs):
            plt.step(
                tlist[:-1],
                coeff,
                where="post",
                label=f"Control {i}: {control_labels[i]}",
            )
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(visible=True)  # Changed from plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_fidelity_evolution(logbook):
        """
        Plots the evolution of average and maximum fidelity over generations.

        Args:
            logbook (deap.tools.Logbook): The logbook containing generation statistics.

        """
        genetic_book = pd.DataFrame(logbook)
        plt.figure(figsize=(10, 5))
        plt.plot(genetic_book["gen"], genetic_book["avg"], label="Average Fidelity")
        plt.plot(genetic_book["gen"], genetic_book["max"], label="Max Fidelity")
        plt.fill_between(
            genetic_book["gen"],
            genetic_book["avg"] - genetic_book["std"],
            genetic_book["avg"] + genetic_book["std"],
            alpha=0.2,
            label="Standard Deviation",
        )
        plt.xlabel("Generation")
        plt.ylabel("Fidelity")
        plt.title("Fidelity Evolution during Optimization")
        plt.legend()
        plt.grid(visible=True)  # Changed from plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_histogram_fidelities(pop):
        """
        Plots a histogram of fidelities in the final population.

        Args:
            pop (list): The final population of individuals.

        """
        final_fidelities = [ind.fitness.values[0] for ind in pop]
        plt.figure(figsize=(8, 6))
        plt.hist(final_fidelities, bins=20, color="skyblue", edgecolor="black")
        plt.title("Distribution of Fidelities in Final Population")
        plt.xlabel("Fidelity")
        plt.ylabel("Number of Individuals")
        plt.grid(visible=True)  # Changed from plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_parameter_evolution(pop, parameters):
        """
        Plots the evolution of parameters (num_tslots and evo_time) for each gate.

        Args:
            pop (list): The final population of individuals.
            parameters (list): List of gate names (e.g., ["SNOT", "X", "CNOT"]).

        """
        for gate in parameters:
            num_tslots = [ind[gate]["num_tslots"] for ind in pop]
            evo_time = [ind[gate]["evo_time"] for ind in pop]

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sns.boxplot(y=num_tslots, color="lightgreen")
            plt.title(f"Evolution of num_tslots for {gate}")
            plt.ylabel("num_tslots")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.subplot(1, 2, 2)
            sns.boxplot(y=evo_time, color="lightcoral")
            plt.title(f"Evolution of evo_time for {gate}")
            plt.ylabel("evo_time")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_correlation(pop, parameters):
        """
        Plots a correlation matrix and scatter plots between parameters and fidelity.

        Args:
            pop (list): The final population of individuals.
            parameters (list): List of gate names (e.g., ["SNOT", "X", "CNOT"]).

        """
        population_data = []
        for ind in pop:
            data = {
                "Fidelity": ind.fitness.values[0],
                "SNOT_num_tslots": ind["SNOT"]["num_tslots"],
                "SNOT_evo_time": ind["SNOT"]["evo_time"],
                "X_num_tslots": ind["X"]["num_tslots"],
                "X_evo_time": ind["X"]["evo_time"],
                "CNOT_num_tslots": ind["CNOT"]["num_tslots"],
                "CNOT_evo_time": ind["CNOT"]["evo_time"],
            }
            population_data.append(data)

        genetic_book_population = pd.DataFrame(population_data)

        # Correlation Matrix
        plt.figure(figsize=(10, 8))
        corr = genetic_book_population.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix between Parameters and Fidelity")
        plt.tight_layout()
        plt.show()

        # Scatter Plots of Parameters vs Fidelity
        for gate in parameters:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.scatterplot(
                x=f"{gate}_num_tslots", y="Fidelity", data=genetic_book_population, alpha=0.7
            )
            plt.title(f"Fidelity vs num_tslots for {gate}")
            plt.xlabel("num_tslots")
            plt.ylabel("Fidelity")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.subplot(1, 2, 2)
            sns.scatterplot(
                x=f"{gate}_evo_time", y="Fidelity", data=genetic_book_population, alpha=0.7
            )
            plt.title(f"Fidelity vs evo_time for {gate}")
            plt.xlabel("evo_time")
            plt.ylabel("Fidelity")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_histogram_parameters(pop, parameters):
        """
        Plots histograms of parameters (num_tslots and evo_time) for each gate.

        Args:
            pop (list): The final population of individuals.
            parameters (list): List of gate names (e.g., ["SNOT", "X", "CNOT"]).

        """
        population_data = []
        for ind in pop:
            data = {
                "SNOT_num_tslots": ind["SNOT"]["num_tslots"],
                "SNOT_evo_time": ind["SNOT"]["evo_time"],
                "X_num_tslots": ind["X"]["num_tslots"],
                "X_evo_time": ind["X"]["evo_time"],
                "CNOT_num_tslots": ind["CNOT"]["num_tslots"],
                "CNOT_evo_time": ind["CNOT"]["evo_time"],
            }
            population_data.append(data)

        genetic_book_population = pd.DataFrame(population_data)

        for gate in parameters:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.histplot(
                genetic_book_population[f"{gate}_num_tslots"],
                bins=10,
                kde=True,
                color="green",
                edgecolor="black",
            )
            plt.title(f"Distribution of num_tslots for {gate}")
            plt.xlabel("num_tslots")
            plt.ylabel("Frequency")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.subplot(1, 2, 2)
            sns.histplot(
                genetic_book_population[f"{gate}_evo_time"],
                bins=10,
                kde=True,
                color="purple",
                edgecolor="black",
            )
            plt.title(f"Distribution of evo_time for {gate}")
            plt.xlabel("evo_time")
            plt.ylabel("Frequency")
            plt.grid(visible=True)  # Changed from plt.grid(True)

            plt.tight_layout()
            plt.show()

    @staticmethod
    def compare_pulses(processor_initial, processor_optimized):
        """
        Compares the initial and optimized pulses for each control in the quantum circuit.

        Args:
            processor_initial (OptPulseProcessor): The pulse processor with initial pulses.
            processor_optimized (OptPulseProcessor): The pulse processor with optimized pulses.

        """
        tlist_initial = processor_initial.get_full_tlist()
        tlist_optimized = processor_optimized.get_full_tlist()
        coeffs_initial = processor_initial.coeffs
        coeffs_optimized = processor_optimized.coeffs

        plt.figure(figsize=(12, 6))
        num_controls = len(coeffs_initial)
        for i in range(num_controls):
            # Plot initial pulse
            plt.step(
                tlist_initial[:-1],
                coeffs_initial[i],
                where="post",
                label=f"Control {i} Initial",
                linestyle="-",
                alpha=0.7,
            )
            # Plot optimized pulse
            plt.step(
                tlist_optimized[:-1],
                coeffs_optimized[i],
                where="post",
                linestyle="--",
                label=f"Control {i} Optimized",
                alpha=0.7,
            )
        plt.title("Comparison of Initial and Optimized Pulses")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(visible=True)  # Changed from plt.grid(True)
        plt.tight_layout()
        plt.show()
