# tests/test_visualizer.py
import unittest
from unittest.mock import patch, MagicMock
from src.visualizer import Visualizer
from deap.tools import Logbook
import pandas as pd
import matplotlib.pyplot as plt

class TestVisualizer(unittest.TestCase):

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.plt.figure')
    @patch('src.visualizer.plt.step')
    def test_plot_pulses(self, mock_step, mock_figure, mock_show):
        """
        Testea que plot_pulses se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        processor = MagicMock()
        processor.coeffs = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        processor.get_full_tlist.return_value = [0, 1, 2, 3]
        processor.get_control_labels.return_value = ['Control A', 'Control B']
        
        title = "Test Pulse Plot"

        # Ejecutar la función
        try:
            Visualizer.plot_pulses(processor, title)
        except Exception as e:
            self.fail(f"plot_pulses raised an exception {e}")
        
        # Verificar que se llamaron las funciones de matplotlib correctamente
        mock_figure.assert_called_once_with(figsize=(12, 6))
        self.assertEqual(mock_step.call_count, len(processor.coeffs))
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.plt.plot')
    @patch('src.visualizer.plt.fill_between')
    @patch('src.visualizer.plt.figure')
    @patch('src.visualizer.pd.DataFrame')
    def test_plot_fidelity_evolution(self, mock_dataframe, mock_figure, mock_fill_between, mock_plot, mock_show):
        """
        Testea que plot_fidelity_evolution se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        logbook = Logbook()
        logbook.header = ["gen", "avg", "std", "min", "max"]
        logbook.record(gen=0, avg=0.8, std=0.05, min=0.7, max=0.9)
        logbook.record(gen=1, avg=0.85, std=0.03, min=0.8, max=0.9)
        
        # Configurar el mock para pandas DataFrame
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df
        mock_df['gen'] = [0, 1]
        mock_df['avg'] = [0.8, 0.85]
        mock_df['std'] = [0.05, 0.03]
        mock_df['min'] = [0.7, 0.8]
        mock_df['max'] = [0.9, 0.9]

        # Ejecutar la función
        try:
            Visualizer.plot_fidelity_evolution(logbook)
        except Exception as e:
            self.fail(f"plot_fidelity_evolution raised an exception {e}")
        
        # Verificar llamadas a matplotlib
        mock_figure.assert_called_once_with(figsize=(10, 5))
        mock_plot.assert_any_call([0, 1], [0.8, 0.85], label='Average Fidelity')
        mock_plot.assert_any_call([0, 1], [0.9, 0.9], label='Max Fidelity')
        mock_fill_between.assert_called_once()
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.plt.hist')
    @patch('src.visualizer.plt.figure')
    def test_plot_histogram_fidelities(self, mock_figure, mock_hist, mock_show):
        """
        Testea que plot_histogram_fidelities se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        pop = [
            MagicMock(fitness=MagicMock(values=(0.8,))),
            MagicMock(fitness=MagicMock(values=(0.85,))),
            MagicMock(fitness=MagicMock(values=(0.75,)))
        ]

        # Ejecutar la función
        try:
            Visualizer.plot_histogram_fidelities(pop)
        except Exception as e:
            self.fail(f"plot_histogram_fidelities raised an exception {e}")
        
        # Verificar llamadas a matplotlib
        mock_figure.assert_called_once_with(figsize=(8, 6))
        mock_hist.assert_called_once_with([0.8, 0.85, 0.75], bins=20, color='skyblue', edgecolor='black')
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.plt.boxplot')
    @patch('src.visualizer.plt.subplot')
    @patch('src.visualizer.plt.figure')
    def test_plot_parameter_evolution(self, mock_figure, mock_subplot, mock_boxplot, mock_show):
        """
        Testea que plot_parameter_evolution se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        pop = [
            {
                "SNOT": {"num_tslots": 3, "evo_time": 1.5},
                "X": {"num_tslots": 2, "evo_time": 0.5},
                "CNOT": {"num_tslots": 15, "evo_time": 5.0}
            },
            {
                "SNOT": {"num_tslots": 4, "evo_time": 2.0},
                "X": {"num_tslots": 3, "evo_time": 0.7},
                "CNOT": {"num_tslots": 18, "evo_time": 7.0}
            }
        ]
        parameters = ["SNOT", "X", "CNOT"]

        # Ejecutar la función
        try:
            Visualizer.plot_parameter_evolution(pop, parameters)
        except Exception as e:
            self.fail(f"plot_parameter_evolution raised an exception {e}")
        
        # Verificar llamadas a matplotlib
        self.assertEqual(mock_figure.call_count, len(parameters))
        self.assertEqual(mock_subplot.call_count, len(parameters)*2)
        self.assertEqual(mock_boxplot.call_count, len(parameters)*2)
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.seaborn.heatmap')
    @patch('src.visualizer.plt.scatter')
    @patch('src.visualizer.plt.figure')
    @patch('src.visualizer.pd.DataFrame')
    def test_plot_correlation(self, mock_dataframe, mock_figure, mock_scatter, mock_heatmap, mock_show):
        """
        Testea que plot_correlation se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        pop = [
            {
                "SNOT": {"num_tslots": 3, "evo_time": 1.5},
                "X": {"num_tslots": 2, "evo_time": 0.5},
                "CNOT": {"num_tslots": 15, "evo_time": 5.0},
                "fitness": MagicMock(values=(0.8,))
            },
            {
                "SNOT": {"num_tslots": 4, "evo_time": 2.0},
                "X": {"num_tslots": 3, "evo_time": 0.7},
                "CNOT": {"num_tslots": 18, "evo_time": 7.0},
                "fitness": MagicMock(values=(0.85,))
            }
        ]
        parameters = ["SNOT", "X", "CNOT"]

        # Configurar el mock para pandas DataFrame
        mock_df = MagicMock()
        mock_dataframe.return_value = mock_df
        mock_df.corr.return_value = pd.DataFrame({
            'Fidelity': [1.0, 0.5, 0.3, 0.4, 0.2, 0.1],
            'SNOT_num_tslots': [0.5, 1.0, 0.3, 0.4, 0.2, 0.1],
            'SNOT_evo_time': [0.3, 0.5, 1.0, 0.6, 0.2, 0.1],
            'X_num_tslots': [0.3, 0.2, 0.5, 0.4, 0.1, 0.0],
            'X_evo_time': [0.1, 0.2, 0.4, 0.5, 0.3, 0.0],
            'CNOT_num_tslots': [0.2, 0.1, 0.3, 0.6, 1.0, 0.4],
            'CNOT_evo_time': [0.1, 0.0, 0.2, 0.3, 0.4, 1.0]
        })

        # Ejecutar la función
        try:
            Visualizer.plot_correlation(pop, parameters)
        except Exception as e:
            self.fail(f"plot_correlation raised an exception {e}")
        
        # Verificar llamadas a matplotlib y seaborn
        mock_figure.assert_called_once_with(figsize=(10, 8))
        mock_heatmap.assert_called_once()
        self.assertEqual(mock_scatter.call_count, len(parameters)*2)
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.seaborn.histplot')
    @patch('src.visualizer.plt.subplot')
    @patch('src.visualizer.plt.figure')
    def test_plot_histogram_parameters(self, mock_figure, mock_subplot, mock_histplot, mock_show):
        """
        Testea que plot_histogram_parameters se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        pop = [
            {
                "SNOT": {"num_tslots": 3, "evo_time": 1.5},
                "X": {"num_tslots": 2, "evo_time": 0.5},
                "CNOT": {"num_tslots": 15, "evo_time": 5.0}
            },
            {
                "SNOT": {"num_tslots": 4, "evo_time": 2.0},
                "X": {"num_tslots": 3, "evo_time": 0.7},
                "CNOT": {"num_tslots": 18, "evo_time": 7.0}
            }
        ]
        parameters = ["SNOT", "X", "CNOT"]

        # Ejecutar la función
        try:
            Visualizer.plot_histogram_parameters(pop, parameters)
        except Exception as e:
            self.fail(f"plot_histogram_parameters raised an exception {e}")
        
        # Verificar llamadas a matplotlib y seaborn
        self.assertEqual(mock_figure.call_count, len(parameters))
        self.assertEqual(mock_subplot.call_count, len(parameters)*2)
        self.assertEqual(mock_histplot.call_count, len(parameters)*2)
        mock_show.assert_called_once()

    @patch('src.visualizer.plt.show')
    @patch('src.visualizer.plt.step')
    @patch('src.visualizer.plt.figure')
    def test_compare_pulses(self, mock_figure, mock_step, mock_show):
        """
        Testea que compare_pulses se ejecute correctamente sin errores.
        """
        # Configurar datos de prueba
        processor_initial = MagicMock()
        processor_initial.get_full_tlist.return_value = [0, 1, 2, 3]
        processor_initial.coeffs = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]

        processor_optimized = MagicMock()
        processor_optimized.get_full_tlist.return_value = [0, 1, 2, 3]
        processor_optimized.coeffs = [
            [0.15, 0.25, 0.35],
            [0.45, 0.55, 0.65]
        ]

        # Ejecutar la función
        try:
            Visualizer.compare_pulses(processor_initial, processor_optimized)
        except Exception as e:
            self.fail(f"compare_pulses raised an exception {e}")
        
        # Verificar llamadas a matplotlib
        mock_figure.assert_called_once_with(figsize=(12, 6))
        self.assertEqual(mock_step.call_count, len(processor_initial.coeffs)*2)
        mock_show.assert_called_once()

if __name__ == "__main__":
    unittest.main()
