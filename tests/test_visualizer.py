import unittest
from src.visualizer import Visualizer
from deap.tools import Logbook
import pandas as pd

class TestVisualizer(unittest.TestCase):

    def setUp(self):
        self.logbook = Logbook()
        self.logbook.header = ["gen", "avg", "std", "min", "max"]
        self.logbook.record(gen=0, avg=0.8, std=0.05, min=0.7, max=0.9)
        self.logbook.record(gen=1, avg=0.85, std=0.03, min=0.8, max=0.9)

    def test_plot_fidelity_evolution(self):
        try:
            Visualizer.plot_fidelity_evolution(self.logbook)
        except KeyError as e:
            self.fail(f"plot_fidelity_evolution raised an exception {e}")
        except Exception as e:
            self.fail(f"plot_fidelity_evolution raised an unexpected exception {e}")

if __name__ == "__main__":
    unittest.main()
