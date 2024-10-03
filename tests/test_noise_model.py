# tests/test_noise_model.py
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qutip import Qobj

from src.noise_model import NoiseModel

# Add the root folder of the project to PYTHONPATH
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


class TestNoiseModel(unittest.TestCase):

    @patch("src.noise_model.QuantumUtils.expand_operator")
    def test_c_ops_length_single_qubit(self, mock_expand_operator):
        """
        Verifies that initializing NoiseModel with 1 qubit generates 8 collapse operators.
        """
        num_qubits = 1

        # Configure the mock for expand_operator
        # Each call returns a Qobj with unique data for identification
        mock_expand_operator.return_value = Qobj([[1]])

        # Initialize NoiseModel
        noise_model = NoiseModel(num_qubits=num_qubits)

        # Verify that expand_operator was called 8 times (8 operators per qubit)
        expected_calls = 8 * num_qubits
        self.assertEqual(mock_expand_operator.call_count, expected_calls,
                         f"Expected {expected_calls} calls to expand_operator, but got {mock_expand_operator.call_count}.")

        # Verify that c_ops has 8 operators
        self.assertEqual(len(noise_model.c_ops), expected_calls,
                         f"Expected {expected_calls} collapse operators, but got {len(noise_model.c_ops)}.")

        # Verify that each operator is a correctly scaled Qobj
        t1 = noise_model.t1
        t2 = noise_model.t2
        bit_flip_prob = noise_model.bit_flip_prob
        phase_flip_prob = noise_model.phase_flip_prob

        # Define the expected scaling factors for each type of operator
        expected_factors = [
            np.sqrt(1.0 / t1),                                # Relaxation
            np.sqrt(1.0 / (2 * t2)),                          # Dephasing
            np.sqrt(bit_flip_prob),                           # Bit Flip
            np.sqrt(phase_flip_prob),                         # Phase Flip
            np.sqrt(bit_flip_prob * phase_flip_prob),         # Bit-Phase Flip
            np.sqrt(bit_flip_prob / 3),                       # Depolarizing Bit Flip
            np.sqrt(bit_flip_prob / 3),                       # Depolarizing Phase Flip
            np.sqrt(bit_flip_prob / 3),                        # Depolarizing Bit-Phase Flip
        ]

        for i, op in enumerate(noise_model.c_ops):
            expected_op = expected_factors[i] * Qobj([[1]])
            np.testing.assert_allclose(op.full(), expected_op.full(),
                                       err_msg=f"Collapse operator at index {i} is not correctly scaled.")

    @patch("src.noise_model.QuantumUtils.expand_operator")
    def test_c_ops_length_multiple_qubits(self, mock_expand_operator):
        """
        Verifies that initializing NoiseModel with multiple qubits generates the correct number of collapse operators.
        """
        num_qubits = 2

        # Configure the mock for expand_operator
        mock_expand_operator.return_value = Qobj([[2]])

        # Initialize NoiseModel
        noise_model = NoiseModel(num_qubits=num_qubits)

        # Verify that expand_operator was called 16 times (8 operators per qubit)
        expected_calls = 8 * num_qubits
        self.assertEqual(mock_expand_operator.call_count, expected_calls,
                         f"Expected {expected_calls} calls to expand_operator, but got {mock_expand_operator.call_count}.")

        # Verify that c_ops has 16 operators
        self.assertEqual(len(noise_model.c_ops), expected_calls,
                         f"Expected {expected_calls} collapse operators, but got {len(noise_model.c_ops)}.")

    def test_zero_qubits(self):
        """
        Verifies that initializing NoiseModel with zero qubits results in an empty c_ops list.
        """
        num_qubits = 0

        # Initialize NoiseModel
        noise_model = NoiseModel(num_qubits=num_qubits)

        # Verify that c_ops is empty
        self.assertEqual(len(noise_model.c_ops), 0,
                         "c_ops should be empty when num_qubits is 0.")

    def test_negative_qubits(self):
        """
        Verifies that initializing NoiseModel with a negative number of qubits raises a ValueError.
        """
        num_qubits = -1

        # Attempt to initialize NoiseModel and verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            NoiseModel(num_qubits=num_qubits)

        self.assertIn("must be non-negative", str(context.exception),
                      "Initializing NoiseModel with negative qubits should raise ValueError.")

    @patch("src.noise_model.QuantumUtils.expand_operator")
    def test_custom_parameters(self, mock_expand_operator):
        """
        Verifies that initializing NoiseModel with custom parameters correctly scales the collapse operators.
        """
        num_qubits = 1
        custom_t1 = 60.0
        custom_t2 = 30.0
        custom_bit_flip_prob = 0.1
        custom_phase_flip_prob = 0.2

        # Configure the mock for expand_operator
        mock_expand_operator.return_value = Qobj([[3]])

        # Initialize NoiseModel with custom parameters
        noise_model = NoiseModel(num_qubits=num_qubits,
                                 t1=custom_t1,
                                 t2=custom_t2,
                                 bit_flip_prob=custom_bit_flip_prob,
                                 phase_flip_prob=custom_phase_flip_prob)

        # Verify that expand_operator was called 8 times
        expected_calls = 8 * num_qubits
        self.assertEqual(mock_expand_operator.call_count, expected_calls,
                         f"Expected {expected_calls} calls to expand_operator, but got {mock_expand_operator.call_count}.")

        # Define the expected scaling factors for each type of operator
        expected_factors = [
            np.sqrt(1.0 / custom_t1),                                     # Relaxation
            np.sqrt(1.0 / (2 * custom_t2)),                               # Dephasing
            np.sqrt(custom_bit_flip_prob),                                # Bit Flip
            np.sqrt(custom_phase_flip_prob),                              # Phase Flip
            np.sqrt(custom_bit_flip_prob * custom_phase_flip_prob),      # Bit-Phase Flip
            np.sqrt(custom_bit_flip_prob / 3),                            # Depolarizing Bit Flip
            np.sqrt(custom_bit_flip_prob / 3),                            # Depolarizing Phase Flip
            np.sqrt(custom_bit_flip_prob / 3),                             # Depolarizing Bit-Phase Flip
        ]

        for i, op in enumerate(noise_model.c_ops):
            expected_op = expected_factors[i] * Qobj([[3]])
            np.testing.assert_allclose(op.full(), expected_op.full(),
                                       err_msg=f"Collapse operator at index {i} is not correctly scaled with custom parameters.")

    @patch("src.noise_model.QuantumUtils.expand_operator")
    def test_large_number_of_qubits(self, mock_expand_operator):
        """
        Verifies that NoiseModel can handle a large number of qubits without issues.
        """
        num_qubits = 10

        # Configure the mock for expand_operator
        mock_expand_operator.return_value = Qobj([[4]])

        # Initialize NoiseModel
        noise_model = NoiseModel(num_qubits=num_qubits)

        # Verify that expand_operator was called 80 times (8 operators per qubit)
        expected_calls = 8 * num_qubits
        self.assertEqual(mock_expand_operator.call_count, expected_calls,
                         f"Expected {expected_calls} calls to expand_operator, but got {mock_expand_operator.call_count}.")

        # Verify that c_ops has 80 operators
        self.assertEqual(len(noise_model.c_ops), expected_calls,
                         f"Expected {expected_calls} collapse operators, but got {len(noise_model.c_ops)}.")

        # Optionally, verify scaling for a few operators
        # For brevity, we'll check the first and last operators
        t1 = noise_model.t1
        t2 = noise_model.t2
        bit_flip_prob = noise_model.bit_flip_prob
        phase_flip_prob = noise_model.phase_flip_prob

        expected_factors_first = [
            np.sqrt(1.0 / t1),                                     # Relaxation
            np.sqrt(1.0 / (2 * t2)),                               # Dephasing
            np.sqrt(bit_flip_prob),                                # Bit Flip
            np.sqrt(phase_flip_prob),                              # Phase Flip
            np.sqrt(bit_flip_prob * phase_flip_prob),              # Bit-Phase Flip
            np.sqrt(bit_flip_prob / 3),                            # Depolarizing Bit Flip
            np.sqrt(bit_flip_prob / 3),                            # Depolarizing Phase Flip
            np.sqrt(bit_flip_prob / 3),                             # Depolarizing Bit-Phase Flip
        ]

        # Check the first qubit's first operator (Relaxation)
        first_op = noise_model.c_ops[0]
        expected_first_op = expected_factors_first[0] * Qobj([[4]])
        np.testing.assert_allclose(first_op.full(), expected_first_op.full(),
                                   err_msg="First collapse operator is not correctly scaled.")

        # Check the last qubit's last operator (Depolarizing Bit-Phase Flip)
        last_op = noise_model.c_ops[-1]
        expected_last_op = expected_factors_first[-1] * Qobj([[4]])
        np.testing.assert_allclose(last_op.full(), expected_last_op.full(),
                                   err_msg="Last collapse operator is not correctly scaled.")

    @patch("src.noise_model.QuantumUtils.expand_operator")
    def test_c_ops_types_single_qubit(self, mock_expand_operator):
        """
        Verifies that each type of collapse operator is correctly generated and scaled for a single qubit.
        """
        num_qubits = 1

        # Configure the mock for expand_operator to return unique identifiers
        # Each call returns Qobj([[5 + i]]) where i ranges from 0 to 7
        mock_expand_operator.side_effect = [Qobj([[5 + i]]) for i in range(8)]

        # Initialize NoiseModel
        noise_model = NoiseModel(num_qubits=num_qubits)

        # Define the expected scaling factors for each type of operator
        expected_factors = [
            np.sqrt(1.0 / noise_model.t1),                                # Relaxation
            np.sqrt(1.0 / (2 * noise_model.t2)),                          # Dephasing
            np.sqrt(noise_model.bit_flip_prob),                           # Bit Flip
            np.sqrt(noise_model.phase_flip_prob),                         # Phase Flip
            np.sqrt(noise_model.bit_flip_prob * noise_model.phase_flip_prob),  # Bit-Phase Flip
            np.sqrt(noise_model.bit_flip_prob / 3),                       # Depolarizing Bit Flip
            np.sqrt(noise_model.bit_flip_prob / 3),                       # Depolarizing Phase Flip
            np.sqrt(noise_model.bit_flip_prob / 3),                        # Depolarizing Bit-Phase Flip
        ]

        for i, op in enumerate(noise_model.c_ops):
            expected_op = expected_factors[i] * Qobj([[5 + i]])
            np.testing.assert_allclose(op.full(), expected_op.full(),
                                       err_msg=f"Collapse operator at index {i} is not correctly scaled.")

if __name__ == "__main__":
    unittest.main()
