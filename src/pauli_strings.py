import quantum_tools.quantum_tools.utils as utils
import numpy as np


class PauliString:
    """
    TODO ADD TESTS.
    """
    I = 0
    X = 1
    Y = 2
    Z = 3

    def __init__(self, pauli_term_list: list, n):
        """
        Creates a Pauli string.
        This has two possible input formats.

        INPUT FORMAT 1:
        For instance, for n=3, an input of [(2, 1)] corresponds to: I_0 X_1 I_2.
        :param pauli_term_list: List of tuples of form `[(g_i, h_i), ...]` where the first index specifies
                                the qubit subsystem/index upon which the Pauli term specified as the second index
                                acts. Only one Pauli term may be given for.
                                For all i,
                                g_i : int
                                h_i : int
                                0 <= g_i < n
                                1 <= h_i <= 3

        INPUT FORMAT 2:
        :param pauli_term_list: List of the form `[t_0, t_1, ..., t_n]` where t_j gives the type of the gate at qubit
                                index j.

        :param n: the number of qubits in the system.
        """
        assert isinstance(n, int) and n > 0, "Error. Number of qubits, n, must be an integer greater than 0."
        self.n = n
        # Ensure that at least one pauli term is given.
        assert len(pauli_term_list) >= 1, "Error. Must give at least one Pauli term."
        # Ensure that the number of qubits is at least as large as the number of terms given
        assert len(pauli_term_list) <= n, "Error. Cannot have more terms than qubits. Only one Pauli term per index."
        valid_set = [PauliString.I, PauliString.X, PauliString.Y, PauliString.Z]

        # First determine the format of the input. Do so by checking an arbitrary element to see if it is an integer.
        if isinstance(pauli_term_list[0], int):
            # If one element is an integer, all inputs must be integers as it means input format 2 is being used.
            assert False not in [isinstance(pauli_term_list[i], int) for i in range(len(pauli_term_list))],\
                "Error. Input format not recognized. If using format type 2, ensure list contains only integers."
            # Ensure that all of the numbers specified in the list are valid Pauli gates (i.e. are 0, 1, 2, or 3)
            assert False not in [gate in valid_set for gate in pauli_term_list],\
                "Error. Gave Pauli-term index not in the set {I, X, Y, Z}."
            is_format_1 = False

        elif len(pauli_term_list) == 2:
            # Ensure that pauli string list is of correct form; all elements are pairs of ints
            terms_valid = [(len(term) == 2) and (isinstance(term[0], int)) and (isinstance(term[1], int))
                           for term in pauli_term_list]
            qubit_index_list = [term[0] for term in pauli_term_list]
            operator_type_list = [term[1] for term in pauli_term_list]
            assert False not in terms_valid, "Error. Input format invalid."
            # Ensure that the number of qubits is larger than the largest qubit index given
            assert max(qubit_index_list) < n, "Error. Cannot specify an index greater than n."
            # Ensure that all of the specified Pauli terms are in the set {X, Y, Z}.
            terms_valid = [gate in valid_set for gate in operator_type_list]
            assert False not in terms_valid, "Error. Gave Pauli-term index not in the set {I, X, Y, Z}."
            # Ensure that no qubit is assigned more than one term.
            assert len(set(qubit_index_list)) == len(qubit_index_list), "Error. Cannot specify a qubit more than once."
            is_format_1 = True

        else:
            assert len(pauli_term_list[0])
            # Unrecognized input format.
            raise RuntimeError

        # Create the list specifying the Pauli indices acting on which qubits
        # (adding identity for all non-specified terms).
        # Note that input format 2 is already of this form.
        self.__pauli_terms = []
        if is_format_1:
            qubit_index_list = [term[0] for term in pauli_term_list]
            operator_type_list = [term[1] for term in pauli_term_list]
            for i in range(n):
                if i in qubit_index_list:
                    term_index = qubit_index_list.index(i)
                    self.__pauli_terms.append(operator_type_list[term_index])
                else:
                    self.__pauli_terms.append(PauliString.I)

        else:  # Must be input format 2
            self.__pauli_terms = pauli_term_list

        assert len(self.__pauli_terms) == n, "Error. Pauli term list produced does not contain n terms."


    @staticmethod
    def __pauli_index_to_matrix(T):
        if T == PauliGate.I:
            return utils.I
        elif T == PauliGate.X:
            return utils.X
        elif T == PauliGate.Y:
            return utils.Y
        elif T == PauliGate.Z:
            return utils.Z
        else:
            raise RuntimeError

    def get_matrix(self):
        return utils.tensor([self.__pauli_index_to_matrix(term) for term in self.__pauli_terms])

    def get_pauli_string(self):
        return self.__pauli_terms


class PauliGate(PauliString):
    def __init__(self, i, T, n):
        """
        Yields: sigma_{iT}, where all non-i indices are assigned identity.

        E.g. if i = 2, T = 2, n = 3, then:
        sigma_{12} = I_1 Y_2 I_3.

        :param i: Index
        :param T: PauliGate.I, PauliGate.X, PauliGate.Y, PauliGate.Z,
        :param n: Number of qubits.
        """
        super().__init__([(i, T)], n)


"""
Following classes yeild X_{i}, that is, X at index i, and I at all other indices.
"""
class X(PauliGate):
    def __init__(self, i, n):
        super().__init__(i, PauliGate.X, n)


class Y(PauliGate):
    def __init__(self, i, n):
        super().__init__(i, PauliGate.Y, n)


class Z(PauliGate):
    def __init__(self, i, n):
        super().__init__(i, PauliGate.Z, n)


class I(PauliGate):
    def __init__(self, i, n):
        super().__init__(i, PauliGate.I, n)


def dot(M, theta_vec):
    """
    :param M: list of k operators (of size NxN)
    :param theta_vec: list of k parameters
    :return: theta_1 M_1 + ... + theta_k M_k
    """
    assert len(M) == len(theta_vec), "Error. Input lists must have same length."
    N = M[0].shape[0]
    # dtype = np.double if M[0].dtype == np.int else M[0].dtype
    out = np.zeros((N, N), dtype=np.complex_)
    for k in range(len(theta_vec)):
        out += M[k] * theta_vec[k]
    return out
