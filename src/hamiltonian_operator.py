"""
This class is intended to be used directly with Pauli matrices

-parameters
-operator list

-dot
"""
from quantum_tools.quantum_programming_language.src.pauli_strings import *
from scipy import linalg


class HamiltonianOperator:
    def __init__(self, n):
        self.__n = n
        self.hamiltonian_operators = []  # type: list[[float, list[PauliString]]]

    def append(self, param: float, hamiltonian_term_list: list) -> None:
        """
        :param param: float, the parameter associated with the specified Hamiltonian term list.
        :param hamiltonian_term_list: list[PauliString], the list of Hamiltonian terms associated with the given param.
        """
        param = float(param)
        assert isinstance(hamiltonian_term_list, list), "Error. Hamiltonian term list must be be a list."
        assert len(hamiltonian_term_list) > 0, "Error. Hamiltonian term list must include at least one term."
        all_terms_of_correct_type = not (False in [isinstance(term, PauliString) for term in hamiltonian_term_list])
        assert all_terms_of_correct_type, "Error. All Hamiltonian terms in the term list must be of type PauliString."
        self.hamiltonian_operators.append([param, hamiltonian_term_list])

    def add_weighted_term_list(self, new_hamiltonian_operators) -> None:
        """
        Adds the given list of [parameter, [hamiltonian operators]] terms to the existing hamiltonian operator list.
        NOTE: This does not create a deep copy of the given list.
        :param new_hamiltonian_operators: list[[float, list[PauliString]]
        """
        assert isinstance(new_hamiltonian_operators, list), "Error. Input must be a list."
        assert len(new_hamiltonian_operators) > 0, "Error. New hamiltonian term list must include at least one term."
        assert False not in [len(term) == 2
                             for term in new_hamiltonian_operators], "Error. Invalid format for elements in list."
        assert False not in [isinstance(elem[1], list)
                             for elem in new_hamiltonian_operators], "Error. Invalid input format."
        assert False not in [len(elem[1]) > 0
                             for elem in new_hamiltonian_operators], "Error. A hamiltonian term list is empty."
        assert False not in [isinstance(term[0], float) or isinstance(term[0], int)
                             for term in new_hamiltonian_operators], "Error. 1st element of each list must be numeric."
        # TODO NOTE: Assuming this method creates deep copies of the given list may cause correctness bugs.
        self.hamiltonian_operators += new_hamiltonian_operators

    def get_weighted_operator(self):
        """
        Returns \vec{\theta} \cdot  \vec{H}; i.e. an NxN operator consisting of the weighted sum of all of the operators
        contained in the operator list (weighted by the corresponding elements in the parameter list).
        :return: An NxN operator.
        """
        M = np.zeros((2 ** self.__n, 2 ** self.__n), dtype=np.complex_)
        for i in range(len(self.hamiltonian_operators)):
            param_i = self.hamiltonian_operators[i][0]
            # Yeild the sum of all the operators associated with the given parameter
            new_M_term = param_i * sum([term.get_matrix() for term in self.hamiltonian_operators[i][1]])
            M += new_M_term
        return M

    def get_unitary(self):
        """
        :return: Let `M = self.get_total_operator()`, then this method returns exp(-i M).
        """
        return linalg.expm(-1j * self.get_weighted_operator())

    def set_parameters(self, new_params: list) -> None:
        assert len(new_params) == len(self.hamiltonian_operators), "Error. Parameters given are of incorrect length."
        for i in range(len(new_params)):
            self.hamiltonian_operators[i][0] = new_params[i]

    def get_parameters(self):
        """
        :return: list of the parameters.
        """
        params = []
        for term in self.hamiltonian_operators:
            params.append(term[0])
        return params

    def __len__(self):
        """
        Yields the number of **parameters** present in the top-level of the Hamiltonian, noting that multiple operators
        might be assigned to a given parameter.
        @TODO A possible future extension may make this return the total number of parameters (including nested ones).
        :return: the number of top-level parameters present.
        """
        return len(self.hamiltonian_operators)

