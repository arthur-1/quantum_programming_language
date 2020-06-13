"""
Defines the Hamiltonian Class.
"""
import numpy as np
import scipy as scp
from scipy.sparse import sparsetools


class Hamiltonian:
    def __init__(self, num_qubits: int):
        """
        Defines the dimension of this Hamiltonian.
        Initializes the Hamiltonian as all zeros sparse matrix.
        """
        self.__n = num_qubits
        self.__N = 2 ** self.n
        self._H = scp.sparse.csr_matrix((self.N, self.N))

    def __getitem__(self, indices):
        """
        Get the matrix element of the Hamiltonian specified by the key.

        :param key: (i, j) tuple containing the desired index.

        :return: H[key[0], key[1]]
        """
        raise NotImplementedError

    def __setitem__(self, key, value):
        """
        -Adds an element to the sparse matrix at index (key[0], key[1]) with value `value`.
        -Additionally adds element to the sparse matrix at index (key[1], key[0]) with value `value^*`.
        -Creates the corresponding Pauli strings required to place the two elements.
        The number of Pauli terms created is not final, but current approaches allow the creation of
        O(1) terms for diagonal elements, and O(N) terms for significantly off-diagonal elements.

        :param key: (i, j) tuple containing the desired index. Requirement that i <= j (user only handles upper
                    triangular block).

        :param value: value to insert at the index.
        """
        raise NotImplementedError

    def duplicate(self):
        """
        @TODO Determine if this modification should be in-place, or if it should return a new H'.
        Yields H' such that H' = direct_sum(H, H).
        Thus, H goes from N x N to N' x N' with N' = 2N.
        Implemented on the underlying quantum gates with,
        H' = I \otimes H.
        :return: H'
        """
        raise NotImplementedError

    def get_pauli_terms(self):
        """
        Returns the Hamiltonian's representation as a list of Pauli terms.
        @TODO If it is decided to also use the elements of SU(4), this could potentially include CX gates, etc.
        """
        raise NotImplementedError

    def get_matrix(self):
        """
        Yields the matrix representation for the Hamiltonian. This will be a sparse matrix if possible.
        """

    def __set_arbitrary_element(self, key, value):
        """
        Sets `H[i, j] = value`. This method should be used when there is no known way to place the element that is more
        efficient. For an NxN Hamiltonian, this method produces \Theta(N) Pauli terms.

        :param key: (i, j) tuple giving the index into the Hamiltonian at which `value` will be placed.

        :param value: the value to place at H[i, j].
        """
        raise NotImplementedError


