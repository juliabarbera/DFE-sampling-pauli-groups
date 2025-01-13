import numpy as np 
from qutip import rand_ket_haar, w_state, ghz_state
import random
from itertools import permutations, product
import expectation_values as expectvals
import time
from copy import deepcopy

""" ---------------------- Implementation of Direct Fidelity Estimation (DFE) with grouping Class ----------------------- """

class DFE:
    """
    Direct Fidelity Estimation (DFE) with Pauli operator grouping techniques. 
    This class generates noisy quantum states, computes expectations of Pauli strings, 
    groups commuting Pauli operators, and estimates fidelity through projections.
    """
    def __init__(self, eps, delta, beta, qubits, p, state):
        super(DFE, self).__init__()
        """
        Initializes the DFE class with the given parameters and quantum state configuration.
        
        Parameters:
        - eps: The error tolerance for fidelity estimation.
        - delta: The confidence interval for the estimation.
        - beta: The truncation threshold for fidelity estimation.
        - qubits: The number of qubits in the quantum system.
        - p: The depolarizing noise level for the quantum state.
        - state: The type of quantum state to generate (e.g., 'haar_state', 'w_state', or 'ghz_state').
        """
        self.eps = eps  
        self.delta = delta 
        self.beta = beta  
        self.qubits = qubits  
        self.p = p  
        self.d = 2 ** qubits  
        self.l = int(np.round(1 / (eps ** 2 * delta)))  # Sample size
        self.state = state  


    def noisy_state(self, state): 
        """
        Generates a noisy quantum state based on the specified type (Haar, W-state, or GHZ-state).
        Adds depolarizing noise to the state and returns the true state, noisy state, and its density matrix.
        
        Parameters:
        - state: The type of state to generate ('haar_state', 'w_state', 'ghz_state').

        Returns:
        - true_matrix: The true state density matrix without noise.
        - rhoInput: The noisy state with depolarizing noise added.
        - true_state: The true quantum state as a vector.
        """
        if state == "haar_state": 
            true_state = np.array(rand_ket_haar(self.d))
        elif state == "w_state":
            true_state = np.array(w_state(self.qubits))
        elif state == "ghz_state":
            true_state = np.array(ghz_state(self.qubits))

        # Normalize and add depolarizing noise to the true state
        true_matrix = np.outer(true_state.conj().T, true_state)
        true_matrix = true_matrix/np.trace(true_matrix) 
        true_matrix = (true_matrix + true_matrix.conj().T)/2 
        rhoInput =  self.p * np.eye(self.d)/self.d + (1 - self.p) * np.array(true_matrix)
        rhoInput = (rhoInput + rhoInput.conj().T)/2 # Ensure Hermitian matrix
        return true_matrix, rhoInput, true_state
    

    def generate_pauli_strings(self):
        """
        Generates all possible Pauli strings (tensor products of Pauli operators) for the given number of qubits.
        
        Returns:
        - tensor_products: A list of tensor products of Pauli matrices (X, Y, Z, I) for all qubits.
        """
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        paulis = [X, Y, Z, I]
        pauli_combinations = permutations(paulis)  # Generate all combinations of Pauli operators

        pauli_combinations = product(paulis, repeat = self.qubits)
        tensor_products = []
        for combination in pauli_combinations:
            tensor_product = combination[0]
            for pauli in combination[1:]:
                tensor_product = np.kron(tensor_product, pauli)  # Compute the tensor product for each combination
            tensor_products.append(tensor_product)
        return tensor_products


    def probs_and_xki(self, psi, tensor_products):
        """
        Computes the expectation values (x_ki) for each Pauli string on the given quantum state.
        
        Parameters:
        - psi: The quantum state as a vector.
        - tensor_products: The list of Pauli strings (X, Y, Z, I format) for which to compute expectations.

        Returns:
        - list_x_ki: A list of expectation values for each Pauli string.
        """
        list_x_ki = []
        pauli_mapping = {'X': 0, 'Y': 1, 'Z': 2, 'I':3}
        print("Generating list probabilities...")
        for k in range(len(tensor_products)):
            idx = [tuple([pauli_mapping[char] for char in tensor_products[k]])]
            pauli = expectvals.string2pauli(idx)[:, :, :, 0]  # Convert Pauli string to matrix form
            x_ki = (np.real(psi.conj().T @ expectvals.kron_vec_prod(pauli, psi) )/ np.sqrt(self.d)).item()  # Compute expectation
            list_x_ki.append(x_ki)
        return list_x_ki


    def generate_combinations(self, n):
        """
        Generates all possible combinations of Pauli operators ('X', 'Y', 'Z', 'I') of length n.
        
        Parameters:
        - n: The number of qubits (length of each Pauli string).
        
        Returns:
        - combis: A list of tuples containing all combinations of Pauli operators.
        """
        letters = ['X', 'Y', 'Z', 'I']
        print("Generating combinations...")
        combis = list(product(letters, repeat=n))
        return combis


    def commutes(self, pauli_list, pauli_a, condition):
        """
        Checks if a Pauli operator commutes with a list of Pauli operators under specific conditions (QWC or FC).
        
        Parameters:
        - pauli_list: A list of Pauli operators.
        - pauli_a: The Pauli operator to check against the list.
        - condition: The commuting condition ('qwc' for qubit-wise commuting, 'fc' for fully commuting).
        
        Returns:
        - True if the Pauli operator commutes with the list under the given condition, False otherwise.
        """
        if condition == "qwc": # Qubit-wise commuting
            for pauli_b in pauli_list:
                if pauli_a == pauli_b:
                    return False
                for a, b in zip(pauli_a, pauli_b):
                    if (a == b) or (a == "I") or (b == "I"):
                        continue
                    return False
            return True
        elif condition == "fc": # Fully commuting
            for pauli_b in pauli_list:
                if pauli_a == pauli_b:
                    return False
                
                anti_commute_count = 0
                for p_a, p_b in zip(pauli_a, pauli_b):
                    if p_a != 'I' and p_b != 'I' and p_a != p_b:
                        anti_commute_count += 1
                
                if not anti_commute_count % 2 == 0: # If odd number of anti-commutes, don't commute
                    return False
            return True

    
    def sorted_insertion_grouping(self, pauli_list, x_ki_list, condition):
        """
        Groups Pauli operators based on whether they commute with each other, 
        using sorted insertion (SI) https://arxiv.org/abs/1908.06942.
        SI sorts pauli strings in decreasing order by their weights and then groups them.
        
        Parameters:
        - pauli_list: A list of Pauli operators to be grouped.
        - x_ki_list: A list of expectation values corresponding to each Pauli operator.
        - condition: The condition under which operators should be grouped ('qwc' or 'fc').
        
        Returns:
        - no_groups: A list of groups of Pauli operators that commute with each other.
        - x_ki_group: A list of grouped expectation values corresponding to each Pauli group.
        """
        no_groups = []
        x_ki_group = []

        for i, pauli in enumerate(pauli_list):
            added_to_any_group = False
            for _, (group, group_2) in enumerate(zip(no_groups, x_ki_group)):
                if self.commutes(group, pauli, condition):
                    group.append(pauli)
                    group_2.append(x_ki_list[i])
                    added_to_any_group = True
                    break
            if not added_to_any_group:
                no_groups.append([pauli])
                x_ki_group.append([x_ki_list[i]])
        return no_groups, x_ki_group
    

    def projectors_group(self, groups, phi):
        """
        Computes the probabilities of projection operators within each group of commuting Pauli operators.
        
        Parameters:
        - groups: A list of groups of commuting Pauli operators.
        - phi: The quantum state vector.
        
        Returns:
        - proj_prob: A list of dictionaries containing the probabilities for each projector within a group.
        """
        proj_prob = []
        pauli_mapping = {'X': 0, 'Y': 1, 'Z': 2, 'I':3}
        for _, group in enumerate(groups): 
            dict_probs = {}
            for string in group: 
                idx = [tuple([pauli_mapping[char] for char in string])]
                pauli = expectvals.string2paulibasis(idx)[:, :, :, 0]
                probs = expectvals.probabilities(phi, pauli)
                dict_probs[string] = [(1-self.p) * x + self.p / self.d for x in probs]
            proj_prob.append(dict_probs)
        return proj_prob
        

    def combinations_proj(self, input_string):
        """
        Generates all possible projectors for a given Pauli string.

        Parameters:
        - input_string: A string of Pauli operators (X, Y, Z, I).

        Returns:
        - combinations: A list of combinations of projectors derived from the input Pauli string.
        """
        proj_mapping = {'X': [1, 2], 'Y': [3, 4], 'Z': [5, 6], 'I': [7]}
        lists_of_numbers = [proj_mapping[char] for char in input_string]
        combinations = list(product(*lists_of_numbers))
        combinations = [list(combination) for combination in combinations]
        return combinations


    def kronecker_product(self, matrices):
        """
        Computes the Kronecker product of a list of matrices.

        Parameters:
        - matrices: A list of matrices.

        Returns:
        - result: The Kronecker product of all the matrices.
        """
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result
    
    
    def combinations_probs(self, list_proj, state):
        """
        Computes the probabilities of projection operators acting on a given quantum state.

        Parameters:
        - list_proj: A list of projectors.
        - state: The quantum state.

        Returns:
        - list_probs: A list of probabilities associated with each projector.
        """
        plus = 0.5 * np.array([[1, 1], [1, 1]])
        minus = 0.5 * np.array([[1, -1], [-1, 1]])
        left = 0.5 * np.array([[1, -1j], [1j, 1]])
        right = 0.5 * np.array([[1, 1j], [-1j, 1]])
        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])        
        id = np.array([[1, 0], [0, 1]])
        proj_mapping2 = {1: plus, 2: minus, 3: left, 4: right, 5: zero, 6: one, 7: id}
        list_probs = []
        for proj in list_proj:
            lists_of_matrices = [proj_mapping2[char] for char in proj]
            list_probs.append(self.measure_pauli_string(state, self.kronecker_product(lists_of_matrices)))
        return list_probs
    
    
    def projectors_group_2(self, groups, state):
        """
        Groups projectors and calculates their probabilities for each quantum state.

        Parameters:
        - groups: A list of groups of Pauli strings.
        - state: The quantum state.

        Returns:
        - proj_group: A list of dictionaries containing projectors for each group.
        - proj_prob: A list of dictionaries containing probabilities for each projector within a group.
        """
        proj_group = []
        proj_prob = []
        for i, group in enumerate(groups): 
            dict_strings = {}
            dict_probs = {}
            for string in group: 
                list_proj = self.combinations_proj(string)
                dict_strings[string] = list_proj
                dict_probs[string] = self.combinations_probs(list_proj, state)
            proj_group.append(dict_strings)
            proj_prob.append(dict_probs)
        return proj_group, proj_prob


    def calculate_matrix_product(self, string):
        """
        Computes the Kronecker product of Pauli matrices specified by the given string.

        Parameters:
        - string: A string of Pauli operators (e.g., 'X', 'Y', 'Z').

        Returns:
        - product: The resulting matrix from the Kronecker product of the Pauli matrices.
        """
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
    
        matrix_mapping = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        for s, matrix_str in enumerate(string):
            if s == 0: 
                product = matrix_mapping[matrix_str]
            else:
                product = np.kron(product, matrix_mapping[matrix_str])
        return product
    
    def convert_pauli_matrices(self, groups):
        """
        Converts groups of Pauli strings into corresponding matrices.

        Parameters:
        - groups: A list of groups of Pauli strings.

        Returns:
        - pauli_matrices_group: A list of lists, where each inner list contains matrices corresponding to a group of Pauli strings.
        """
        pauli_matrices_group = []
        for group in groups:
            matrices = []
            for string in group:
                matrices.append(self.calculate_matrix_product(string))
            pauli_matrices_group.append(matrices)
        return pauli_matrices_group
    

    def measure_pauli_string(self, state, operator): 
        """
        Measures the expectation value of a Pauli string in a given quantum state.

        Parameters:
        - state: The quantum state.
        - operator: The Pauli operator.

        Returns:
        - expect_val: The expectation value of the operator in the state.
        """
        expect_val = np.real(np.trace(state @ operator))
        return expect_val
    

    def measure_pauli_string_exp(self, mi, li, list_x_ki_group, proj_probs):        
        """
        Computes the expected value of a Pauli string using a sampling method in a quantum state.

        Parameters:
        - mi: The number of measurements.
        - li: The probability associated with the measurement.
        - list_x_ki_group: Grouped X_ki values for Pauli string sampling.
        - proj_probs: The probabilities associated with the projectors.

        Returns:
        - Aij: The computed expected value of the Pauli string.
        """ 
        lista = list(range(2 ** self.qubits))
        binary_list = [format(x, '0{}b'.format(self.qubits)) for x in lista]
        k = 0
        Aij = 0
        for key, values in proj_probs.items():
            result = {}
            for idx, binary in enumerate(binary_list):
                grouping_key = ''.join(binary[i] for i in range(len(binary)) if key[i] in ('X', 'Y', 'Z'))
                
                if grouping_key not in result:
                    result[grouping_key] = 0
                # Sum the values for the current binary
                result[grouping_key] += values[idx]
            keys_list = list(result.keys())
            values_list = list(result.values())
            if len(values_list) == 1: 
                values_list = [1.]
            sampling_proj = np.random.multinomial(n = mi * li, pvals = values_list)
            for p, idx in enumerate(sampling_proj): 
                proj = keys_list[p]
                signs = [1 if bit == '0' else -1 for bit in proj]
                product = 1
                for sign in signs:
                    product *= sign
                Aij += list_x_ki_group[k] * product * idx
            k += 1
        return Aij


    def compute_true_fidelity(self, rho, sigma):
        """
        Computes the true fidelity between two density matrices.

        Parameters:
        - rho: The first density matrix.
        - sigma: The second density matrix.

        Returns:
        - F: The fidelity between the two matrices.
        """
        F = np.real(np.trace(rho @ sigma))
        return F


    def truncation(self, list_x_ki):
        """
        Truncates small values in a list and computes their squared probabilities.

        Parameters:
        - list_x_ki: A list of values to truncate.

        Returns:
        - new_list_x_ki: The truncated list of values.
        - new_pauli_probs: The squared probabilities associated with the truncated values.
        """
        bound = self.beta/self.d
        new_list_x_ki = [0.0 if abs(x) < bound else x for x in list_x_ki]
        new_pauli_probs = [x ** 2 for x in new_list_x_ki]
        return new_list_x_ki, new_pauli_probs
    

    def measure_pauli_wk(self, grouping, method, condition):
        """
        Measures the expectation value of Pauli strings, with support for grouping and using sampling methods 
        to estimate the fidelity between an expertimental state (sigma) and a true state (rho). 
        Implements DFE protocol of https://arxiv.org/abs/1104.4695

        Parameters:
        - grouping: If True, applies grouping of Pauli operators before sampling.
        - method: The method for grouping ('si' for sorted insertion).
        - condition: The condition for the grouping method.

        Returns:
        - Y: The estimated fidelity.
        - F: The true fidelity between the original and the noisy state.
        - sum(m): The total number of measurements used.
        - len(groups): The number of groups if grouping is enabled.
        """
        Xi = 0
        # 1. preparation 
        rho, sigma, psi = self.noisy_state(self.state)
        pauli_list_0 = self.generate_combinations(self.qubits)
        list_x_ki = self.probs_and_xki(psi, pauli_list_0)
        pauli_probs = [x ** 2 for x in list_x_ki]
        if not grouping:
            print("Probabilities add to {}".format(sum(pauli_probs)))

        # 2. grouping of pauli operators
        if grouping: 
            print("Grouping Pauli strings...")

            # 2.1. grouping via SI algorithm
            sorted_indices = np.argsort(-np.abs(list_x_ki)) 
            list_x_ki = [list_x_ki[i] for i in sorted_indices]
            pauli_list = [pauli_list_0[i] for i in sorted_indices]    
            if method == "si":
                groups, list_x_ki_group = self.sorted_insertion_grouping(pauli_list, list_x_ki, condition)
                
           # 2.2. calculate probability distribution of projectors for each group
            proj_probs = self.projectors_group(groups, psi)
            list_x_ki = [sum(x) for x in list_x_ki_group]
            pauli_probs = []

            # 2.3 compute new probabilities after grouping
            for group in list_x_ki_group: 
                pauli_probs.append(sum([x ** 2 for x in group]))
            print("Probabilities add to {}".format(sum(pauli_probs)))
            print("Grouping finished. ") 

        else:
            pauli_list = [[x] for x in pauli_list_0]
            proj_probs = self.projectors_group(pauli_list, psi)
            ### To check projectors probabilities 
            # _, proj_probs2 = self.projectors_group_2(pauli_list, sigma)

        # start sampling of pauli strings
        m = []
        print("Running DFE with l = ", self.l)
        strings_sampling = np.random.multinomial(n = self.l, pvals = pauli_probs)

        for s in range(len(strings_sampling)):
            Aij = 0
            x_ki = list_x_ki[s]
            li = strings_sampling[s]
            if li == 0.0: 
                pass 
            elif li != 0.0:
                if grouping:
                    sum_ki_abs = sum([abs(x) for x in list_x_ki_group[s]]) ** 2
                    sum_ki_sq = sum([x ** 2 for x in list_x_ki_group[s]]) ** 2
                    mi = int(np.ceil((2/(self.d * self.l * self.eps ** 2)) * np.log(2/self.delta) * (sum_ki_abs/ sum_ki_sq))) 
                    Aij =  self.measure_pauli_string_exp(mi, li, list_x_ki_group[s], proj_probs[s])
                    x_ki_group = sum([x ** 2 for x in list_x_ki_group[s]])
                    Xi += 1/(mi * np.sqrt(self.d) * x_ki_group) * Aij
                else: 
                    mi = int(np.ceil((2/(self.d * x_ki ** 2 * self.l * self.eps ** 2)) * np.log(2/self.delta)))
                    Aij =  self.measure_pauli_string_exp(mi, li, [1], proj_probs[s])
                    Xi += 1/(mi * np.sqrt(self.d) * x_ki) * Aij
                m.append(mi * li)

        Y = (1 / self.l) * Xi
        F = self.compute_true_fidelity(np.array(rho), sigma)

        if grouping:
            return Y, F, sum(m), len(groups)
        else: 
            return Y, F, sum(m)
        
