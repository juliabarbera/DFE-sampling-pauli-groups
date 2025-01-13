
import numpy as np
from functools import reduce


def string2pauli(idx):
    """
    Given a binary string, gives the Pauli basis.
    PARAMETERS:
        idx: list(tuples). It must contain elements in [0, 1, 2, 3]
             where 0 corresponds to the identity matrix I.
    RETURNS:
        bases: (2, 2, n_qubits, n_tuples) complex array. It contains
               all the eigenvectors of all Pauli strings of n qubits including I.
    """
    # Define the one-qubit bases
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex) 
    I = np.array([[1, 0], [0, 1]], dtype=complex)

    # Include the identity matrix in the stack
    sigma = np.stack([X, Y, Z, I], axis=-1)
    
    n_qubits = len(idx[0])
    n_tuples = len(idx)
    bases = np.zeros((2, 2, n_qubits, n_tuples), dtype="complex")

    # Calculate all the combinations of tensor products
    for i in range(n_tuples):
        for j in range(n_qubits):
            bases[:, :, j, i] = sigma[:, :, idx[i][j]]

    return bases


def string2paulibasis(idx):
    """
    Given a binary string, gives the Pauli basis.
    PARAMETERS:
        idx: list(tuples). It must contain elements in [0, 1, 2, 3]
             where 0 corresponds to the identity matrix I.
    RETURNS:
        bases: (2, 2, n_qubits, n_tuples) complex array. It contains
               all the eigenvectors of all Pauli strings of n qubits including I.
    """
    # Define the one-qubit bases
    X = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2) + 1j*0.0
    Y = np.array([[1., -1.*1j], [1, 1.*1j]]) / np.sqrt(2) + 1j*0.0
    Z = np.array([[1., 0.], [0., 1.]]) + 1j*0.0
    I = np.array([[1., 0.], [0., 1.]]) + 1j*0.0

    # Include the identity matrix in the stack
    sigma = np.stack([X, Y, Z, I], axis=-1)
    
    n_qubits = len(idx[0])
    n_tuples = len(idx)
    bases = np.zeros((2, 2, n_qubits, n_tuples), dtype="complex")

    # Calculate all the combinations of tensor products
    for i in range(n_tuples):
        for j in range(n_qubits):
            bases[:, :, j, i] = sigma[:, :, idx[i][j]]

    return bases



def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.
    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape
    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def refold(vec, mode, dims):
    """
    Refolds vector into tensor.
    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape
    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)

# ==== KRON-VEC PRODUCT COMPUTATIONS ==== #

def kron_vec_prod(As, v):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    n_qubits = As.shape[-1]
    dims = [As[:, :, j].shape[0] for j in range(n_qubits)]
    vt = v.reshape(dims)
    for i in range(n_qubits):
        vt = refold(As[:, :, i] @ unfold(vt, i, dims), i, dims)
    return vt.ravel()


def kron_brute_force(As, v):
    """
    Computes kron-matrix times vector by brute
    force (instantiates the full kron product).
    """
    return reduce(np.kron, As) @ v


def probabilities(psi, base):
    di = psi.shape[0]
    n_qubits = int(np.log2(di))
    dot_product = kron_vec_prod(base, psi)
    probs = np.abs(dot_product)**2
    return probs
    
