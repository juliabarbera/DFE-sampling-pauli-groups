import numpy as np 
import qutip as qt
from qutip import rand_ket_haar, w_state, ghz_state
from itertools import permutations, product
import experiments_dfe_v10dec as experiments 
import matplotlib.pyplot as plt
import time 
import pickle

eps = 0.05
delta = 0.05
beta = 0.0
qubits = 8
p = 0.1
d = 2 ** qubits
states = ['haar_state', 'w_state', 'ghz_state']
iters = 1000

for state in states:  
    machine = experiments.DFE(eps, delta, beta, qubits, p, state)
    total_copies_m = []
    differences = []
    time_dfe = []
    init_time = time.process_time()
    for it in range(iters):
        print("--------------  k = {} --------------".format(it))
        Y, F, m = machine.measure_pauli_wk(grouping = False, method = "si", condition = "qwc", overlapping = False)
        print("Noisy fidelity = {}, True fidelity = {}, Total copies = {}".format(Y, F, m))
        differences.append(Y - F)
        total_copies_m.append(m)
    time_dfe.append(time.process_time()- init_time)
    print("time = {} s.".format(time.process_time() - init_time))


    machine = experiments.DFE(eps, delta, beta, qubits, p, state)
    differences_group_no_fc = []
    total_copies_m_group_no_fc = []
    groups_no_fc = []
    time_no_fc = []
    init_time = time.process_time()
    for it in range(iters):
        print("--------------  k = {} --------------".format(it))
        Y, F, m, g = machine.measure_pauli_wk(grouping = True, method = "si", condition = "fc", overlapping = False)
        print("Noisy fidelity = {}, True fidelity = {}, Total copies = {}".format(Y, F, m))
        differences_group_no_fc.append(Y - F)
        total_copies_m_group_no_fc.append(m)
        groups_no_fc.append(g)
    time_no_fc.append(time.process_time() - init_time)
    print("time = {} s.".format(time.process_time() - init_time))


    machine = experiments.DFE(eps, delta, beta, qubits, p, state)
    differences_group_no_qw = []
    total_copies_m_group_no_qw = []
    groups_no_qw = []
    time_no_qw = []
    init_time = time.process_time()
    for it in range(iters):
        print("--------------  k = {} --------------".format(it))
        Y, F, m, g = machine.measure_pauli_wk(grouping = True, method = "si", condition = "qwc", overlapping = False)
        print("Noisy fidelity = {}, True fidelity = {}, Total copies = {}".format(Y, F, m))
        differences_group_no_qw.append(Y - F)
        total_copies_m_group_no_qw.append(m)
        groups_no_qw.append(g)
    time_no_qw.append(time.process_time() - init_time)
    print("time = {} s.".format(time.process_time() - init_time))


    data = {
        "differences": differences,
        "total_copies_m": total_copies_m,
        "time_dfe": time_dfe,
        "differences_group_no_fc": differences_group_no_fc,
        "total_copies_m_group_no_fc": total_copies_m_group_no_fc,
        "time_no_fc": time_no_fc,
        "differences_group_no_qw": differences_group_no_qw,
        "total_copies_m_group_no_qw": total_copies_m_group_no_qw,
        "time_no_qw": time_no_qw
    }

    # Step 2: Save to a pickle 
    name_file = f"dfe_8_qubits_non_overlapping_{state}_1000it.pkl"
    with open(name_file, "wb") as file:
        pickle.dump(data, file)