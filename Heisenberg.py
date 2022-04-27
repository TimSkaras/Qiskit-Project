from qiskit import QuantumCircuit, QuantumRegister, transpile, Aer, IBMQ, assemble, execute
from qiskit.quantum_info import DensityMatrix
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.library import SaveState
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import extensions  # import aer snapshot instructions
import copy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.opflow import Zero, One, I, X, Y, Z, VectorStateFn
from qiskit.test.mock import FakeJakarta

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

def heis_op(pauli_idx, pair, Nq):
	"""
	Create heisenberg operator (e.g., X^X^I^I) term for Heis hamiltonian

	INPUTS:
	pauli_idx - int 0,1,2 for x,y,z
	pair -- indices of non-identity elements (ex: [1, 3] for I^X^I^X^I)
	nq -- int number of qubits
	"""
	paulis = [X,Y,Z]
	p = paulis[pauli_idx]

	ps = p if 0 in pair else I
	for i in range(1,Nq):
		ps = (ps^p) if i in pair else (ps^I)
	
	
	return ps


def H_heis(qbts, Nq):
	"""
	Returns the matrix representation of the XXX Heisenberg model for spin-1/2 particles in a line
	
	INPUTS:
	qbts -- array of qubit indices
		Ex: [1,3,5] to simulate qubits 2,4,6
	nq -- int number of qubits
	
	"""
	pairs = [[qbts[i], qbts[i+1]] for i in range(len(qbts)-1)]


	XXs = sum([heis_op(0, pair, Nq) for pair in pairs])
	YYs = sum([heis_op(1, pair, Nq) for pair in pairs])
	ZZs = sum([heis_op(2, pair, Nq) for pair in pairs])

	# Sum interactions
	H = XXs + YYs + ZZs

	# Return Hamiltonian
	return H

# Returns the matrix representation of U_heis3(t) for a given time t assuming an XXX Heisenberg Hamiltonian for 3 spins-1/2 particles in a line
def U_heis(t, qbts, Nq):
	# Compute XXX Hamiltonian for spins in a line
	H = H_heis(qbts, Nq)

	# Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
	return (t * H).exp_i()

def trotter_step(qc, t_n, qbts):
	"""
	Gate for trotterized step on whole circuit with time slice t/n

	INPUTS:
	qc -- QuantumCircuit object to be modified
	t_n -- float giving t/n for this trotter step
	qbts -- array of qubit indices

	OUTPUT:
	trotter gate

	"""

	qc.rxx(2*t_n, qbts[0], qbts[1])
	qc.ryy(2*t_n, qbts[0], qbts[1])
	qc.rzz(2*t_n, qbts[0], qbts[1])
	qc.rxx(2*t_n, qbts[1], qbts[2])
	qc.ryy(2*t_n, qbts[1], qbts[2])
	qc.rzz(2*t_n, qbts[1], qbts[2])

	return qc #.to_instruction()

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs):
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

nq = 7
qbts = [1,3,5]
T = np.pi

# True result
init_state = Zero^One^Zero^One^Zero^Zero^Zero
wf_true = (U_heis(T, qbts, nq) @ init_state).eval()
wf_true = wf_true.to_matrix()

# simulate
provider = IBMQ.load_account()

# key = "8d5d40203b561a03caba49fffcc0f33968d812029bcfd8bf09a2992df0cb9de632e3eb47ccb09421b6fbad972460972dd1d07efcaf7d024d4307b0601cb21a4f"
# IBMQ.save_account(key, overwrite=True)
# provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
# backend = provider.get_backend("ibmq_jakarta")
# # backend = FakeJakarta()
# noise_model = NoiseModel.from_backend(backend)

# coupling_map = backend.configuration().coupling_map
# basis_gates = noise_model.basis_gates
# basis_gates.extend([ 'save_density_matrix'])

# backend = QasmSimulator(method='density_matrix', noise_model=noise_model)

shots = 8192
reps = 15

fidelities = np.zeros([9,reps], dtype=float)

# for i in range(9):
# 	N = 4 + i # number of trotter steps

# 	qc = QuantumCircuit(nq)
# 	qc.x(qbts[-2])
# 	qc.x(qbts[-1])


# 	for k in range(N):
# 		qc = trotter_step(qc, T/N, qbts)

# 	st_qcs = state_tomography_circuits(qc, [1,3,5])
# 	# qc.save_density_matrix()



# 	runs = []

# 	for k in range(reps):

# 		run = execute(st_qcs, backend,
# 		        coupling_map=coupling_map,
# 		        basis_gates=basis_gates,
# 		        noise_model=noise_model,
# 		        optimization_level=3,
# 		        shots = shots,
# 		        )
# 		runs.append(run)

# 	for k in range(reps):
# 		fidelities[i, k] = (state_tomo(runs[k].result(), st_qcs))

# np.save('fidelities.npy', fidelities)
fidelities = np.load('fidelities.npy')

means = np.mean(fidelities, axis=1)
stds = np.std(fidelities, axis=1)


font = {'size'   : 15}
plt.rc('font', **font)
steps = np.arange(4,4+9)
plt.errorbar(steps, means, stds, capsize=3.5)
plt.xlabel('# Trotter Steps')
plt.title("Tomography Fidelity vs. # Trotter Steps (Measuring Noise Model)")
plt.ylim([0,.5])
plt.grid()
plt.show()


# DM=result.data()['density_matrix']


# font = {'size'   : 15}
# plt.rc('font', **font)
# labels = ["No Noise", "Noisy - Opt 1", "Noisy - Opt 2","Noisy - Opt 3"]
# steps = np.arange(4,15)
# for k in range(4): plt.bar(steps + 0.2*k - 0.3, fidelities[k,:], width=0.2, label=labels[k])
# plt.xlabel('# Trotter Steps')
# plt.title("Quantum Fidelity vs. # Trotter Steps (Jakarta w/ Noise Model")
# plt.grid()
# plt.legend()
# plt.show()
