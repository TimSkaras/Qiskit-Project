from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble, execute
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.library import SaveState
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel
import copy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.opflow import Zero, One, I, X, Y, Z, VectorStateFn

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

N = 2 # number of trotter steps
nq = 5
qbts = [0,1,2]
T = np.pi

qc= QuantumCircuit(nq)
qc.x(qbts[-2])
qc.x(qbts[-1])


for j in range(N):
	qc = trotter_step(qc, T/N, qbts)

# qc.save_statevector()

# simulate
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_santiago')
noise_model = NoiseModel.from_backend(backend)

coupling_map = backend.configuration().coupling_map
basis_gates = noise_model.basis_gates
basis_gates.extend(['rxx','ryy','rzz'])
print(basis_gates)
backend = QasmSimulator(method='density_matrix', noise_model=noise_model)

result = execute(qc, backend,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        noise_model=noise_model).result()
SaveState(result)
print(result)

# wf_trot = result.get_statevector()
# wf_trot = VectorStateFn(wf_trot).to_matrix()

# init_state = One^Zero^One^Zero^Zero
# wf_true = (U_heis(T, qbts, nq) @ init_state).eval()
# wf_true = wf_true.to_matrix()

# print("Trotter results:")
# print(wf_trot)
# print()

# print("Real Answer")
# print(wf_true)
# print()


# print("Quantum Fidelity")
# print(np.abs( np.dot(np.conjugate(wf_true), wf_trot) )**2)

# x_vals = np.arange(2**nq)
# plt.bar(x_vals, np.abs(wf_trot)**2, width=0.25)
# plt.bar(x_vals+0.25, np.abs(wf_true)**2, width=0.25)
# plt.grid()
# plt.show()
