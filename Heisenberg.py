from qiskit import QuantumCircuit, transpile, Aer, IBMQ, assemble, execute
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer.noise import NoiseModel
import copy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.opflow import Zero, One, I, X, Y, Z, VectorStateFn


# Returns the matrix representation of the XXX Heisenberg model for 3 spin-1/2 particles in a line
def H_heis3():
    # Interactions (I is the identity matrix; X, Y, and Z are Pauli matricies; ^ is a tensor product)
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    
    # Sum interactions
    H = XXs + YYs + ZZs
    
    # Return Hamiltonian
    return H

# Returns the matrix representation of U_heis3(t) for a given time t assuming an XXX Heisenberg Hamiltonian for 3 spins-1/2 particles in a line
def U_heis3(t):
    # Compute XXX Hamiltonian for 3 spins in a line
    H = H_heis3()
    
    # Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
    return (t * H).exp_i()

def trotter_step(qc, t_n, nq):
	"""
	Gate for trotterized step on whole circuit with time slice t/n

	INPUTS:
	t_n -- float giving t/n for this trotter step

	OUTPUT:
	trotter gate

	"""

	# qc = QuantumCircuit(nq)

	qc.rxx(2*t_n, 0, 1)
	qc.ryy(2*t_n, 0, 1)
	qc.rzz(2*t_n, 0, 1)
	qc.rxx(2*t_n, 1, 2)
	qc.ryy(2*t_n, 1, 2)
	qc.rzz(2*t_n, 1, 2)

	return qc #.to_instruction()

N = 20 # number of trotter steps
nq = 3
T = np.pi

qc= QuantumCircuit(nq)
qc.x(1)
qc.x(2)

for j in range(N):
	qc = trotter_step(qc, T/N, nq)

qc.save_statevector()
# qc.draw(output='mpl')
# plt.show()

svsim = Aer.get_backend('aer_simulator')
result = svsim.run(assemble(qc)).result()
wf_trot = result.get_statevector()
wf_trot = VectorStateFn(wf_trot).to_matrix()

init_state = One^One^Zero
wf_true = (U_heis3(T) @ init_state).eval()
wf_true = wf_true.to_matrix()

# print("Trotter results:")
# print(wf_trot)
# print()

# print("Real Answer")
# print(wf_true)
# print()

print("Quantum Fidelity")
print(np.abs( np.dot(np.conjugate(wf_true), wf_trot) )**2)

x_vals = np.arange(2**nq)
plt.bar(x_vals, np.abs(wf_trot)**2, width=0.25)
plt.bar(x_vals+0.25, np.abs(wf_true)**2, width=0.25)
plt.grid()
plt.show()
