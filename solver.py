import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# utility function
def getMatrix(a):
	return np.array([[a, 0], [1-a, 1]])

# simulate M process forward
class Sim:

	def __init__(self, N):
		self.N = N

	def propagateData(self, M, S):
		return self.v(S.shape[1]-1, M, S)

	def v(self, t, M, S):
		# recursively defined
		if t == 0:
			return S[:,[0]]
		else:
			return S[:,[t]] + np.dot(M, self.v(t-1, M, S))


# solve for the M matrix in the 2-class case
class TwoClassSolver:

	def __init__(self, S, v_f):
		self.N = S.shape[1]
		self.S = S
		self.v_f = v_f

	def solve(self):
		# get polynomial coefficients
		self.p = []
		for i in range(self.N):
			self.p.append(self.S[0][i])
		self.p[self.N-1] -= float(self.v_f[0])

		# get roots using numpy.roots
		self.a = -1
		sols = np.roots(self.p)
		for sol in sols:
			if sol.imag == 0 and sol.real >= 0:
				self.a = sol.real
		if self.a == -1:
			print('Error, no solution found.')
			print('S =\n', self.S)
			print('v_f =\n', self.v_f)
			print('p =', self.p)
			print()
			return -1
		else:
			return self.a


class TwoClassSolveTester:

	def __init__(self):
		pass

	# basic test with a single, randomly generated matrix and source dataset
	def example(self, N):
		a = np.random.rand()
		S = 100*np.random.rand(2, N)

		M = getMatrix(a)
		sim = Sim(N)
		v_f = sim.propagateData(M, S)

		tcSolver = TwoClassSolver(S, v_f)
		a_s = tcSolver.solve()
		M_s = getMatrix(a_s)
		v_f_s = sim.propagateData(M_s, S)

		print('Actual a =', a)
		print('Solved a =', a_s)
		print('Error =', abs((a_s-a)/a))
		print()
		print('Actual v_f =\n', v_f)
		print('Predicted v_f =\n', v_f_s)
		print('Error =', np.sum(np.square((v_f_s - v_f)/v_f))/v_f.shape[0])
		print()

	# many-trial test of randomly generated matrix and source datasets, returns RMSE error for matrix and final distribution
	def idealRMSE(self, N, trials):
		drmse = 0
		vrmse = 0

		for i in range(trials):
			a = np.random.rand()
			S = 100*np.random.rand(2, N)

			M = getMatrix(a)
			sim = Sim(N)
			v_f = sim.propagateData(M, S)

			tcSolver = TwoClassSolver(S, v_f)
			a_s = tcSolver.solve()
			M_s = getMatrix(a_s)
			v_f_s = sim.propagateData(M_s, S)

			drmse += np.square((a_s - a)/a)
			vrmse += np.sum(np.square((v_f_s - v_f)/v_f))
		print('Decay Parameter Error =', np.sqrt(drmse))
		print('Final Distribution Error =', np.sqrt(vrmse))
		print()
		return [np.sqrt(drmse), np.sqrt(vrmse)]

	# same as above but with a perturbation to the final distribution
	def perturbedRMSE(self, mag, N, trials):
		drmse = 0
		vrmse = 0

		for i in range(trials):
			a = np.random.rand()*0.99+0.01
			S = 100*np.random.rand(2, N)

			M = getMatrix(a)
			sim = Sim(N)
			v_f = sim.propagateData(M, S)

			S += mag * np.random.randn(S.shape[0], S.shape[1])
			tcSolver = TwoClassSolver(S, v_f)
			a_s = tcSolver.solve()
			if a_s != -1:
				M_s = getMatrix(a_s)
				v_f_s = sim.propagateData(M_s, S)

				drmse += np.square((a_s - a)/a)
				vrmse += np.sum(np.square((v_f_s - v_f)/v_f))
			else:
				print('a =', a)
				print()
		print('Decay Parameter Error =', np.sqrt(drmse))
		print('Final Distribution Error =', np.sqrt(vrmse))
		print()
		return [np.sqrt(drmse), np.sqrt(vrmse)]

tester = TwoClassSolveTester()
#tester.perturbedRMSE(0.01, 10, 1000)

n = np.arange(10, 101)
drmse = []
vrmse = []
for i in n:
	print(i)
	err = tester.perturbedRMSE(0.01, i, 1000)
	drmse.append(err[0])
	vrmse.append(err[1])
plt.plot(n, drmse/drmse[0], label='Decay Parameter Error')
plt.plot(n, vrmse/vrmse[0], label='Final Distribution Error')
plt.legend()
plt.show()






class OldTwoClass:

	def __init__(self, S, v_f):
		self.N = S.shape[1]
		self.S = S
		self.v_f = v_f

	def solve_matrix(self):
		res = opt.minimize_scalar(self.error)
		#if not res.success: print('WARNING: OPTIMIZATION UNSUCCESSFUL\n')
		#return (self.create_matrix(res.x), self.v(self.N-1, self.create_matrix(res.x)), self.error(res.x)/np.linalg.norm(res.x))
		return res.x

	def error(self, a):
		M = self.create_matrix(a)
		return np.linalg.norm(self.v_f - self.v(self.N-1, M))/np.sqrt(self.N)

	def v(self, t, M):
		if t == 0:
			return self.S[:,[0]]
		else:
			return self.S[:,[t]] + np.dot(M, self.v(t-1, M))

	def create_matrix(self, a):
		return np.array([[a, 0],[1-a,1]])











class ReductionManager:

	def __init__(self, S, v_f):
		self.N = S.shape[1]
		self.L = S.shape[0]
		self.S = S
		self.v_f = v_f
		self.a = np.full((self.L), 0.5)

	def solve_matrix(self, S, v_f):
		if S.shape[0] == 2:
			twoClass = TwoClass(S, v_f)
			self.a[self.L-2] = twoClass.solve_matrix()
			self.a[self.L-1] = 1
		else:
			S_small = np.vstack((S[[0],:], [np.sum(S[1:,:], axis=0)]))
			v_f_small = np.vstack((v_f[[0],:], [np.sum(v_f[1:,:], axis=0)]))
			twoClass = TwoClass(S_small, v_f_small)
			a = twoClass.solve_matrix()
			self.a[self.L-S.shape[0]] = a
			S_prime = self.add_adjustments(S, a)
			self.solve_matrix(S_prime, v_f[1:,:])

	def add_adjustments(self, S, a):
		S_prime = S[1:,:]
		v_1 = S[0,0]
		for n in range(1, self.N):
			S_prime[0,n] += (1-a)*v_1
			v_1 = a*v_1 + S[0,n]
		return S_prime

	def get_matrix(self):
		self.solve_matrix(self.S, self.v_f)
		M = np.zeros((self.L, self.L))
		for i in range(self.L):
			M[i, i] = self.a[i]
			if i<self.L-1:
				M[i+1,i] = 1 - self.a[i]
		return M


class Simulation:

	def __init__(self, S, M):
		self.N = S.shape[1]
		self.L = S.shape[0]
		self.S = S
		self.M = M

	def v(self, t):
		if t == 0:
			return self.S[:,[0]]
		else:
			return self.S[:,[t]] + np.dot(self.M, self.v(t-1))

	def simulate_forward(self):
		t = np.arange(0, self.N, 1)
		self.V = np.zeros((self.L, self.N))
		for n in range(self.N):
			self.V[:, [n]] = self.v(t[n])

		return self.V[:, [self.N-1]]


# L_real=5
# N_real=100
# a_real = np.random.rand(L_real-1)
# S_real = np.zeros((L_real,N_real))
# for i in range(L_real):
# 	for n in range(N_real):
# 		S_real[i,n] = 100*np.random.rand()
# M_real = np.zeros((L_real,L_real))
# for i in range(L_real-1):
# 	M_real[i,i] = a_real[i]
# 	M_real[i+1,i] = 1-a_real[i]
# M_real[L_real-1,L_real-1]=1

# simulation = Simulation(S_real, M_real)
# v_f_real = simulation.simulate_forward()

# print('M_real = \n', M_real)
# print('v_f_real = \n', v_f_real)

# print('\n')

# reductionManager = ReductionManager(S_real, v_f_real)
# M_calculated = reductionManager.get_matrix()

# solve_simulation = Simulation(S_real, M_calculated)
# v_f_calculated = solve_simulation.simulate_forward()


# print('M_calculated = \n', M_calculated)
# print('v_f_calculated = \n', v_f_calculated)


# v_f_error = np.linalg.norm((v_f_real - v_f_calculated))/np.sqrt(N_real)
# M_error = np.linalg.norm((M_real - M_calculated))/np.sqrt(2*L_real-1)
# print('\n')
# print('RMS v_f error =', v_f_error)
# print('RMS M error =', M_error)



class GeneralSolver:

	def __init__(self, S, v_f):
		L = 2
		N = 5
		self.S = S
		self.v_f = v_f

	def solve_matrix(self):
		a0 = np.full((L, 1), 0.5)
		res = opt.minimize(self.error, a0)
		#if not res.success: print('WARNING: OPTIMIZATION UNSUCCESSFUL\n')
		return (self.create_matrix(res.x), self.v(N-1, M), self.error(res.x)/np.linalg.norm(res.x))

	def error(self, a):
		# 'a' should be an L-vector describing the diagonal elements of M
		# 'v_f' should be an L-vector describing the final class distribution
		# 'S' should be an LxN matrix describing the entire history of sourcing
		# function returns an RMS error of the predicted v_f based on this M from the actual v_f
		M = self.create_matrix(a)
		return np.linalg.norm(self.v_f - self.v(N-1, M))/np.sqrt(N)

	def v(self, t, M):
		if t == 0:
			return self.S[:,[0]]
		else:
			return self.S[:,[t]] + np.dot(M, self.v(t-1, M))

	def create_matrix(self, a):
		M = np.zeros((L, L))
		for i in range(L):
			M[i, i] = a[i]
			if i<L-1:
				M[i+1,i] = 1 - a[i]
		return M
