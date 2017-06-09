import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class MProcessSolver:

	def __init__(self, S, v_f):
		self.S = S
		self.v_f = v_f
		self.a = []

	def get2Matrix(self, a):
		return np.array([[a, 0], [1-a, 1]])

	def twoSolve(self, S, v_f):
		N = S.shape[1]
		p = []
		for i in range(N):
			p.append(S[0][i])
		p[N-1] -= float(v_f[0])

		a = -1
		sols = np.roots(p)
		for sol in sols:
			if sol.imag == 0 and sol.real >= 0:
				a = sol.real
		return a

	def v(self, t, M, S):
	    if t == 0:
	        return S[:,[0]]
	    else:
	        return S[:,[t]] + np.dot(M, self.v(t-1, M, S))

	def getGenSol(self):
		self.genSolve(self.S, self.v_f)
		self.a.append(1)
		return self.a

	def genSolve(self, S, v_f):
		if S.shape[0] > 2:
			S_small = np.vstack((S[[0],:], [np.sum(S[1:,:], axis=0)]))
			v_f_small = np.vstack((v_f[[0],:], [np.sum(v_f[1:,:], axis=0)]))
			self.a.append(self.twoSolve(S=S_small, v_f=v_f_small))
			S_prime = S[1:,:]
			v_1 = S[0,0]
			for n in range(1, S_prime.shape[1]):
				S_prime[0][n] += (1-self.a[-1])*v_1
				v_1 = self.a[-1]*v_1 + S[0,n]
			self.genSolve(S_prime, v_f[1:,:])
		else:
			self.a.append(self.twoSolve(S, v_f))