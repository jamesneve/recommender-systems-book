import numpy as np


class LatentFactorSGD(object):

    def __init__(self, r, k, alpha, iterations, lmda):
        self.alpha = alpha
        self.R = r
        self.U = np.zeros((len(r), k,))
        self.V = np.zeros((len(r[0]), k,))
        self.iterations = iterations
        self.lmda = lmda

    def uniform_initialize_uv(self):
        self.U = np.random.normal(0, 1.0, np.shape(self.U))
        self.V = np.random.normal(0, 1.0, np.shape(self.V))

    def train(self):
        s = self.generate_s()

        for i in range(0, self.iterations):
            np.random.shuffle(s)
            self.gradient_descent(s)
            mse = self.mse(s)
            print("MSE: %f" % mse)

        return self.U, self.V

    def mse(self, s):
        PR = np.dot(self.U, self.V.T)
        mse = 0.0
        for i, j, r in s:
            mse += np.power(r - PR[i, j], 2)
        return np.sqrt(mse)

    def generate_s(self):
        s = []
        for i, row in enumerate(self.R):
            for j, val in enumerate(row):
                if np.nonzero(val) and not np.isnan(val):
                    s.append((i, j, val))

        return s

    def r_minus_uvt(self):
        vt = np.transpose(self.V)
        uvt = np.matmul(self.U, vt)

        r_uvt = self.R - uvt
        return r_uvt

    def gradient_descent(self, s):
        for cnt, (i, j, r) in enumerate(s):
            prediction = np.dot(self.U[i, :], self.V[j, :].T)
            e = r - prediction

            reg_u = self.lmda * self.U[i, :]  # Regularization term for U
            self.U[i, :] += self.alpha * (e * self.V[j, :] - reg_u) # Update single row of U
            reg_v = self.lmda * self.V[j, :]  # Regularization term for V
            self.V[j, :] += self.alpha * (e * self.U[i, :] - reg_v) # Update single row of V
