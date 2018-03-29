
import math
import numpy as np
from scipy import special
import random
import copy as cp


class BrownBoost:
    def __init__(self, base_estimator, c=10, convergence_criterion=0.0001, max_iter=10000):
        """ Initiates BrownBoost classifier
        
        Parameters
        ----------
        base_estimator: classifier from scikit-learn
            The base leaner in ensemble
        c: int or float
            A positive real value
            default = 10
        convergence_criterion: float
            A small constant(>0) used to avoid degenerate cases.
            default = 0.0001
        """
        self.base_estimator = base_estimator
        self.c = c
        self.max_iter = max_iter
        self.max_iter_newton_raphson = max_iter / 100
        self.convergence_criterion = convergence_criterion
        self.alphas = []
        self.hs = []
        self.ss = []

    def fit(self, X, y):
        """ Trains the classifier
        Parameters
        ----------
        X: ndarray
            The training instances
        y: ndarray
            The target values for The training instances
            
        returns
        --------
            self
        """

        # Initiate parameters
        self.__init__(base_estimator=self.base_estimator,
                      c=self.c,
                      max_iter=self.max_iter,
                      convergence_criterion=self.convergence_criterion)

        s = self.c
        r = np.zeros(X.shape[0])
        k = 0
        while s >= 0 and k < self.max_iter :
#             print(f'iter is {k}\ts = {s}')
            self.ss.append(s)
            k += 1
            w = np.exp(-(r + s)**2 / self.c)

            h = cp.deepcopy(self.base_estimator)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            
            error = np.multiply(pred, y)
            gamma = np.dot(w, error)

            alpha, t = self.newton_raphson(r, error, s, gamma)
#             theta = (0.1/self.c)**2
#             A = 32 * math.sqrt(self.c*math.log(2/theta))
#             if t < gamma**2/A:
#                 (new_t * w).sum()
#                 t = new_t + gamma**2/A

            r += alpha * error
            s -= t

            self.alphas.append(alpha)
            self.hs.append(h)

    def predict(self, X):
        """ Classify the samples
        Parameters
        ----------
        X: ndarray
            The test instances
            
        Returns
        -------
        y: ndarray
            The pred with BrownBoost for the test instances
        """

        y = np.zeros(X.shape[0])
        for i in range(0, len(self.hs)):
            y += self.alphas[i] * self.hs[i].predict(X)
        return np.sign(y)

    def newton_raphson(self, r, error, s, gamma):
        """ Computes alpha and t
        Parameters
        ----------
        r: array
            margins for the instances
        error: ndarray
            error vec between pred and true instances
        s: float
            'time remaining'
        gamma: float
            correlation
        y: ndarray
            the target values
            
        Retruns
        -------
        alpha: float
        t: float
        """

        # Theorem 3 & 5
        alpha = min([0.1, gamma])
        t = (alpha**2) / 3

        a = r + s
        change_amount = self.convergence_criterion + 1
        k = 0

        while change_amount > self.convergence_criterion and k < self.max_iter_newton_raphson:
            d = a + alpha * error - t
            w = np.exp(-d**2 / self.c)

            # Coefficients for jacobian
            W = w.sum()
            U = (w * d * error).sum()
            B = (w * error).sum()
#             if abs(B) < 0.001:
#                 break
            V = (w * d * error**2).sum()
            E = (special.erf(d / math.sqrt(self.c)) - special.erf(a / math.sqrt(self.c))).sum()

            sqrt_pi_c = math.sqrt(math.pi * self.c)
            denominator = 2*(V*W - U*B)
            alpha_step = (self.c*W*B + sqrt_pi_c*U*E)/denominator
            t_step = (self.c*B*B + sqrt_pi_c*V*E)/denominator

            alpha += alpha_step
            t += t_step
            change_amount = math.sqrt(alpha_step**2 + t_step**2)
#             print(f'\t newton_raphson iter is {k}, {change_amount}')
            k += 1
        
        return alpha, t