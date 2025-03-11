import numpy as np
from tqdm import tqdm

class L96_rl:
    def __init__(self, N, F):
        self.N = N
        self.F = F

    def dx(self, x0, t):
        # dx_i/dt = (x_{i + 1} - x_{i - 2}) * x_{i - 1} - x_i + F
        dxs = np.zeros(self.N)
        dxs[0] = (x0[1] - x0[self.N - 2]) * x0[self.N - 1] - x0[0] + self.F
        dxs[1] = (x0[2] - x0[self.N - 1]) * x0[0] - x0[1] + self.F
        dxs[self.N - 1] = (x0[0] - dxs[self.N - 3]) * x0[self.N - 2] - x0[self.N - 1] + self.F
        for i in range(2, self.N - 1):
            dxs[i] = (x0[i+1] - x0[i - 2]) * x0[i - 1] - x0[i] + self.F

        return dxs

class L96:
    def __init__(self, N, F, dt):
        self.model = L96_rl(N, F)
        self.dt = dt

    def forward_ens(self, ens, u0_ens, Nt):
        enses = u0_ens[:]
        for _ in range(Nt):
            enses = [runge_kutta_4(self.model.dx, ens_element, 0, self.dt) for ens_element in enses]
        return enses

def runge_kutta_4(func, x0, t, dt):
    '''Apply runge-kutta to a function'''
    k1 = func(x0, t)
    k2 = func(x0 + (dt / 2.0) * k1, t + (dt / 2.0))
    k3 = func(x0 + (dt / 2.0) * k2, t + (dt / 2.0))
    k4 = func(x0 + dt * k3, t + dt)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def generate_l96(N, F, timesteps, dt, noise=0.1):
    '''Generate l96 data using runge-kutta approx.
    Initial conditions will be initialized to [F, ..., F] + N(0, 0.01)
    '''
    system = L96_rl(N, F)
    u0 = np.ones(N) * F + np.random.multivariate_normal(np.zeros(N), np.eye(N) * 0.01)
    data = [u0]
    observations = [u0 + np.random.multivariate_normal(np.zeros(N), np.eye(N) * noise)]
    t = 0

    for i in tqdm(range(timesteps - 1)):
        u = data[-1]
        u = runge_kutta_4(system.dx, u, t, dt)
        data.append(u)
        observations.append(u + np.random.multivariate_normal(np.zeros(N), np.eye(N) * noise))
        t += dt

    return u0, np.array(data), np.array(observations)
