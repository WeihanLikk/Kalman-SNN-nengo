import numpy as np
import nengo
from nengo.dists import Uniform
from kalman import Kalman
from nengo.processes import Piecewise


class Kalman_SNN:
    def __init__(self):
        self.A_k = None
        self.B_k = None

        self.dt = 0.02  # simulation time step
        self.t_rc = 0.01  # membrane RC time constant
        self.t_ref = 0.001  # refractory period
        self.tau = 0.009  # synapse time constant for standard first-order lowpass filter synapse
        self.N_A = 1000  # number of neurons in first population
        self.rate_A = 200, 400  # range of maximum firing rates for population A

        self.model = nengo.Network(label="Kalman SNN")
        self.sim = None
        self.kalman = Kalman()

        self.output = None
        self.testX = None
        self.count = 0
        self.chN = None

    def train(self, NeuralData, NeuralDim, KinData, KinDim):
        NeuralData = np.reshape(np.asarray(NeuralData), (NeuralDim, -1))
        KinData = np.reshape(np.asarray(KinData), (KinDim, -1))
        self.chN = np.where(np.sum(NeuralData, axis=1) != 0)
        NeuralData = np.squeeze(NeuralData[self.chN, :])

        self.A_k, self.B_k = self.kalman.calculate(NeuralData, KinData, pool=0, dt=self.dt, tau=self.tau)

    def build(self, testY):
        self.count = 0

        def update(x):
            Externalmat = np.mat(x[2:4]).T
            Inputmat = np.mat(x[0:2]).T

            next_state = np.squeeze(np.asarray(self.A_k * Inputmat + Externalmat))
            return next_state

        with self.model:
            LIF_Neurons = nengo.Ensemble(
                self.N_A,
                dimensions=2 + 2,
                intercepts=Uniform(-1, 1),
                max_rates=Uniform(self.rate_A[0], self.rate_A[1]),
                neuron_type=nengo.LIFRate(tau_rc=self.t_rc, tau_ref=self.t_ref)
            )

            state_func = Piecewise({
                0.0: [0.0, 0.0],
                self.dt: np.squeeze(np.asarray(np.mat([testY[0], testY[1]]).T)),
                2 * self.dt: [0.0, 0.0]
            })

            state = nengo.Node(output=state_func)
            # state_probe = nengo.Probe(state)

            external_input = nengo.Node(output=lambda t: self.data(t))
            # external_input_probe = nengo.Probe(external_input)

            conn0 = nengo.Connection(state, LIF_Neurons[0:2])

            conn1 = nengo.Connection(external_input, LIF_Neurons[2:4])

            conn3 = nengo.Connection(LIF_Neurons, LIF_Neurons[0:2],
                                     function=update,
                                     synapse=self.tau
                                     )

            self.output = nengo.Probe(LIF_Neurons[0:2])
            self.sim = nengo.Simulator(self.model, dt=self.dt)

    def data(self, t):
        if t == 0.0:
            return [0.0, 0.0]
        yt = np.mat(self.testX)
        out = np.transpose(self.B_k * yt.T)
        return np.squeeze(np.asarray(out))

    def test(self, testX):
        self.testX = testX[self.chN]
        self.sim.step()
        res = self.sim.data[self.output][self.count]
        self.count = self.count + 1
        return res
