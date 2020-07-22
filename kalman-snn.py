import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.dists import Uniform
from dataloader import Dataloader
from kalman import Kalman
from nengo.processes import Piecewise

dt = 0.1  # simulation time step
t_rc = 0.04  # membrane RC time constant
t_ref = 0.002  # refractory period
tau = 0.2  # synapse time constant for standard first-order lowpass filter synapse
N_A = 1000  # number of neurons in first population
rate_A = 200, 400  # range of maximum firing rates for population A
pool = 0

dataloader = Dataloader()
kalman = Kalman()

trainX, trainY, testX, testY = dataloader.getData()
ChN = np.where(np.sum(trainX, axis=1) != 0)
trainX = np.squeeze(trainX[ChN, :])
testX = np.squeeze(testX[ChN, :])

A_0, B_0 = kalman.calculate(trainX, trainY, pool=pool, dt=dt, tau=tau)


# A_1, B_1 = kalman.calculate(trainX, trainY, pool=1, dt=dt, tau=tau)

# kalman.Kalman_Filter(testX, testY)

# kalman.standard_Kalman_Filter(testX, testY)

def data(t):
    """
    Neuron records, Y_k, and calculate B * Y_k

    """
    if t == 0.0:
        return [0.0, 0.0]
    yt = np.mat(testX[:, int(10 * t)])
    out = np.transpose(B_0 * yt.T)
    return np.squeeze(np.asarray(out))


lifRate_neuron = nengo.LIFRate(tau_rc=t_rc, tau_ref=t_ref)
model = nengo.Network(label="NEF")


def update(x):
    """
    Kalman Filter: X_k = A * X_k_1 + B * Y_k

    """
    Externalmat = np.mat(x[2:4]).T
    Inputmat = np.mat(x[0:2]).T

    next_state = np.squeeze(np.asarray(A_0 * Inputmat + Externalmat))
    return next_state


with model:
    LIF_Neurons = nengo.Ensemble(
        N_A,
        dimensions=2 + 2,
        intercepts=Uniform(-1, 1),
        max_rates=Uniform(rate_A[0], rate_A[1]),
        neuron_type=lifRate_neuron
    )

    origin = nengo.Node(lambda t: testY[:, int(10 * t)])  # dt = 100ms
    origin_probe = nengo.Probe(origin)

    state_func = Piecewise({
        0.0: [0.0, 0.0],
        dt: np.squeeze(np.asarray(np.mat([testY[0, 0], testY[1, 0]]).T)),
        2 * dt: [0.0, 0.0]
    })

    state = nengo.Node(output=state_func)
    state_probe = nengo.Probe(state)

    external_input = nengo.Node(output=lambda t: data(t))
    external_input_probe = nengo.Probe(external_input)

    conn0 = nengo.Connection(state, LIF_Neurons[0:2])

    conn1 = nengo.Connection(external_input, LIF_Neurons[2:4])

    conn3 = nengo.Connection(LIF_Neurons, LIF_Neurons[0:2],
                             function=update,
                             synapse=tau
                             )

    neurons_out = nengo.Probe(LIF_Neurons)

    with nengo.Simulator(model, dt=dt) as sim:
        # for i in range(2999):
        #     sim.step()
        sim.run(299)

    print(np.corrcoef(sim.data[neurons_out][:, 0], sim.data[origin_probe][:, 0]))
    print(np.square(sim.data[neurons_out][:, 0] - sim.data[origin_probe][:, 0]).mean())

    print(np.corrcoef(sim.data[neurons_out][:, 1], sim.data[origin_probe][:, 1]))
    print(np.square(sim.data[neurons_out][:, 1] - sim.data[origin_probe][:, 1]).mean())

    plt.figure()
    # plt.plot(sim.trange(), sim.data[neurons_probe], label="rates")
    # plt.plot(sim.trange(), sim.data[input_probe], label="Input signal")
    plt.plot(sim.trange(), sim.data[neurons_out][:, 0], label="Decoded estimate")
    # plt.plot(sim.trange(), sim.data[neurons_out][:, 1], label="Control signal")
    # plt.plot(sim.trange(), sim.data[external_input_probe][:, 0], label="external_input probe")
    plt.plot(sim.trange(), sim.data[origin_probe][:, 0], label="Origin X")
    # plt.plot(sim.trange(), sim.data[input_probe], label="input")
    plt.legend()
    plt.show()