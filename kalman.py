import numpy as np
import matplotlib.pyplot as plt

dim = 2


class Kalman(object):
    def __init__(self):
        self.M_DT_x = None
        self.M_DT_y = None

        self.M_CT_x = None
        self.M_CT_y = None

        self.A_ = None
        self.C_ = None
        self.W_ = None
        self.Q_ = None
        self.P_ = None

    def calculate(self, trainX, trainY, pool, dt, tau):

        """
        Calculate state matrix and observation matrix and so on
        :param trainX: neuron record
        :param trainY: x-y coordinate
        :param pool: pool index
        :param dt: time window
        :param tau: synaptic time constant
        :return:
        """

        M, T = np.shape(trainY)

        # trainY_new = np.ones((M + 1, T))
        # trainY_new[0:2, :] = trainY

        trainY_new = trainY
        # if pool == 0:
        #     trainY_new = np.mat(trainY[0, :])
        # elif pool == 1:
        #     trainY_new = np.mat(trainY[1, :])
        #
        A, y, e = self.least_squre(trainY_new[:, 0:-1], trainY_new[:, 1:], 0.1)
        W = np.dot(e, np.transpose(e)) / (T - 1)

        C, y, e = self.least_squre(trainY_new, trainX, 0.1)
        Q = np.dot(e, np.transpose(e)) / T

        A = np.mat(A)
        C = np.mat(C)
        W = np.mat(W)
        Q = np.mat(Q)

        self.A_ = A
        self.C_ = C
        self.W_ = W
        self.Q_ = Q

        self.P_ = np.mat(trainY_new) * np.mat(trainY_new).T / T

        K = np.linalg.pinv(np.eye(dim, dim) + W * C.T * Q.I * C) * W * C.T * Q.I

        self.M_DT_x = (np.eye(dim, dim) - K * C) * A
        self.M_DT_y = K

        Z, V = np.shape(self.M_DT_x)
        self.M_CT_x = (self.M_DT_x - np.eye(Z, V)) / dt
        self.M_CT_y = self.M_DT_y / dt

        A_ = tau * self.M_CT_x + np.eye(Z, V)
        B_ = tau * self.M_CT_y

        return A_, B_

    def least_squre(self, x, d, alpha):
        M, T = np.shape(x)

        R = np.dot(x, x.T) / T
        P = np.dot(x, d.T) / T
        C, D = np.shape(R)
        W = np.dot(np.linalg.pinv(R + alpha * np.eye(C, D)), P)
        W = np.transpose(W)
        y = np.dot(W, x)
        e = d - y

        return W, y, e

    def Kalman_Filter(self, testX, testY):

        """
        Kalman Filter without dynamic change on Kalman Gain
        :param testX: neuron record
        :param testY: x-y coordinate
        :return:
        """

        pre = []
        control = []
        initial = testY[:, 0]
        pre.append(initial)
        for i in range(2999):
            pre_state = self.M_DT_x * np.mat(initial).T
            pre_state = np.squeeze(np.asarray(pre_state))

            current_input = np.squeeze(np.asarray(self.M_DT_y * np.mat(testX[:, i]).T))

            initial = pre_state + current_input

            pre.append(initial)
            control.append(current_input)

        pre = np.transpose(pre)
        control = np.asarray(control)

        print(np.shape(pre), np.shape(testY))

        cc_x = np.corrcoef(pre[0, :], testY[0, :])
        cc_y = np.corrcoef(pre[1, :], testY[1, :])
        rmse_x = np.sqrt(np.mean((pre[0, :] - testY[0, :]) ** 2))
        rmse_y = np.sqrt(np.mean((pre[1, :] - testY[1, :]) ** 2))
        # rmse_x = np.square(pre[0, :] - testY[0, :])
        # rmse_y = np.square(pre[1, :] - testY[1, :]).mean()

        print(cc_x)
        print(cc_y)
        # 85%, 67%

        print(rmse_x)
        print(rmse_y)

        x = range(3000)
        plt.plot(x, pre[0, :], label='prediction')
        plt.plot(x, testY[0, :], label='origin')
        # plt.plot(range(2999), control, label='control')
        plt.legend()
        plt.show()

    def standard_Kalman_Filter(self, testX, testY, length=18000):

        """
        Kalman Filter with dynamic change on Kalman Gain
        :param testX: neuron record
        :param testY: x-y coordinate
        :return:
        """

        pre = []
        # initial = np.mat(testY[:, 0]).T
        initial = np.mat([0, 0]).T
        # pre.append(np.squeeze(np.asarray(initial)))
        print(testY)
        for i in range(length - 1):
            pre_state = self.A_ * initial
            Pa = self.A_ * self.P_ * self.A_.T + self.W_
            S = self.C_ * Pa * self.C_.T + self.Q_

            K = Pa * self.C_.T * np.linalg.inv(S)

            initial = pre_state + K * (np.mat(testX[:, i + 1]).T - self.C_ * self.A_ * pre_state)

            self.P_ = Pa - K * self.C_ * Pa

            pre.append(np.squeeze(np.asarray(initial)))

        pre = np.transpose(pre)
        cc_x = np.corrcoef(pre[0, :], testY[0, 1:length])
        cc_y = np.corrcoef(pre[1, :], testY[1, 1:length])
        # mse_x = np.square(pre[0, :] - testY[0, 0:18001]).mean()
        # mse_y = np.square(pre[1, :] - testY[1, 0:18001]).mean()

        rmse_x = np.sqrt(np.mean((pre[0, :] - testY[0, 1:length]) ** 2))
        rmse_y = np.sqrt(np.mean((pre[1, :] - testY[1, 1:length]) ** 2))

        print(cc_x)
        print(cc_y)
        # 85%, 67%

        print(rmse_x)
        print(rmse_y)

        x = range(length - 1)
        ax = plt.gca()
        ax.set_ylim([-1, 1])
        plt.plot(x, pre[0, :], label='prediction', linewidth=0.5)
        plt.plot(x, testY[0, 1:length], label='origin', linewidth=0.5)
        plt.legend()
        plt.show()

    def K_update(self, dt, tau):
        """
        update Kalman Gain during iteration
        :param dt: time window
        :param tau: synaptic time constant
        :return:
        """
        Pa = self.A_ * self.P_ * self.A_.T + self.W_
        S = self.C_ * Pa * self.C_.T + self.Q_

        K = Pa * self.C_.T * np.linalg.inv(S)

        self.M_DT_x = (np.eye(dim, dim) - K * self.C_) * self.A_
        self.M_DT_y = K

        self.P_ = Pa - K * self.C_ * Pa

        Z, V = np.shape(self.M_DT_x)
        self.M_CT_x = (self.M_DT_x - np.eye(Z, V)) / dt
        self.M_CT_y = self.M_DT_y / dt

        A_ = tau * self.M_CT_x + np.eye(Z, V)
        B_ = tau * self.M_CT_y

        return A_, B_

    def save(self, path=None):
        data = [self.A_, self.C_, self.W_, self.Q_, self.P_]
        np.save(path, data)

    def load(self, path=None):
        data = np.load(path, allow_pickle=True)
        self.A_ = data[0]
        self.C_ = data[1]
        self.W_ = data[2]
        self.Q_ = data[3]
        self.P_ = data[4]

    def getParam(self):
        data = [self.A_, self.C_, self.W_, self.Q_, self.P_]
        return data
