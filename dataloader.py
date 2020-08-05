import scipy.io as scio
import numpy as np


class Dataloader(object):

    def __init__(self):
        train_kin_data = scio.loadmat('./data/2019-11-11-S2.mat')['train_Y']
        train_neural_data = scio.loadmat('./data/2019-11-11-S2.mat')['train_X']

        # test_kin_data = scio.loadmat('./data/2019-11-11-S2.mat')['train_Y']
        # test_neural_data = scio.loadmat('./data/2019-11-11-S2.mat')['train_X']
        #
        # train_kin_data = train_kin_data[:, 0:2500]
        # train_neural_data = train_neural_data[:, 0:2500]
        #
        # test_kin_data = test_kin_data[:, 2500:3925]
        # test_neural_data = test_neural_data[:, 2500:3925]

        test_kin_data = scio.loadmat('./data/2019-11-11-S2.mat')['test_Y']
        test_neural_data = scio.loadmat('./data/2019-11-11-S2.mat')['test_X']

        # train_kin_data = train_kin_data[0:2, :]
        # test_kin_data = test_kin_data[0:2, :]
        # train_kin_data = scio.loadmat('./train/KinData1.mat')['KinData']
        # train_neural_data = scio.loadmat('./train/NeuralData1.mat')['NeuralData']
        #
        # test_kin_data = scio.loadmat('./test/KinData1.mat')['KinData']
        # test_neural_data = scio.loadmat('./test/NeuralData1.mat')['NeuralData']

        # 9799
        # train_kin_data = scio.loadmat('./ob_avoid_succ/2014031201.mat')['KinData']
        # train_neural_data = scio.loadmat('./ob_avoid_succ/2014031201.mat')['NeuralData']
        #
        # train_kin_data = train_kin_data[:, 0:4500]
        # train_neural_data = train_neural_data[:, 0:4500]
        # #
        # # # 10402
        # test_kin_data = scio.loadmat('./ob_avoid_succ/2014031201.mat')['KinData']
        # test_neural_data = scio.loadmat('./ob_avoid_succ/2014031201.mat')['NeuralData']
        #
        # test_kin_data = test_kin_data[:, 4500:9000]
        # test_neural_data = test_neural_data[:, 4500:9000]

        self.trainY = np.transpose(train_kin_data)
        t_mean = np.mean(self.trainY, axis=0)
        t_std = np.std(self.trainY, axis=0)

        self.trainY = np.transpose((self.trainY - t_mean) / t_std)

        self.testY = np.transpose(test_kin_data)
        self.testY = np.transpose((self.testY - t_mean) / t_std)

        self.trainX = train_neural_data
        self.testX = test_neural_data

    def getTrainData(self, type):
        if type == 'KinData':
            return self.trainY
        elif type == 'NeuralData':
            return self.trainX

    def getTestData(self, type):
        if type == 'KinData':
            return self.testY
        elif type == 'NeuralData':
            return self.testX

    def getData(self):
        return self.trainX, self.trainY, self.testX, self.testY
