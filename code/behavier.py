import numpy as np
import matplotlib.pyplot as plt
import math


def readAtt(FilePath, times, Yaw):
    ekfX = []
    ekfY = []
    ekfZ = []
    Q = []

    with open(FilePath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if (len(line) >17):
                tt = float(line[1])
                id = findGps(tt, times)
                yaw = Yaw[id]
                yaw_Q = euler2q(yaw)
                ekfQ = []
                ekfQ.append(float(line[11]))
                ekfQ.append(float(line[12]))
                ekfQ.append(float(line[13]))
                ekfQ.append(float(line[14]))


                # q_new = q_x(ekfQ,yaw_Q)
                # q_new = q_x(yaw_Q,ekfQ)
                q_new = ekfQ
                Q.append(q_new)

                ekfEuler = Q2Euler(ekfQ)
                ekfX.append(ekfEuler[0])
                ekfY.append(ekfEuler[1])
                ekfZ.append(ekfEuler[2])
    Q = np.matrix(Q)

    return ekfX, ekfY, ekfZ, Q


def WriteEuler(Filepath, W, X, Y, Z):
    with open(Filepath, 'a+') as f:
        for i in range(len(X)):
            f.write(str(W[i]) + '\t' + str(X[i]) + '\t' + str(Y[i]) + '\t' + str(Z[i]) + '\n')
        f.write('\n')


def transform(x, y, z, T):
    for i in range(len(x)):
        x_ = T[0, 0] * x[i] + T[0, 1] * y[i] + T[0, 2] * z[i]
        y_ = T[1, 0] * x[i] + T[1, 1] * y[i] + T[1, 2] * z[i]
        z_ = T[2, 0] * x[i] + T[2, 1] * y[i] + T[2, 2] * z[i]
        x[i] = x_
        y[i] = y_
        z[i] = z_
    return x, y, z


def findGps(t, tts):
    min = 10000000000000000
    id = 0
    for i in range(len(tts)):
        if min > (np.abs(t - tts[i])):
            min = np.abs(t - tts[i])
            id = i
    return id


def readGps(FilePath):
    time = []
    bear = []
    Ts = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            line = line.strip().split('\t')
            time.append(float(line[0]))
            euler = (float(line[4])) / 360.0 * 2 * np.pi -np.pi
            bear.append(euler)
            Ts.append(euler2matrix(euler))
            i += 1
    return time, bear


def Q2Euler(q):
    eulerAngles = []
    eulerAngles.append(
        (float)(np.math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                              1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))))  # / np.pi * 180))
    eulerAngles.append((float)(
        math.asin(2.0 * (q[0] * q[2] - q[3] * q[1]))))
    eulerAngles.append((float)(
        math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))))
    return eulerAngles


def euler2matrix(e):
    T = np.mat(np.zeros((3, 3)))
    T[0, 0] = np.cos(e)
    T[0, 1] = -np.sin(e)

    T[1, 0] = np.sin(e)
    T[1, 1] = np.cos(e)

    T[2, 2] = 1
    return T.transpose()


def euler2q(e):
    q = []
    q.append(np.cos(e / 2))
    q.append(0)
    q.append(0)
    q.append(-np.sin(e / 2))
    return q


def q_x(q1, q2):
    q = [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]]
    return q


def graph(d1, d2, d3, d4):
    plt.figure()
    plt.plot(d1, c='k', lw=0.5)
    plt.plot(d2, c='r', lw=0.5)
    plt.plot(d3, c='g', lw=0.5)
    plt.plot(d4, c='b', lw=0.5)
    plt.show()


if __name__ == '__main__':
    times, Yaw = readGps(r'/home/halo/Workspace/Har_Server/behavier/Data2018_12_16_11_12_0/Location_2018_12_16_11_12_0.txt')

    X, Y, Z, Q = readAtt(r'/home/halo/Workspace/Har_Server/behavier/Data2018_12_16_11_12_0/raw_2018_12_16_11_12_0.txt', times, Yaw)

    # graph(Q.A.T[0], Q.A.T[1], Q.A.T[2], Q.A.T[3])

    # WriteEuler(r'/home/halo/attitude_static.txt',Q.A.T[0], Q.A.T[1], Q.A.T[2], Q.A.T[3])
