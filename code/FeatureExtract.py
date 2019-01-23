import numpy as np
import scipy.stats as ss
import os
import matplotlib.pyplot as plt


class FeatureExtract(object):
    AccNorm = []
    Euler = []
    Quaternion = []
    State = 0
    id = 0

    Features = []

    sampleRate = 0
    winSize = 0

    def __init__(self, WinSize, sampleRate):
        self.winSize = WinSize
        self.sampleRate = sampleRate

    def inputData(self, AccNorm, Quaternion=None, Euler=None, State=0):
        self.AccNorm = AccNorm
        self.Quaternion = Quaternion
        self.Euler = Euler
        self.State = State

    def timeDomainFeature(self):
        window = self.AccNorm
        a = np.mean(window)  # 均值
        b = np.std(window)  # 标准差
        # c = np.median(window)  # 中位数
        d = ss.skew(window)  # 峰度
        e = ss.kurtosis(window)  # 偏度
        sortedWindow = sorted(window)
        f = np.mean(sortedWindow[0:20])  # K个最大数均值
        g = np.mean(sortedWindow[-20:])  # K个最小数均值

        self.Features.append(a)
        self.Features.append(b)
        # self.Features.append(c)
        self.Features.append(d)
        self.Features.append(e)
        self.Features.append(f)
        self.Features.append(g)
        # print(self.Features)

    def frequencyDomainFeature(self):
        window = self.AccNorm
        sampleRate = self.sampleRate  # 采样率（样本频率）
        fftWindow = np.fft.rfft(window)  # 频谱系数
        Spectrum = list(map((lambda x: np.abs(x)), fftWindow))  # 幅度谱
        powerSpectrum = list(map((lambda x: x / sampleRate), Spectrum))  # 功率谱

        density = (int)(len(powerSpectrum) / (0.5 * sampleRate))
        # 取频率为0, 0-1, 1-2, 2-3 ... 24-25部分的功率谱，分别计算密度作为特征
        # self.Features.append(powerSpectrum[0])
        index = []
        for i in range(len(powerSpectrum)):
            index.append((i)*(0.5 * sampleRate)/len(powerSpectrum))
        draw2D_line(index,powerSpectrum,str(self.State)+'_'+str(self.id),'red')

        for i in range((int)(sampleRate * 0.5)):
            # print(i)
            if i == 0:
                self.Features.append(np.mean(powerSpectrum[1:density]))

            elif i  < 10:
                self.Features.append(np.mean(powerSpectrum[i * density:(i + 1) * density]))

            else:
                self.Features.append(np.mean(powerSpectrum[i * density:len(powerSpectrum) - 1]))

                break

def draw3D(x,y,z,name,colors):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,c = colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylabel('Z')
    ax.set_title("3D")

    plt.savefig('./3d/'+ str(name)+'.jpg')
    plt.show()


def draw2D_point(x,y,name,colors):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y,c= colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("2D")

    plt.savefig('./2d/'+ str(name)+'.jpg')
    plt.show()

def draw2D_line(x,y,name,colors):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("2D")
    plt.xlabel('f /(Hz)',size=16)
    plt.ylabel('A',size=16)
    plt.plot(x,y,c = 'b',label = 'frequency spectrum')
    my_y_ticks = np.arange(0, 10, 1)
    plt.yticks(my_y_ticks)
    my_x_ticks = np.arange(0, 26, 1)
    plt.tick_params(labelsize=15)
    plt.legend()
    plt.xticks(my_x_ticks)

    # plt.savefig('./2d/'+ str(name)+'.jpg')
    plt.show()