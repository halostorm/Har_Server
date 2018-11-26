import numpy as np
import scipy.stats as ss
import os

class FeatureExtract(object):
    AccNorm = []
    Euler = []
    Quaternion = []
    State = 0

    Features = []

    sampleRate = 0
    winSize = 0

    def __init__(self, WinSize, sampleRate):
        self.winSize = WinSize
        self.sampleRate = sampleRate

    def inputData(self, AccNorm, Quaternion = None, Euler=None, State=0):
        self.AccNorm = AccNorm
        self.Quaternion = Quaternion
        self.Euler = Euler
        self.State = State

    def timeDomainFeature(self):
        window = self.AccNorm
        staticFeature = []
        a = np.mean(window)  # 均值
        b = np.std(window)  # 标准差
        c = np.median(window) #中位数
        d = ss.skew(window)  # 峰度
        e = ss.kurtosis(window)  # 偏度
        sortedWindow = sorted(window)
        f = np.mean(sortedWindow[0:20])  # K个最大数均值
        g = np.mean(sortedWindow[-20:])  # K个最小数均值

        staticFeature.append(a)
        staticFeature.append(b)
        staticFeature.append(c)
        staticFeature.append(d)
        staticFeature.append(e)
        staticFeature.append(f)
        staticFeature.append(g)
        #print(staticFeature)
        self.Features.append(staticFeature)

    def frequencyDomainFeature(self):
        window = self.AccNorm
        sampleRate = self.sampleRate  # 采样率（样本频率）
        fftWindow = np.fft.rfft(window)  # 频谱系数
        Spectrum = list(map((lambda x: np.abs(x)), fftWindow))  # 幅度谱
        powerSpectrum = list(map((lambda x: x / sampleRate), Spectrum))  # 功率谱

        SpectrumFeatures = []
        density = (int)(len(powerSpectrum) / (0.5 * sampleRate))
        # 取频率为0, 0-1, 1-2, 2-3 ... 24-25部分的功率谱，分别计算密度作为特征
        # SpectrumFeatures.append(powerSpectrum[0])
        for i in range((int)(sampleRate * 0.5)):
            if i == 0:
                SpectrumFeatures.append(np.mean(powerSpectrum[1:density]))
            else:
                SpectrumFeatures.append(np.mean(powerSpectrum[i * density:(i + 1) * density]))

        self.Features.extend(SpectrumFeatures)
        #print(SpectrumFeatures)


def readData(FilePath):
    accNorm = []
    with open(FilePath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            tmp = np.sqrt((float)(line[2]) ** 2 + (float)(line[3]) ** 2 + (float)(line[4]) ** 2)-9.806
            accNorm.append(tmp)
    return accNorm

def extractFeatures(accNorm,state,featurePath):
    featureExtract = FeatureExtract(512, 100)
    featureStream = []
    N = len(accNorm)/512
    for i in range(N):
        featureExtract.inputData(accNorm[512 * i:512 * i+1],state)
        featureExtract.timeDomainFeature()
        featureExtract.frequencyDomainFeature()
        featureStream.append(featureExtract.Features)
    #Save Features


if __name__ == '__main__':

    FilePath = r'../traindata/data/'
    for root, dirs, fileName in os.walk(FilePath):
        for i in fileName:
            dataDir = os.path.join(root, i)
            accNorm = readData(dataDir)
            print(dataDir)
