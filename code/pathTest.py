import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

def readData(FilePath, times, Ts, writePath):
    pathX = []
    pathY = []
    pathZ = []
    time = []
    with open(FilePath, 'r') as f:
        i = 0
        for line in f:
            if len(line) < 10:
                if len(pathX) > 10:
                    # draw3D(pathX, pathY, pathZ, 'path_' + str(i)+'raw', 'r')
                    tt = np.mean(time)
                    id = findGps(tt, times)
                    pathX, pathY, pathZ = transform(pathX, pathY, pathZ, Ts[id])
                    # draw3D(pathX, pathY, pathZ, 'path_' + str(i)+'yaw', 'r')
                    WritePath(writePath, pathX, pathY, pathZ)
                    pathX = []
                    pathY = []
                    pathZ = []
                    i += 1
                continue
            line = line.strip().split('\t')
            time.append(float(line[0]))
            pathX.append(float(line[13]))
            pathY.append(float(line[14]))
            pathZ.append(float(line[15]))
            i += 1


def WritePath(Filepath, X, Y, Z):
    with open(Filepath, 'a+') as f:
        for i in range(len(X)):
            f.write(str(X[i]) + '\t' + str(Y[i]) + '\t' + str(Z[i]) + '\n')
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
            euler = (float(line[4])) / 360.0 * 2 * 3.14159265
            bear.append(euler)
            Ts.append(euler2matrix(euler))
            i += 1
    return time, Ts


def euler2matrix(e):
    T = np.mat(np.zeros((3, 3)))
    T[0, 0] = np.cos(e)
    T[0, 1] = -np.sin(e)

    T[1, 0] = np.sin(e)
    T[1, 1] = np.cos(e)

    T[2, 2] = 1
    return T.transpose()

def draw3D(x, y, z, name, colors, m,label=None):
    ax.plot(x, y, z, c=colors,marker=m, markersize =15,linestyle="None")
    ax.legend()
    ax.plot(x, y, z, c='k')
    ax.set_xlabel('X /(meters)')
    ax.set_ylabel('Y /(meters)')
    ax.set_zlabel('Z /(meters)')
    # ax.set_title("3D trajectories of Driver's phone and passenger's phone ")

def draw3D1(x, y, z, name, colors, m,label=None,titles = None):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c=colors,marker=m, markersize =15,linestyle="None",label=label)
    ax.legend()
    ax.plot(x, y, z, c='k')
    ax.set_xlabel('X /(meters)')
    ax.set_ylabel('Y /(meters)')
    ax.set_zlabel('Z /(meters)')
    ax.set_title(titles)
    plt.show()
    # ax.set_title("3D trajectories of Driver's phone and passenger's phone ")



def draw2D_point(x, y, name, colors):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, c=colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("2D")

    # plt.savefig('./2d/'+ str(name)+'.jpg')
    plt.show()


def draw2D_line(x, y, name, colors):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("2D")
    plt.xlabel('frequency')
    plt.ylabel('A')
    plt.plot(x, y, c=colors)
    plt.show()


def readDataPath(FilePath, savePath):
    X = []
    Y = []
    Z = []
    with open(FilePath) as f:
        id = 0
        for line in f:
            line = line.strip().split('\t')
            if len(line) > 2:
                if (np.abs(float(line[0]))<10 and np.abs(float(line[1]))<10 and np.abs(float(line[2]))<10):
                    X.append(float(line[0])/3)
                    Y.append(-float(line[1])/2)
                    Z.append(-float(line[2]))
            else:
                if (id==26 or len(Y)>0 and np.abs(Y[len(Y)-1])>1.5):
                    id+=1
                    X = []
                    Y = []
                    Z = []
                    continue
                if (len(Y)>0 and Y[int(len(Y) - 1)] < 0):
                    c = 'r'
                    m = '3'
                    label = 'driver'
                else:
                    c = 'b'
                    m = '4'
                    label = 'passenger'
                draw3D(X, Y, Z, "", c,m, label)#,str(id))
                with open(savePath,'w')as f:
                    for i in range(len(X)):
                        f.write(str(X[i])+'\t'+str(Y[i])+'\t'+str(Z[i])+'\n')
                X = []
                Y = []
                Z = []
                print(id)
                id += 1
    plt.show()


def readDataPathO(FilePath):
    X = []
    Y = []
    Z = []
    with open(FilePath) as f:
        i = 0
        k = 0
        for line in f:
            line = line.strip().split('\t')
            print(len(line))
            if len(line) > 2:
                # if (np.abs(float(line[0]))<10 and np.abs(float(line[1]))<10 and np.abs(float(line[2]))<10):
                if i%3==0:
                    X.append(float(line[0]))
                    Y.append(float(line[1]))
                    if k == 1:
                        Z.append(-float(line[2]))
                    else:
                        Z.append(float(line[2]))
            else:
                if (Y[int(len(Y) - 1)] > 0):
                    c = 'r'
                    m = '3'
                else:
                    c = 'b'
                    m = '4'
                draw3D(X, Y, Z, "", c,m)
                print(Z)
                X = []
                Y = []
                Z = []
                k+=1
            i += 1
    plt.show()


if __name__ == '__main__':
    # times, Ts = readGps(r'../call/Location.txt')
    # readData(r'../call/path_.txt',times,Ts,r'../call/path.txt')

    pathFile = r'/home/halo/Workspace/Har_Server/Path_Texting.txt'
    saveFileL = r'/home/halo/Workspace/Har_Server/Path_TextingL.txt'
    saveFileR = r'/home/halo/Workspace/Har_Server/Path_TextingR.txt'

    pathFileO = r'/home/halo/Workspace/Har_Server/Path_One.txt'
    readDataPath(pathFile,saveFileL)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # x = np.linspace(0, 100, 30)
    # y = np.linspace(0, 100, 30)
    # z = np.linspace(0, 100, 30)
    #
    # ax.plot(x,y,z, c='b', marker='3', markersize=15, linestyle="None")
    # ax.plot(x, y, z, c='k')
    # ax.legend()
    # plt.show()
