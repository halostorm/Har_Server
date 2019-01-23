import numpy as np
import matplotlib.pyplot as plt


def readFile(file):
    with open (file,'r') as f:
        acc = []
        for line in f:
            line = line.strip().split('\t')
            x = float(line[2])
            y = float(line[3])
            z = float(line[4])
            acc.append(np.sqrt(x**2+y**2+z**2))
        return acc

def write(acc1,acc2, wF):
    with open(wF,'w') as f:
        l1 = len(acc1)
        l2 = len(acc2)
        L = np.min([l1,l2])
        print(L)
        for i in range(L):
            f.write(str(acc1[i])+'\t'+str(acc2[i])+'\n')

    graph(acc1,acc2)

def graph(d1,d2):
    plt.figure()
    plt.plot(d1,c = 'b',lw = 0.1)
    # plt.plot(d2,c = 'g')
    plt.show()

if __name__=='__main__':
    file1 = r'/home/halo/Workspace/Har_Server/qh/3la.txt'
    file2 = r'/home/halo/Workspace/Har_Server/qh/4la.txt'
    writeFile = '/home/halo/qh.txt'
    acc1 = readFile(file1)
    acc2 = readFile(file2)

    write(acc1,acc2,writeFile)
