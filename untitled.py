import numpy as np
import matplotlib.pyplot as plt


def init():
    # 退火系数
    alpha = 0.99
    # 终止温度，初始温度
    T = (1e-4, 1000)
    # 一个温度的循环次数
    L = 1000
    return alpha, T, L


def sa():
    num = points[:, 0].size
    distmat = getdistmatrix(points)
    steptemp = np.arange(num)

    valuecurrent = 0
    for i in range(num):
        for j in range(i, num):
            valuecurrent += distmat[i][j]

    valuebest = valuecurrent

    alpha, temp, l = init()
    t = temp[1]
    tmin = temp[0]
    # result = []
    while t > tmin:
        # print(t, tmin)
        for i in np.arange(l):
            if np.random.rand() > 0.5:
                while True:
                    r1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    r2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    if r1 != r2:
                        break
                steptemp[r1], steptemp[r2] = steptemp[r2], steptemp[r1]
            else:
                while True:
                    r1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    r2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    r3 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    if (r1 != r2) & (r2 != r3) & (r3 != r1):
                        break
                if r1 > r2:
                    r1, r2 = r2, r1
                if r2 > r3:
                    r2, r3 = r3, r2
                if r1 > r2:
                    r1, r2 = r2, r1
                temp = steptemp[r1:r2].copy()
                steptemp[r1 : r3 - r2 + 1 + r1] = steptemp[r2 : r3 + 1].copy()
                steptemp[r3 - r2 + 1 + r1 : r3 + 1] = temp.copy()

            valuetemp = 0
            for i in range(num - 1):
                valuetemp += distmat[steptemp[i]][steptemp[i + 1]]
            valuetemp += distmat[steptemp[0]][steptemp[num - 1]]

            if valuetemp < valuecurrent:
                valuecurrent = valuetemp
                stepcurrent = steptemp.copy()

                if valuetemp < valuebest:
                    valuebest = valuetemp
                    stepbest = steptemp.copy()
            else:
                if np.random.rand() < np.exp((valuecurrent - valuetemp) / t):
                    valuecurrent = valuetemp
                    stepcurrent = steptemp.copy()
                else:
                    steptemp = stepcurrent.copy()

        t = alpha * t
        print(valuebest)
        # result.append(valuebest)

    return stepbest


def getdistmatrix(points):
    num = points[:, 0].size
    distmatrix = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):
            distmatrix[i][j] = distmatrix[j][i] = (
                (points[i, 1] - points[j, 1]) ** 2 + (points[i, 2] - points[j, 2]) ** 2
            ) ** 0.5
    return distmatrix


def visualize(points):
    plt.figure(1)
    plt.ion()
    plt.plot(points[:, 1], points[:, 2], "o")
    iter = 0
    while True:
        if iter >= points[:, 0].size - 1:
            plt.pause(2)
            plt.clf()
            plt.plot(points[:, 1], points[:, 2], "o")
            iter = 0
        else:
            plt.plot(
                [points[iter, 1], points[iter + 1, 1]],
                [points[iter, 2], points[iter + 1, 2]],
                c="r",
                ls="-",
                marker="o",
                mec="b",
                mfc="w",
            )
            plt.pause(0.1)
            iter += 1


if __name__ == "__main__":
    with open(r"xqf131.tsp", "r") as f:
        points = []
        for line in f.readlines():
            points.append([float(x) for x in line.split()])
        points = np.array(points)
        solution = sa()
        solution = np.append(solution, solution[0])
        # visualize(points[solution[:]])
