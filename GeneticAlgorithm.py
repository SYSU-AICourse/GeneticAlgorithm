from typing import Optional
import xml.etree.ElementTree as ET
import random
import numpy as np


class GeneticAlgorithmTSP(object):
    def __init__(self, filepath, num_city, num_population=150, pro_cross=0.3, num_cross=1, pro_mutate=0.1, pro_reverse=0.1, pro_search=0.4, num_search=1):
        self.num_city = num_city
        self.num_population = num_population
        self.pro_cross = pro_cross
        self.pro_mutate = pro_mutate
        self.pro_reverse = pro_reverse
        self.pro_search = pro_search
        self.num_cross = num_cross
        self.num_search = num_search

        self.parseTSP(filepath)
        self.initial()
        # print(self.colony)

    def parseTSP(self, filepath):
        points = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                points.append([float(x) for x in line.split()])
        self.points = np.array(points)

        self.cities = [[0 for i in range(self.num_city)] for j in range(self.num_city)]
        for i in range(self.num_city):
            for j in range(i+1, self.num_city):
                self.cities[i][j] = self.cities[j][i] = self.dist(points[i][1] - points[j][1], points[i][2] - points[j][2])

    def dist(self, dx, dy):
        return (dx*dx + dy*dy) ** 0.5

    def parseTSPXML(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        self.cities = []
        for node in root.iter("vertex"):
            edges = [0] * self.num_city
            for edge in node:
                num = int(edge.text)
                cost = float(edge.attrib['cost'])
                edges[num] = cost
            
            self.cities.append(edges)


    def initial(self):
        self.colony = []
        for _ in range(self.num_population):
            one = list(range(self.num_city))
            random.shuffle(one)
            self.colony.append(one)


    def calculate(self, one):
        totDist = sum([self.cities[one[x-1]][one[x]] for x in range(self.num_city)])
        return 1 / max(totDist - self.best, 0.1)

    def roulette_gambler(self):
        tot = sum([self.calculate(x) for x in self.colony])
        pros = [self.calculate(x) / tot for x in self.colony]
        pick = random.random()
        for i in range(self.num_population):
            pick -= pros[i]
            if pick <= 0:
                return i
        
        return 0

    def stochastic_tournament(self):
        pick1 = self.roulette_gambler()
        pick2 = self.roulette_gambler()
        return pick1 if self.calculate(self.colony[pick1]) > self.calculate(self.colony[pick2]) else pick2


    def select(self):
        new_colony = []
        for _ in range(self.num_population):
            pick = self.roulette_gambler()
            new_colony.append(self.colony[pick])
    
        self.colony = new_colony.copy()
    

    def cross(self):
        pairs = list(range(self.num_population))
        random.shuffle(pairs)
        for i in range(1, self.num_population, 2):
            idx = [random.randint(0, self.num_city-1) for _ in range(2*self.num_cross)]
            idx.sort()

            for j in range(0, len(idx), 2):
                self.crossFunc(pairs[i-1], pairs[i], idx[j], idx[j+1])
            

    def crossFunc(self, i, j, idx1, idx2):
        one1 = self.colony[i].copy()
        one2 = self.colony[j].copy()

        self.colony[i][idx1:idx2], self.colony[j][idx1:idx2] = self.colony[j][idx1:idx2], self.colony[i][idx1:idx2]
        seg1 = set(self.colony[i][idx1:idx2])
        seg2 = set(self.colony[j][idx1:idx2])
        x, y = 0, 0
        while x < self.num_city and y < self.num_city:
            while x < self.num_city and not ((x < idx1 or x >= idx2) and self.colony[i][x] in seg1):
                x += 1

            while y < self.num_city and not ((y < idx1 or y >= idx2) and self.colony[j][y] in seg2):
                y += 1
            
            if x < self.num_city and y < self.num_city:
                self.colony[i][x], self.colony[j][y] = self.colony[j][y], self.colony[i][x]

            x += 1
            y += 1
        
        if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
            self.colony[i] = one1.copy()
        if(self.calculate(one2) > self.calculate(self.colony[j])) and random.random() > self.pro_cross:
            self.colony[j] = one2.copy()

        assert(len(set(self.colony[i])) == self.num_city)
        assert(len(set(self.colony[j])) == self.num_city)


    def mutate(self):
        for i in range(self.num_population):
            one = self.colony[i].copy()

            idx1 = random.randint(0, self.num_city-1)
            idx2 = random.randint(0, self.num_city-1)
            if idx1 == idx2:
                continue
            
            self.colony[i][idx1], self.colony[i][idx2] = self.colony[i][idx2], self.colony[i][idx1]

            if self.calculate(one) > self.calculate(self.colony[i]) and random.random() > self.pro_mutate:
                self.colony[i] = one


    def reverse(self):
        for i in range(self.num_population):
            one = self.colony[i].copy()
            if random.random() > self.pro_mutate:
                continue

            idx1 = random.randint(1, self.num_city-2)
            idx2 = random.randint(idx1+1, self.num_city-1)
            
            self.colony[i][idx1:idx2] = self.colony[i][idx2-1:idx1-1:-1]

            if self.calculate(one) > self.calculate(self.colony[i]) and random.random() > self.pro_reverse:
                self.colony[i] = one

    def search(self):
        for k in range(self.num_population):
            idx = [random.randint(0, self.num_city-1) for _ in range(2*self.num_search)]
            idx.sort()

            for j in range(0, len(idx), 2):
                self.searchFunc(k, idx[j], idx[j+1])


    def searchFunc(self, k, idx1, idx2):
        cur = self.colony[k][idx1]
        seg = set(self.colony[k][idx1+1:idx2])

        one = self.colony[k].copy()
        for j in range(idx1+1, idx2):
            minval = 1e7
            next = 0
            for i in seg:
                if self.cities[cur][i] < minval:
                    minval = self.cities[cur][i]
                    next = i
            
            cur = next
            self.colony[k][j] = cur
            seg.remove(cur)
        
        if self.calculate(one) > self.calculate(self.colony[k]) and random.random() > self.pro_search:
            self.colony[k] = one

        assert(len(set(self.colony[k])) == self.num_city)
    

    def Solve(self, epochs=200):
        history = {}
        optimal = 1e7
        paths = []
        stepopt = []
        path = []
        for _ in range(epochs):
            self.best = min([sum([self.cities[one[x-1]][one[x]] for x in range(self.num_city)]) for one in self.colony])
            self.select()
            self.cross()
            self.mutate()
            # self.reverse()
            self.search()

            vals = [sum([self.cities[one[x-1]][one[x]] for x in range(self.num_city)]) for one in self.colony]
            opt = 1e7
            k = 0
            for i in range(len(vals)):
                if opt > vals[i]:
                    opt = vals[i]
                    k = i
            print(_, opt)

            stepopt.append(opt)
            paths.append(self.colony[k])
            # optimal = min(optimal, opt)
            if(opt < optimal):
                optimal = opt
                path = self.colony[k].copy()
        
        history['optimal'] = optimal
        history['paths'] = paths.copy()
        history['path'] = path.copy()
        history['step'] = stepopt

        self.history = history
        return history

    def Greedy(self):
        minval = 1e7

        for start in range(self.cities):
            path = [start]
            answer = 0
            for _ in range(self.num_city-1):
                best = 1e7
                k = 0
                for i in range(self.num_city):
                    if i not in set(path) and self.cities[path[-1]][i] < best:
                        best = self.cities[path[-1]][i]
                        k = i
                path.append(k)
                answer += best

            answer += self.cities[path[-1]][start]
            assert(len(path) == self.num_city)
            minval = min(minval, answer)
        
        return minval


def visualize(points):
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(points[:, 1], points[:, 2], "o")
    iter = 0
    while True:
        if iter >= points[:, 0].size - 1:
            break
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
            iter += 1


if __name__ == "__main__":
    solution = GeneticAlgorithmTSP(
        'xqf131.tsp', 131,
        num_population=150,
        pro_cross=0.3,
        num_cross=1,
        pro_mutate=0.05,
        pro_reverse=0,
        pro_search=0.4,
        num_search=2
    )
    history = solution.Solve(epochs=200)
    # print(ans)
    visualize(solution.points[history['path'][:]])
    # import matplotlib.pyplot as plt
    # plt.plot()
    # print(np.array(solution.cities).shape)