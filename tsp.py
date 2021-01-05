from typing import Optional
import xml.etree.ElementTree as ET
import random
import numpy as np


class GeneticAlgorithmTSP(object):
    def __init__(self, xmlpath, optimal, num_city, num_population=150, pro_cross=0.3, pro_mutate=0.1, pro_reverse=0.1, pro_search=0.4):
        self.num_city = num_city
        self.num_population = num_population
        self.pro_cross = pro_cross
        self.pro_mutate = pro_mutate
        self.pro_reverse = pro_reverse
        self.pro_search = pro_search
        self.optimal = optimal

        self.parseTSPXML(xmlpath)
        self.initial()
        # print(self.colony)


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
        return 1 / (totDist - self.optimal)

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
            pick = self.stochastic_tournament()
            new_colony.append(self.colony[pick])
    
        self.colony = new_colony.copy()
    

    def cross(self):
        for i in range(0, self.num_population, 2):
            idx1 = random.randint(0, self.num_city-2)
            idx2 = random.randint(idx1+1, self.num_city)
            
            one1 = self.colony[i].copy()
            one2 = self.colony[i+1].copy()

            self.colony[i][idx1:idx2], self.colony[i+1][idx1:idx2] = self.colony[i+1][idx1:idx2], self.colony[i][idx1:idx2]
            seg1 = set(self.colony[i][idx1:idx2])
            seg2 = set(self.colony[i+1][idx1:idx2])
            x, y = 0, 0
            while x < self.num_city and y < self.num_city:
                while x < self.num_city and not ((x < idx1 or x >= idx2) and self.colony[i][x] in seg1):
                    x += 1

                while y < self.num_city and not ((y < idx1 or y >= idx2) and self.colony[i+1][y] in seg2):
                    y += 1
                
                if x < self.num_city and y < self.num_city:
                    self.colony[i][x], self.colony[i+1][y] = self.colony[i+1][y], self.colony[i][x]


                x += 1
                y += 1
            
            if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
                self.colony[i] = one1.copy()
            if(self.calculate(one2) > self.calculate(self.colony[i+1])) and random.random() > self.pro_cross:
                self.colony[i+1] = one2.copy()

            assert(len(set(self.colony[i])) == self.num_city)
            assert(len(set(self.colony[i+1])) == self.num_city)


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
            
            one = self.colony[k].copy()

            idx1 = random.randint(1, self.num_city-3)
            idx2 = random.randint(idx1+2, self.num_city-1)

            cur = self.colony[k][idx1]
            seg = set(self.colony[k][idx1:idx2])
            seg.remove(cur)
            for j in range(idx1+1, idx2):
                minval = 1e7
                next = 0
                for i in range(self.num_city):
                    if i in seg and self.cities[cur][i] < minval:
                        minval = self.cities[cur][i]
                        next = i
                
                cur = next
                self.colony[k][j] = cur
                seg.remove(cur)
            
            if self.calculate(one) > self.calculate(self.colony[k]) and random.random() > self.pro_search:
                self.colony[k] = one

            assert(len(set(self.colony[k])) == self.num_city)
    

    def Solve(self, epochs=200):
        optimal = 1e7
        for _ in range(epochs):
            self.select()
            self.cross()
            self.mutate()
            self.reverse()
            self.search()
            opt = min([sum([self.cities[one[x-1]][one[x]] for x in range(self.num_city)]) for one in self.colony])
            print(_, opt)
            optimal = min(optimal, opt)
        
        return optimal

    def Greedy(self):
        cur = 0
        optimal = 0
        cities = self.cities.copy()
        for _ in range(self.num_city-1):
            cities[cur][cur] = 1e7
            next = np.argmin(cities[cur])
            optimal += cities[cur][next]
            cur = next
        
        optimal += cities[cur][0]
        return optimal


if __name__ == "__main__":
    solution = GeneticAlgorithmTSP('a280.xml', 2579, 280)
    print(solution.Solve(epochs=300))