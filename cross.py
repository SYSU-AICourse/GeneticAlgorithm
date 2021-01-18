    def subtourExchangeCross(self, i, j, idx1, idx2):
        one1 = self.colony[i].copy()
        one2 = self.colony[j].copy()

        idxs = []
        for k in self.colony[i][idx1:idx2]:
            idxs.append(self.colony[j].index(k))
        
        idxs.sort()
        for k in range(idx2-idx1):
            self.colony[i][k+idx1], self.colony[j][idxs[k]] = self.colony[j][idxs[k]], self.colony[i][k+idx1]

        if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
            self.colony[i] = one1.copy()
        if(self.calculate(one2) > self.calculate(self.colony[j])) and random.random() > self.pro_cross:
            self.colony[j] = one2.copy()

        assert(len(set(self.colony[i])) == self.num_city)
        assert(len(set(self.colony[j])) == self.num_city)


    def cycleCross(self, i, j, idx1, idx2):
        one1 = self.colony[i].copy()
        one2 = self.colony[j].copy()

        startCity = one1[idx1]
        currentCity = one2[idx1]
        self.colony[i][idx1], self.colony[j][idx1] = self.colony[j][idx1], self.colony[i][idx1]
        while currentCity != startCity:
            idx = one1.index(currentCity)
            currentCity = one2[idx]
            self.colony[i][idx], self.colony[j][idx] = self.colony[j][idx], self.colony[i][idx]

        if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
            self.colony[i] = one1.copy()
        if(self.calculate(one2) > self.calculate(self.colony[j])) and random.random() > self.pro_cross:
            self.colony[j] = one2.copy()

        assert(len(set(self.colony[i])) == self.num_city)
        assert(len(set(self.colony[j])) == self.num_city)



    def positionBasedCross(self, i, j, idx1, idx2):
        one1 = self.colony[i].copy()
        one2 = self.colony[j].copy()

        seg1 = set([random.randint(0, self.num_city-1) for _ in range(idx2-idx1)])
        seg2 = set([random.randint(0, self.num_city-1) for _ in range(idx2-idx1)])

        p = 0
        for k in range(self.num_city):
            while self.colony[i][p] in seg1:
                p += 1
            if one2[k] not in seg1:
                self.colony[i][p] = one2[k]
                p += 1

        assert(p == self.num_city)

        p = 0
        for k in range(self.num_city):
            while self.colony[j][p] in seg2:
                p += 1
            if one1[k] not in seg2:
                self.colony[j][p] = one1[k]
                p += 1
        
        assert(p == self.num_city)


        if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
            self.colony[i] = one1.copy()
        if(self.calculate(one2) > self.calculate(self.colony[j])) and random.random() > self.pro_cross:
            self.colony[j] = one2.copy()

        assert(len(set(self.colony[i])) == self.num_city)
        assert(len(set(self.colony[j])) == self.num_city)


    def orderCross(self, i, j, idx1, idx2):
        one1 = self.colony[i].copy()
        one2 = self.colony[j].copy()

        seg1 = set(self.colony[i][idx1:idx2])
        seg2 = set(self.colony[j][idx1:idx2])

        p = 0
        for k in range(self.num_city):
            if p >= idx1:
                p = idx2
            if one2[k] not in seg1:
                self.colony[i][p] = one2[k]
                p += 1

        assert(p == self.num_city)

        p = 0
        for k in range(self.num_city):
            if p >= idx1:
                p = idx2
            if one1[k] not in seg2:
                self.colony[j][p] = one1[k]
                p += 1
        
        assert(p == self.num_city)


        if(self.calculate(one1) > self.calculate(self.colony[i])) and random.random() > self.pro_cross:
            self.colony[i] = one1.copy()
        if(self.calculate(one2) > self.calculate(self.colony[j])) and random.random() > self.pro_cross:
            self.colony[j] = one2.copy()

        assert(len(set(self.colony[i])) == self.num_city)
        assert(len(set(self.colony[j])) == self.num_city)

