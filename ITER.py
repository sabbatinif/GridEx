from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import random as rnd
from joblib import load

class ITER:
    def __overlap(self, c1, c2):
        ret = True
        for f1 in c1:
            if f1 != self.target:
                [a1, b1] = c1[f1]
                [a2, b2] = c2[f1] 
                if (a2 >= b1) or (a1 >= b2):
                    return False
        return ret
    
    def __equal(self, cube, hyperCubes):
        c1 = { i : cube[i] for i in cube if i != self.target}
        for f in c1:
            [a, b] = c1[f]
            if abs(a - b) < 1e-3:
                return True
        for hc in hyperCubes:
            c2 = { i : hyperCubes[hc][i] for i in hyperCubes[hc] if i != self.target}
            eq = True
            for f in c1:
                [a1, b1] = c1[f]
                [a2, b2] = c2[f]
                eq &= abs(a1 - a2) < 1e-3
                eq &= abs(b1 - b2) < 1e-3
            if eq:
                return True                
        return False

    def __checkOverlap(self, toCheck, hyperCubes):
        checked = []   
        overlapping = []

        for hc1 in toCheck:
            c1 = toCheck[hc1]
            checked.append(hc1)
            for hc2 in hyperCubes:        
                c2 = hyperCubes[hc2]
                if (hc2 not in checked) and (self.__overlap(c1, c2)):
                    return c2, hc2
        return None
            
    def __getCond(self, c, samples, feat):            
        cond = np.ones((len(samples),), dtype = bool)
        for i, f in enumerate(feat):
            [a, b] = c[f]
            col = samples[:, i]
            cond &= ((a <= col) & (col <= b))
        return cond

    def __updateMean(self, c, samples, feat):
        cond = self.__getCond(c, samples, feat)
        if len(samples[cond] > 0):
            c[self.target] = self.model.predict(samples[cond]).mean()  
    
    def __count(self, c, samples, feat):
        cond = self.__getCond(c, samples, feat)
        return len(np.nonzero(cond)[0]), samples[cond].tolist()   
            
    def __createTempCubes(self, c, minUpdates, feat, surrounding, hyperCubes, limits, hc):
        temp = []
        for i in feat:
            res = self.__checkLimits(limits, hc, i)
            if res == "*":
                continue
            [a, b] = c[i]
            # never go beyond the surrounding cube
            [v0, v1] = surrounding[i]
            a_ = max(a - minUpdates[i], v0)
            b_ = min(b + minUpdates[i], v1)
            # create the temp hyper-cubes
            tempCube = c.copy()  
            if res != "-":
                if a_ == v0:
                    limits.add((hc, i, "-"))
                tempCube[i] = [a_, a]
                # check overlapping for the lower cube
                res = self.__checkOverlap({ hc : tempCube }, hyperCubes)
                if res != None:
                    over, idx = res
                    limits.add((hc, i, "-"))
                    limits.add((idx, i, "+"))
                    a_ = max(over[i][1], a_)
                    tempCube[i] = [a_, a]
                    res = self.__checkOverlap({ hc : tempCube }, hyperCubes)
                if (a_ < a) and (res == None) and (not self.__equal(tempCube, hyperCubes)):
                    temp.append((tempCube.copy(), (i, [a_, a], "-")))
                else:
                    limits.add((hc, i, "-"))
            if res != "+":
                if b_ == v1:
                    limits.add((hc, i, "+"))               
                tempCube[i] = [b, b_]
                # check overlapping for the upper cube
                res = self.__checkOverlap({ hc : tempCube }, hyperCubes)
                if res != None:
                    over, idx = res
                    limits.add((hc, i, "+"))
                    limits.add((idx, i, "-"))  
                    b_ = min(over[i][0], b_)
                    tempCube[i] = [b, b_] 
                res = self.__checkOverlap({ hc : tempCube }, hyperCubes)            
                if (b < b_) and (res == None) and (not self.__equal(tempCube, hyperCubes)):
                    temp.append((tempCube.copy(), (i, [b, b_], "+")))
                else:
                    limits.add((hc, i, "+"))                  
        return temp

    def __checkLimits(self, limits, cube, feature):
        res = None
        for (c, f, s) in limits:
            if (cube == c) & (feature == f):
                if res == None:
                    res = s
                else:
                    return "*"
        return res

    def __merge(self, c, triple, hyperCubes):
        (t, r, sign) = triple
        [a1, b1] = r
        [a2, b2] = c[t]                          
        if sign == "-":
            c[t] = [a1, b2]
            ov = self.__checkOverlap(hyperCubes, hyperCubes)
            self.hyperCubes = hyperCubes
            if ov is not None:
                [a3, b3] = ov[0][t]
                c[t] = [b3, b2]
        else:
            c[t] = [a2, b1]
            ov = self.__checkOverlap(hyperCubes, hyperCubes)
            if ov is not None:
                [a3, b3] = ov[0][t]
                c[t] = [a2, b3]                               
        
    def __iterate(self, hyperCubes, minUpdates, feat, fake, surrounding, limits):
        toAdd = []
        toUp = []
        for hc in hyperCubes:    
            temp = self.__createTempCubes(hyperCubes[hc], minUpdates, feat, surrounding, hyperCubes, limits, hc)           
            meanTemp = []
            cubes = []
            means = []
            for (c, info) in temp: # for each temp hyper-cube
                n, pred = self.__count(c, np.array(self.tr), feat)
                
                for i in range(n, 100):  # repeat n times
                    sample = []
                    for f in feat:      # for each dimension                 
                        [a, b] = c[f]
                        sample.append(rnd.uniform(a, b))                
                    pred.append(sample)
                    fake.append(sample)
                mean = self.model.predict(pred).mean()
                c[self.target] = mean
                cubes.append(c)
                meanTemp.append(info)
                means.append(abs(hyperCubes[hc][self.target] - mean))
            if len(means) > 0:
                means = np.array(means)
                idx = np.where(means == min(means))[0][0]
                toAdd.append((min(means), cubes[idx], hc, meanTemp[idx]))                        
        if len(toAdd) > 0:
            (m, c, hc, mt) = min(toAdd)
            if m > self.THRESHOLD:
                hyperCubes[len(hyperCubes)] = c           
            else:                            
                self.__merge(hyperCubes[hc], mt, hyperCubes)  
        return toUp

    def __measureOverlap(self, c1, c2, feat):
        diff = []
        over = []
        for f in feat:
            a1, b1 = c1[f]
            a2, b2 = c2[f]        
            diff.append(min(b1, b2) - max(a1, a2))
            over.append([max(a1, a2), min(b1, b2)])
        return diff, over

    def __cubeFromPoint(self, point, hyperCubes, feat, minUpdates, samples):
        yyy = False
        [[s]] = self.model.predict(point)
        cube = { self.target : s }
        for j, f in enumerate(feat):
            a, b = max(point[0][j] - minUpdates[f] / self.ratio, self.minmax[f][0]), \
                   min(point[0][j] + minUpdates[f] / self.ratio, self.minmax[f][1])        
            cube[f] = [a, b]
        res = self.__checkOverlap({ len(hyperCubes) : cube }, hyperCubes)
        while res != None:
            diff, over = self.__measureOverlap(cube, res[0], feat)     
            sort = diff.copy()
            sort.sort()     
            yyy = True
            for s in sort:            
                idx = diff.index(s)
                [a, b] = over[idx]
                p = point[0][idx]
                if a < p < b:
                    continue
                [a, b] = cube[feat[idx]]
                [a2, b2] = res[0][feat[idx]]       
                if a2 < a:
                    a = max(a, b2)
                else:
                    b = min(b, a2)
                if (p < a) or (p > b):
                    continue            
                cube[feat[idx]] = [a, b]
                res = self.__checkOverlap({ len(hyperCubes) : cube }, hyperCubes)
                yyy = False
                break
            if yyy:
                print("Questo non dovrebbe succedere")
                for j, f in enumerate(feat):
                    cube[f] = [point[0][j], point[0][j]]
        hyperCubes[len(hyperCubes)] = cube
    
    def __predict(self, samples, hyperCubes):
        ret = []
        for s in samples:
            for hc in hyperCubes:
                found = True
                c = hyperCubes[hc]
                for i, f in enumerate(self.feat):
                    [a, b] = c[f]
                    v = s[i]
                    found &= (a <= v <= b)
                    if ~found:
                        break
                if found:
                    ret.append(c[self.target])
                    break
            if ~found:
                ret.append(np.nan)
        return ret    

    def __init__(self, target, name, ext, step, ratio, minUp, THRESHOLD, feat):
        print("ITER -", name, "data set")
        self.name = name
        self.ext = ext
        self.target = target
        self.ratio = ratio
        self.THRESHOLD = THRESHOLD
        Xtrain = load("datasets/train/x/{}.{}.joblib".format(name, ext))
        Xtest = load("datasets/test/x/{}.{}.joblib".format(name, ext))

        self.model = load_model("models/{}".format(name))
        yTrain = self.model.predict(Xtrain)
        yTest = self.model.predict(Xtest)
        self.yTest = yTest
        
        # feature names (input variables)        
        self.feat = feat
        #print("N. features: ", len(feat)) # j dimensions

        minmax = {} # surrounding cube

        # the surrounding cube include min and max on both train and test sets
        tr = np.array(Xtrain)
        self.tr = tr
        te = np.array(Xtest)
        self.te = te
        for i, c in enumerate(feat):
            minmax[c] = [min(tr[:, i].min(), te[:, i].min()), max(tr[:, i].max(), te[:, i].max())]

        self.minmax = minmax
        
        # minUpdate value for each feature    
        minUpdates = {}
        self.V = 1

        # default value, a fraction of the variable space (from min to max)
        for j in minmax:
            [a, b] = minmax[j]
            minUpdates[j] = (b - a) / minUp
            self.V *= (b - a)

        # initial number of rules 
        a = min(yTrain.min(), yTest.min())
        b = max(yTrain.max(), yTest.max())
        startPoints = np.arange(a, b, step)
        #print("N. rules: ", len(startPoints))

        # hyper-cubes are only points at the beginning
        hyperCubes = {}
        y = yTrain.copy()

        #print("Points creation")

        # each hyper-cube is the point nearest to the integer starting value
        for i, s in enumerate(startPoints):
            idx = np.where(abs(y - s) == min(abs(y - s)))
            cube = { target : y[idx][0] }
            for j, f in enumerate(feat):
                col = tr[:, j].reshape((-1, 1))
                cube[f] = [col[idx][0], col[idx][0]]
            hyperCubes[i] = cube    

        #print("Cubes creation") 

        # each hyper-cube is expanded from a point to a hyper-cube of side minUpdate    
        for hc in hyperCubes:
            cube = hyperCubes[hc]
            newCube = {}
            for f in cube:
                if f == target:
                    newCube[f] = cube[f]
                else:
                    [a, b] = cube[f] 
                    newCube[f] = [max(a - minUpdates[f] / ratio, minmax[f][0]), 
                                  min(b + minUpdates[f] / ratio, minmax[f][1])]
            hyperCubes[hc] = newCube

        # checking if there are overlapping cubes 
        if self.__checkOverlap(hyperCubes, hyperCubes) != None:
            print("There are overlappings")
            return

        # compute the mean predicted value for each hyper-cube using the training set
        for hc in hyperCubes:
            c = hyperCubes[hc]
            self.__updateMean(c, tr, feat)
            #print(self.__count(c, tr, feat)[0])
        
        fake = tr.tolist()
        limits = set()
        i = 1

        temp = tr.copy()
        temp_temp = []
    
        maxIter = 600

        while (len(temp) > 0) and (i < maxIter):
            while((len(limits) < len(feat) * 2 * len(hyperCubes)) and (i < maxIter)):        
                print("iteration", i, " " * 5, end = "\r")

                toUp = self.__iterate(hyperCubes, minUpdates, feat, fake, minmax, limits)       
            
                for hc in toUp:
                    self.__updateMean(hyperCubes[hc], np.array(fake), feat) 
                    
                i += 1

            for s in temp:
                if self.__predict([s], hyperCubes) == [np.nan]:
                    self.__cubeFromPoint(np.array([s]), hyperCubes, feat, minUpdates, np.array(fake))
                    break

            for s in temp:
                if self.__predict([s], hyperCubes) == [np.nan]:
                    temp_temp.append(s)
            
            temp = temp_temp.copy()
            temp_temp = []          

        print("iteration", i, ":", len(temp), "examples of", len(self.tr),
              "left with", len(hyperCubes), "hyper-cubes ({:.2f}%)".format(len(temp) / len(self.tr) * 100))
        self.hyperCubes = hyperCubes

        self.__checkV()
                
    def __checkV(self):
        tot = 0.
        self.vols = []
        for c in self.hyperCubes:
            s = True
            hc = self.hyperCubes[c]
            v = 1.
            for f in self.feat:
                [a, b] = hc[f]
                if (a != b):
                    v *= (b - a)
                else:
                    s = False
                    break
            self.vols.append(v / self.V)
            if s:
                tot += v
        print("Covered {:.2f}% of the surrounding cube".format(tot / self.V * 100))
    
    def metrics(self):
        ITER = np.array(self.__predict(self.te, self.hyperCubes))
        TRUE = load("datasets/test/y/{}.{}.joblib".format(self.name, self.ext)).values
        ANN = self.yTest.flatten()
        nan = np.count_nonzero(np.isnan(ITER))
        if nan > 0:
            print(nan, "outliers of", len(self.te), "test samples ({:.2f}%)".format(nan / len(self.te) * 100))
            idx = np.argwhere(~np.isnan(ITER))
            ITER = ITER[idx]
            TRUE = TRUE[idx]
            ANN = ANN[idx]

        print("MAE wrt data: {:.2f}, wrt ANN: {:.2f}, ANN MAE: {:.2f}".format(self.__mae(ITER, TRUE), self.__mae(ITER, ANN), self.__mae(ANN, TRUE))) 
        print("R2 wrt data: {:.2f}, wrt ANN: {:.2f}, ANN MAE: {:.2f}".format(self.__r2(ITER, TRUE), self.__r2(ITER, ANN), self.__r2(ANN, TRUE))) 
        print()
        
        n = []
        for h in self.hyperCubes:
            n.append(self.__count(self.hyperCubes[h], self.tr, self.feat)[0])
        return (n, self.vols)
    
    def __r2(self, pred, true):
        u = ((true - pred)**2).sum()
        v = ((true - true.mean())**2).sum()
        r2 = 1 - u / v
        return r2
        
    def __mae(self, pred, true):
        return abs(pred - true).mean()