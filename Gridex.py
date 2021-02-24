import numpy as np
from itertools import product
from tensorflow.keras.models import load_model
from sklearn.feature_selection import SelectKBest, f_regression
from joblib import load
import random as rnd

class Gridex:   
    def __init__(self, target, name, ext, feat, steps, th, adap = None):
        print("GridEx -", name, "data set")
        self.name = name
        self.ext = ext
        self.feat = feat
        self.target = target
        self.steps = steps
        self.threshold = th
        self.model = load_model("models/{}".format(name))
        self.fake = load("datasets/train/x/{}.{}.joblib".format(name, ext))
        self.Xtrain = np.array(self.fake)
        self.Xtest = np.array(load("datasets/test/x/{}.{}.joblib".format(name, ext)))       
        self.__adaptiveSplits(adap)
        self.__createSurrounding()
        self.__iterate()               
        
    def __count(self, c, samples, mean = False):
        cond = np.ones((len(samples),), dtype = bool)
        for i, f in enumerate(self.feat):
            [a, b] = c[f]
            col = samples[:, i]
            cond &= ((a <= col) & (col <= b))
        n = len(np.nonzero(cond)[0])
        if mean:
            if n > 0:
                pred = self.model.predict(samples[cond])
                return n, samples[cond].tolist(), pred.mean(), pred.std()
            else:
                return n, samples[cond].tolist(), 0, 0
        else:
            return n, samples[cond].tolist()
    
    def __predict(self, samples):
        ret = []
        for s in samples:
            for hc in self.hyperCubes:
                found = True
                c = self.hyperCubes[hc]
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
    
    def __createSurrounding(self):
        self.minmax = { "std" : 2 * self.threshold, self.target : 0 } # surrounding cube  
        for i, c in enumerate(self.feat):
            mi = min(self.Xtrain[:, i].min(), self.Xtest[:, i].min())
            ma = max(self.Xtrain[:, i].max(), self.Xtest[:, i].max())
            eps = 1e-5
            self.minmax[c] = [mi - eps, ma + eps]
            
        self.V = 1.
        for f in self.feat:
            [a, b] = self.minmax[f]
            self.V *= (b - a)
            
    def __iterate(self):
        prev = { 0 : self.minmax }
        tot = 0

        for step in self.steps:
            self.hyperCubes = {}
            for c in prev:
                self.split = {}
                if self.__count(prev[c], self.Xtrain)[0] == 0:
                    continue

                if prev[c]["std"] < self.threshold:
                    self.hyperCubes[len(self.hyperCubes)] = prev[c]                    
                    continue   
                ranges = {}
                
                for (f, imp) in zip(self.feat, self.scores):
                    r = []
                    [a, b] = prev[c][f]
                    if self.adap is not None:
                        step = self.adap[f]
                    s = (b - a) / step
                    for i in range(step):
                        r.append([a + s * i, a + s * (i + 1)])
                    ranges[f] = r            

                prod = list(product(*ranges.values()))  
                tot += len(prod)

                for (pn, p) in enumerate(prod):
                    print("{:.2f}%".format(pn / len(prod) * 100), end = "\r")
                    cube = { self.target : 0 }
                    for i, f in enumerate(self.feat):
                        cube[f] = p[i]
                    n, s, m, std = self.__count(cube, self.Xtrain, True)
                    self.__produceFake(cube, n)
                    nn, s, m, std = self.__count(cube, np.array(self.fake), True)
                    if n > 0:
                        cube[self.target] = m
                        cube["std"] = std
                        cube["n"] = n
                        if std > self.threshold:
                            self.hyperCubes[len(self.hyperCubes)] = cube
                        else:
                            self.split[len(self.split)] = cube

                co = 0
                to = len(self.split)
                self.oldAdj = {}
                self.oldMer = {}
                self.last = [i for i in self.split]
                while(self.__merge()):
                    co += 1
                    print("merged", co, "of", to, " " * 20, end = "\r")
                for res in self.split:
                    n, s = self.__count(self.split[res], self.Xtrain, False)
                    self.hyperCubes[len(self.hyperCubes)] = self.split[res]                       

            print("Useful hyper-cubes:", len(self.hyperCubes), "of", tot)
            self.__checkV()
            prev = self.hyperCubes.copy()   
            self.metrics()
            print()            
            
    def __merge(self):
        ret = False
        checked = []
        self.temp = []
        for i in self.split:
            checked.append(i)
            for j in self.split:
                if j not in checked:
                    if (i in self.last) or (j in self.last):
                        adj = self.__adjacent(self.split[i], self.split[j]) 
                    else:
                        adj = self.oldAdj[(i, j)]
                    if adj is not None:
                        self.temp.append((i, j, adj))                               
                    self.oldAdj[(i, j)] = adj
        merged = []    
        for (i, j, adj) in self.temp:
            if (i in self.last) or (j in self.last):
                t = self.__tempCube(i, j, adj)
                self.oldMer[(i, j)] = t
            else:
                t = self.oldMer[(i, j)]
            if t is not None:
                merged.append(t)
        if(len(merged) > 0):
            std, c1, c2, mi = min(merged)
            del self.split[c1]
            del self.split[c2]
            self.last = [c1, c2]
            self.split[c1] = mi
            ret = True
        return ret
                    
    def __tempCube(self, i, j, f):
        c1 = self.split[i]
        c2 = self.split[j]
        cube = {}
        for k in self.feat:
            if k != f:
                cube[k] = c1[k]
            else:
                [a1, b1] = c1[f]
                [a2, b2] = c2[f]
                cube[f] = [min(a1, a2), max(b1, b2)]
        n, s, m, std = self.__count(cube, np.array(self.fake), True)
        cube[self.target] = m
        cube["std"] = std
        cube["n"] = n 
        if std < self.threshold:
            return (std, i, j, cube)
        else:
            return None
    
    def __adjacent(self, c1, c2):
        adj = None        
        for f in self.feat:
            if c1[f] == c2[f]:
                continue   
            if adj is not None:
                return None 
            [a1, b1] = c1[f]
            [a2, b2] = c2[f]
            if (b1 == a2) or (b2 == a1):
                adj = f
            else:
                return None
        return adj

    def __produceFake(self, cube, n):
        for i in range(n, 15):
            sample = []
            for f in self.feat:
                [a, b] = self.minmax[f]
                sample.append(rnd.uniform(a, b))
            self.fake.append(sample)           
        
    def __adaptiveSplits(self, adap):
        fs = SelectKBest(score_func = f_regression, k = "all")
        fit = fs.fit(self.Xtrain, self.model.predict(self.Xtrain).flatten())
        self.scores = np.array(fit.scores_) / max(fit.scores_)
        #print(self.scores)
        
        self.adap = {}
        if adap is not None:
            for (f, imp) in zip(self.feat, self.scores):
                step = 1
                for (l, s) in adap:
                    if imp > l:
                        step = s
                    else:
                        break
                self.adap[f] = step
        else:
            self.adap = None   
        #print(self.adap)        
        
    def __volume(self, hc):
        v = 1.
        for f in self.feat:
            [a, b] = hc[f]
            v *= (b - a)   
        return v
        
    def __checkV(self):
        tot = 0.
        self.vols = []
        for c in self.hyperCubes:
            hc = self.hyperCubes[c]
            v = self.__volume(hc)
            self.vols.append(v / self.V)
            tot += v
        print("Covered {:.2f}% of the surrounding cube".format(tot / self.V * 100))
    
    def metrics(self, p = True):
        ITER = np.array(self.__predict(self.Xtest))
        TRUE = load("datasets/test/y/{}.{}.joblib".format(self.name, self.ext)).values
        ANN = self.model.predict(self.Xtest).flatten()
        nan = np.count_nonzero(np.isnan(ITER))
        if nan > 0:
            if p:
                print(nan, "outliers of", len(self.Xtest), "test samples ({:.2f}%)".format(nan / len(self.Xtest) * 100))
            idx = np.argwhere(~np.isnan(ITER))
            ITER = ITER[idx]
            TRUE = TRUE[idx]
            ANN = ANN[idx]

        if p:
            print("MAE wrt data: {:.2f}, wrt ANN: {:.2f}, ANN MAE: {:.2f}".format(self.__mae(ITER, TRUE), self.__mae(ITER, ANN), self.__mae(ANN, TRUE))) 
            print("R2 wrt data: {:.2f}, wrt ANN: {:.2f}, ANN MAE: {:.2f}".format(self.__r2(ITER, TRUE), self.__r2(ITER, ANN), self.__r2(ANN, TRUE)))    
            print()
        
        n = []
        for h in self.hyperCubes:
            n.append(self.__count(self.hyperCubes[h], self.Xtrain, self.feat)[0])
        return (n, self.vols)
    
    def __r2(self, pred, true):
        u = ((true - pred)**2).sum()
        v = ((true - true.mean())**2).sum()
        r2 = 1 - u / v
        return r2
        
    def __mae(self, pred, true):
        return abs(pred - true).mean()