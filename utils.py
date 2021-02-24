from Dataset import *
from Regressor import *
from ITER import *
from Gridex import *

def save(name, ext, target, sep = ",", dec = ".", rem = [], 
         hidden = [100], lr = 0.1, pat = 5, d = 1, l = "mse", 
         act = "tanh", e = 200, bsize = 100, verb = 0, ts = .2):
    ds = Dataset("{}.{}".format(name, ext), target, sep, dec, rem, testSize = ts)
    # Split the data
    Xtrain, Xvalid, yTrain, yValid = train_test_split(ds.Xtrain, ds.ytrain.values, 
                                                      test_size = 0.1, shuffle = True)

    model = Regressor(hidden = hidden, lr = lr, pat = pat, delta = d, loss = l, act = act)
    model.fit(Xtrain, yTrain, Xvalid, yValid,        
              ep = e, bs = bsize, v = verb)
    if verb > 0:
        model.plot()
    print("Metrics for", name, "data set:")
    model.metrics(ds.Xtest, ds.ytest.values)
    print("Saving model...")
    model.save(name)
    if verb > 0:
        return ds
    print()
        
def trainAndSave():        
    save("ARTI0", "csv", "arti1", rem = ["arti2", "arti3", "arti4"],
         hidden = [25, 5], lr = 0.1, l = "mae", 
         bsize = 150, ts = .5, d = .005, pat = 30)
    save("CCPP", "csv", "PE", ";", ",", hidden = [200], l = "mae")
    save("ASN", "dat", "Output", "\t", bsize = 50)
    save("EE", "csv", "Y1", ";", ",", ["Y2"], hidden = [200], l = "mae")
    save("GAS", "csv", "NOX", hidden = [200], l = "mse", bsize = 1000)
    save("WQ", "csv", "quality", ";", bsize = 250)
    
def testIter():
    step, ratio, minUp, THRESHOLD = 1, 2, 10, .2
    it = ITER("arti1", "ARTI0", "csv", 
              step, ratio, minUp, THRESHOLD, 
              ["x", "y"]).metrics()  
    
    step, ratio, minUp, THRESHOLD = 80, 2, 10, 7
    it = ITER("PE", "CCPP", "csv", 
              step, ratio, minUp, THRESHOLD, 
              ["AT", "V", "AP", "RH"]).metrics()
   
    step, ratio, minUp, THRESHOLD = 40, 2, 10, 4
    it = ITER("Output", "ASN", "dat", 
              step, ratio, minUp, THRESHOLD, 
              ["F (Hz)", "Angle", "Chord", "V", "Thickness"]).metrics()   
 
    step, ratio, minUp, THRESHOLD = 40, 2, 10, 4
    it = ITER("Y1", "EE", "csv", 
              step, ratio, minUp, THRESHOLD, 
              ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]).metrics()
    
    step, ratio, minUp, THRESHOLD = 100, 2, 10, 15
    it = ITER("NOX", "GAS", "csv", 
              step, ratio, minUp, THRESHOLD, 
              ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO"]).metrics()    
    
    step, ratio, minUp, THRESHOLD = 6, 2, 10, 2
    it = ITER("quality", "WQ", "csv", step, ratio, minUp, THRESHOLD, 
              ["fa", "va", "ca", "rs", "c", "fs", "ts", "d", "pH", "s", "a"]).metrics()    
    
def testGridex():  
    steps = [2]
    it = Gridex("arti1", "ARTI0", "csv", 
                ["x", "y"], steps, 0.01).metrics(False)
    
    steps = [2, 2]
    it = Gridex("PE", "CCPP", "csv", 
                ["AT", "V", "AP", "RH"], steps, 4.1, 
                [(0.04, 2), (0.5, 4)]).metrics(False)
    
    steps = [2, 2]
    it = Gridex("Output", "ASN", "dat", 
                ["F (Hz)", "Angle", "Chord", "V", "Thickness"], steps, 4, 
                [(0.2, 2), (0.5, 3), (0.7, 4)]).metrics(False)
    
    steps = [2]
    it = Gridex("Y1", "EE", "csv", 
                ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"], steps, 1, 
                [(0.001, 2), (0.5, 3)]).metrics(False) 
    
    steps = [2, 2]
    it = Gridex("NOX", "GAS", "csv", 
                ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO"], steps, 7, 
                [(0.1, 3), (0.7, 4)]).metrics(False) 
    
    steps = [2]
    it = Gridex("quality", "WQ", "csv", 
                ["fa", "va", "ca", "rs", "c", "fs", "ts", "d", "pH", "s", "a"], steps, 1, 
                [(0.03, 2), (0.6, 3)]).metrics(False)    