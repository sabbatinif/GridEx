import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras import backend as K
from pathlib import Path
tf.keras.backend.set_floatx('float64')

class Regressor:
    def __init__(self, hidden = [100], lr = 0.01, pat = 5, delta = 5, loss = "mae", act = "tanh"):
        self.__build(hidden, lr, loss, delta, pat, act)
        self.loss = loss

    def __build(self, hidden, lr, loss, delta, pat, act):
        layers = [Dense(h, kernel_initializer = 'uniform', activation = act) for h in hidden] + \
                 [Dense(1, kernel_initializer = 'uniform')]

        self.model = Sequential(layers)

        opt = Adam(learning_rate = lr)
        self.model.compile(loss = loss, metrics = ['mae', 'mse'], optimizer = opt)
        
        early_stop = EarlyStopping(
            monitor = 'val_loss', 
            min_delta = delta, 
            patience = pat, 
            restore_best_weights = True, 
            verbose = 0
        )

        self.callbacks = [early_stop]
        

    def fit(self, Xt, yt, Xv, yv, bs = 1500, ep = 100, v = 0):
        self.history = self.model.fit(Xt, yt,  
                                      validation_data = (Xv, yv),                        
                                      epochs = ep, batch_size = bs, 
                                      verbose = v, callbacks = self.callbacks)

    def plot(self):
        plt.plot(self.history.history[self.loss], label = "train")
        plt.plot(self.history.history["val_{}".format(self.loss)], label = "valid")
        plt.title(self.loss)
        plt.ylabel("Error")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()
        
    def metrics(self, pred, true):
        pred = self.model.predict(pred).flatten()
        u = sum((true - pred)**2)
        v = sum((true - true.mean())**2)
        r2 = 1 - u / v
        
        mae = (abs(true - pred)).mean()
        mse = ((abs(true - pred))**2).mean()
        print("R2 = {:.2f}, MAE = {:.2f}, MSE = {:.2f}".format(r2, mae, mse))
        
    def save(self, name):
        Path("models").mkdir(parents = True, exist_ok = True)
        self.model.save("models/{}".format(name))