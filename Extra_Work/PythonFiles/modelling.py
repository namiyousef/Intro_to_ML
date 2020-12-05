import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense
from time import time
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
#tf.random.set_seed(1)

class FeedForward():
    __models = []
    __evalution_metrics = []


    def append_model(self,model):
        self.__models.append(model)

    def append_params(self, params):
        self.__evalution_metrics.append(params)

    def get_model(self, index = -1):
        return self.__models[index]

    def print_params(self, index = -1):
        print(self.__evalution_metrics[index:])

    def save_model(self): # TODO change name of this !
        plot_model(self.__models[-1])

    def plot_model(self):
        history = self.__models[-1]
        plt.plot(history.history['accuracy'])
        plt.show()
        plt.plot(history.history['loss'])
        plt.show()

class evaluate_models(FeedForward):
    def __init__(self, model, compiler, X_train, X_test, y_train, y_test):
        super().__init__()
        self.uncompiled_model = model
        self.compiler = compiler
        self.model = compiler(model)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self,*args, **kwargs):
        history = self.model.fit(
            self.X_train if not args else args[0],
            self.y_train if not args else args[1],
            **kwargs)
        return history

    def compile(self):
        self.model = self.compiler(self.model)

    def cross_validate(self, K = 5, **kwargs):
        scores = []
        histories = []
        X = np.concatenate((self.X_train, self.X_test), axis = 0)
        y = np.concatenate((self.y_train, self.y_test), axis = 0)
        for train, test in KFold(n_splits=K, shuffle=True).split(X,y):
            self.compile()
            histories.append(self.fit(X[train], y[train], **kwargs).history)
            scores.append(self.model.evaluate(X[test], y[test], verbose = 1))
        print("average loss: ", np.asarray(scores)[:,0].mean())
        print("average accuracy: ", np.asarray(scores)[:,1].mean()) # make sure that accuracy is the first metric in compile
        return scores, histories

    def plot_histories(self, histories, metrics = ['loss', 'accuracy']):
        fig, axes = plt.subplots(nrows = len(metrics) % 2 + 1, ncols = 2)
        axes = axes.reshape(len(metrics) % 2 + 1, 2)
        for i,metric in enumerate(metrics):
            for history in histories:
                axes[(i+2)//2 - 1, 1 - (i+1)%2].plot(history[metric])
            plt.legend([i for i in range(len(histories))])


    def evaluate(self, **kwargs):
        history_temp = self.fit()
        keys = history_temp.history.keys()
        scores = [[] for i in range(4)]
        start = time()
        for j in range(1):
            self.fit(self.X_train, self.y_train, **kwargs)
            score = self.model.evaluate(self.X_test, self.y_test, verbose=1)
            for i in range(4):
                scores[i].append(score[i])
        scores = [sum(s)/len(s) for s in scores]
        scores.append(time() - start)
        scores.append(self.model.count_params())

        self.append_model(self.model)
        self.append_params(scores)

def uncompiled_model():
    inputs = Input(shape=(6,), name='Data')
    x = Dense(4, activation='relu', name='dense_1')(inputs)
    x = BatchNormalization()(x)
    x = Dense(4, activation='relu', name='dense_2')(x)

    outputs = Dense(2, activation='softmax', name='TargetHit')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def compile_model():
    model = uncompiled_model()
    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 "binary_accuracy",
                 "binary_crossentropy",
                 "categorical_accuracy"
                 ],
    )
    return model

def cross_validate(model, *data, K = 5, **kwargs):

    """" design this such that the model will be reconfigured automatically
    each time! Make it robust so that it can include neural networks as well!
    Also learn how to do GridSearch!!
    """
    if len(data) == 2:
        X = data[0]
        y = data[1]
    elif len(data) == 4:
        X = np.concatenate((data[0], data[1]), axis = 0)
        y = np.concatenate((data[2], data[3]), axis = 0)
    for train, test in KFold(n_splits=K, shuffle=True).split(X, y):

        histories.append(self.fit(X[train], y[train], **kwargs).history)
        scores.append(self.model.evaluate(X[test], y[test], verbose=1))