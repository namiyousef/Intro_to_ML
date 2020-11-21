import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense
from time import time
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
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
    def __init__(self, model):
        super().__init__()
        self.model = model

    def cross_validate(self,X_train, X_test, y_train, y_test, K = 5):
        acc_per_fold = []
        loss_per_fold = []
        X = np.concatenate((X_train, X_test), axis = 0)
        y = np.concatenate((y_train, y_test), axis = 0)
        for train, test in KFold(n_splits=K, shuffle=True).split(X,y):
            self.model.fit(X[train], y[train], epochs = 250, verbose = 0)
            scores = self.model.evaluate(X[test], y[test], verbose = 1)
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])
        print(sum(acc_per_fold)/len(acc_per_fold),sum(loss_per_fold)/len(loss_per_fold))


    def evaluate(self, X_train, X_test, y_train, y_test):
        scores = [[] for i in range(4)] # note, 4 --> length of metrics!
        start = time()
        for j in range(1):
            self.model.fit(X_train, y_train, epochs=1000, verbose=1)
            score = self.model.evaluate(X_test, y_test, verbose=0)
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

instance = evaluate_models(
    compile_model()
)