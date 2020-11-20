import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense
from time import time
#tf.random.set_seed(1)

class FeedForward():
    __models = []
    __evalution_metrics = []


    def append_model(self,model):
        self.__models.append(model)

    def append_params(self, params):
        self.__evalution_metrics.append(params)

    def print_params(self, index = -1):
        print(self.__evalution_metrics[index:])

class evaluate_models(FeedForward):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def evaluate(self, X_train, X_test, y_train, y_test):
        scores = [[] for i in range(4)] # note, 4 --> length of metrics!
        start = time()
        for j in range(10):
            self.model.fit(X_train, y_train, epochs=50, verbose=0)
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