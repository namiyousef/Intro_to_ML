import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense
tf.random.set_seed(1)


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