#!/usr/bin/env python

# import agents

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


def huber_loss_function(clip_value):
    def __huber_loss(y_true, y_pred):
        ''' https://en.wikipedia.org/wiki/Huber_loss '''
        assert clip_value > 0.

        x = y_true - y_pred
        squared_loss = 0.5 * K.square(x)

        # nothing but check
        if np.isinf(clip_value):
            return K.mean(squared_loss)

        condition = K.abs(x) < clip_value
        linear_loss = clip_value * (K.abs(x) - 0.5 * clip_value)

        loss = tf.where(condition, squared_loss, linear_loss)  # condition, true, false
        # return K.mean(loss, axis=-1)
        return K.mean(loss)

    return __huber_loss


# huber_loss_function(np.inf) is equal to mean_squared_loss

def mean_squared_loss(y_true, y_pred):
    x = y_true - y_pred
    squared_loss = 0.5 * K.square(x)
    return K.mean(loss)
    # return K.mean(0.5 * K.square(y_true - y_pred) )


def test():
    shape = (6, 7)
    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    loss = huber_loss_function(clip_value=1.0)

    out1 = K.eval(loss(K.variable(y_a), K.variable(y_b)))
    out2 = K.eval(loss(y_a, y_b))

    print(out1)
    print(out2)
    pass


if __name__ == "__main__":
    test()

    print("passed")
    input("Enter to exit")
