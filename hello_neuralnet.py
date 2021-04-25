
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##
## Guide: https://www.tensorflow.org/guide/keras/sequential_model
##
def neuralnet_demo1():
    # Define Sequential model with 3 layers (input shape tensor tem que ter dim=5 no ultimo eixo)
    meumodelo1 = keras.Sequential(
        [
         layers.Input(shape=(5,)),
         layers.Dense(2, activation="relu", name="camada1"),
         layers.Dense(3, activation="relu", name="camada2"),
         layers.Dense(4, name="camada3"),
        ],
        name="exemplo de rede simples 2x3x4"
    )
    print("Sumario1: ", meumodelo1.summary())

    # Call model on different test inputs
    x = tf.ones((1, 5))
    y = meumodelo1(x)
    print(y)

    x = tf.ones((12, 5))
    y = meumodelo1(x)
    print(y)

    x = tf.ones((7, 3, 5))
    y = meumodelo1(x)
    print(y)

    print("Sumario2: ", meumodelo1.summary())
    print("Pesos: ",meumodelo1.weights)

#
# https://www.tensorflow.org/tutorials/keras/classification
#
def neuralnet_Flatten_demo1():
    meulayer = layers.Flatten(input_shape=(2, 3, 4, 5))
    x = tf.ones((2, 3, 4, 5))
    print(x)
    y = meulayer(x)
    print(y)

def conv2d_Demo1():
    # The inputs are 28x28 RGB images with `channels_last` as default and the batch
    # size is always the first value (1).

    # example 1
    #input_shape = (1, 5, 4, 1)
    #x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], shape=input_shape, dtype=float)
    # example 2:
    input_shape = (1, 5, 4, 1)
    x = tf.constant(1, shape=input_shape, dtype=float)

    #print(x)
    print(x[0,:,:,0])
    # not using initial random values
    conv_func = tf.keras.layers.Conv2D(kernel_size=[2,3],filters=1,kernel_initializer=tf.constant_initializer(1.), input_shape=input_shape[1:], activation=None, use_bias=False)
    conv_func.build(input_shape[1:])
    #print('weigths= ',conv_func.get_weights()) #converse function is conv_func.set_weights()
    print('weigths mat= ', conv_func.get_weights()[0])
    print('weigths mat shape= ',conv_func.get_weights()[0].shape)
    y = conv_func(x)
    #print('y.shape= ', y.shape)
    print('y mat= ',y[0])

if __name__ == '__main__':
    #neuralnet_Flatten_demo1()
    #conv2d_Demo1()
    neuralnet_demo1()
