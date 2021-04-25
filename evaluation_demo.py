import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime

def tf_eval_demo1():
    print("Evaluation tf ' mnist dataset demo: uses train and uses test set as validation set")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    # Reserve 10,000 samples for validation (the same as the dataset test set)
    x_val = x_test
    y_val = y_test
    meumodelo = get_compiled_model4_mnist_dataset() # cria o modelo
    print("Fit model on training data")
    history = meumodelo.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        # Validation data (optional) for monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )
    print("History: ", history.history)

def tf_eval_demo2():
    print("Evaluation tf ' mnist dataset demo: uses train and test set")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    meumodelo = get_compiled_model4_mnist_dataset()
    print("Fit model on training data")
    mybatch_size = 64
    epocas = 1
    history = meumodelo.fit(
        x_train,
        y_train,
        batch_size=mybatch_size,
        epochs=epocas,
    )
    print("History: ", history.history)
    print("Evaluate on test data")
    results = meumodelo.evaluate(x_test, y_test, batch_size=mybatch_size)
    print("test loss, test acc:", results)

def tf_eval_demo3():
    print("Evaluation tf ' mnist dataset demo: uses train and test set and confusion matrix")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    meumodelo = get_compiled_model4_mnist_dataset()
    print("Fit model on training data")
    mybatch_size = 64
    epocas = 1
    meumodelo.fit(
        x_train,
        y_train,
        batch_size=mybatch_size,
        epochs=epocas,
    )

    print("Evaluate on test data")
    results = meumodelo.evaluate(x_test, y_test, batch_size=mybatch_size)
    print("test loss, test acc:", results)

    dim_prediction_set = 5000
    num_classes_mnist = 10
    print("Generate soft predictions for %d samples for the mnist %d class dataset:" % (dim_prediction_set, num_classes_mnist))
    y_predictions_scores = meumodelo.predict(x_test[:dim_prediction_set]) # para as dim_prediction_set primeiras instancias de teste
    y_predictions = tf.reshape(tf.argmax(y_predictions_scores, axis=1), shape=(-1, 1))
    ground_truth = y_test[:dim_prediction_set]
    conf_mat = tf.math.confusion_matrix(ground_truth, y_predictions, num_classes=num_classes_mnist)
    print("Confusion matrix", conf_mat)

def tf_eval_demo4():
    print("Evaluation tf ' mnist dataset demo: uses train and uses test set as validation set and tensorboard")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    # Reserve 10,000 samples for validation (the same as the dataset test set)
    x_val = x_test
    y_val = y_test
    meumodelo = get_compiled_model4_mnist_dataset() # cria o modelo
    print("Fit model on training data")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = meumodelo.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        # Validation data (optional) for monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback]
    )
    print("History: ", history.history)

def get_uncompiled_model4_mnist_dataset():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model4_mnist_dataset():
    model = get_uncompiled_model4_mnist_dataset()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    #keras.utils.plot_model(model, "meumodelo1.png", show_shapes=True)
    return model

if __name__ == '__main__':
    #tf_eval_demo1()  # train + validation set
    #tf_eval_demo2()  # train + test set
    tf_eval_demo3()  # confusion matrix
    #tf_eval_demo4() #tensorboard

