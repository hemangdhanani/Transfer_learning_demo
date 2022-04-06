import argparse
import os
import shutil
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np


STAGE = "Creating base model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)    
    
    ## get data
    data_mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = data_mnist.load_data()
    X_test = X_test/255.
    X_cv, X_train = X_train[:5000]/255., X_train[5000:]/255.
    y_cv, y_train = y_train[:5000], y_train[5000:]

    ## set the seeds
    seed =2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # define layers
    
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
        ] 

    # define the model and compile

    model = tf.keras.models.Sequential(LAYERS)

    loss_func = "sparse_categorical_crossentropy"
    optimizer_val = tf.keras.optimizers.SGD(learning_rate=1e-3)
    matrics_method = ["accuracy"]

    model.compile(optimizer=optimizer_val, loss=loss_func,metrics=matrics_method)

    model.summary()

    ## Train the model 
    epochs_num = 10
    Validation = (X_cv, y_cv)
    history = model.fit(
        X_train, y_train,
        epochs=epochs_num,
        validation_data=Validation,
        verbose=2)

    ## save the model
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation matrics {model.evaluate(X_test, y_test)}")             


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e