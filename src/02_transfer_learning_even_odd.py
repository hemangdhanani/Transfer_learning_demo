import argparse
import os
import shutil
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np
import io


STAGE = "transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        list_of_labels[idx] = np.where(label%2 == 0, 1, 0)
    return list_of_labels    


def main(config_path):
    ## read config files
    config = read_yaml(config_path)    
    
    ## get data
    data_mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = data_mnist.load_data()
    X_test = X_test/255.
    X_cv, X_train = X_train[:5000]/255., X_train[5000:]/255.
    y_cv, y_train = y_train[:5000], y_train[5000:]

    y_train_bin, y_test_bin, y_cv_bin = update_even_odd_labels([y_train, y_test, y_cv])

    ## set the seeds
    seed =2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #load the base model
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)

    
    # freeze the weights
    for layer in base_model.layers[:-1]:
        print(f"trainable status of before  {layer.name} : {layer.trainable}")
        layer.trainable = False
        print(f"trainable status of after  {layer.name} : {layer.trainable}")


    base_layer = base_model.layers[:-1]    

    # # define the model and compile

    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(2, activation='softmax', name = "output_Layer")
    )

    new_model.summary()

    loss_func = "sparse_categorical_crossentropy"
    optimizer_val = tf.keras.optimizers.SGD(learning_rate=1e-3)
    matrics_method = ["accuracy"]

    new_model.compile(optimizer=optimizer_val, loss=loss_func,metrics=matrics_method)    

    ## Train the model 
    epochs_num = 10
    Validation = (X_cv, y_cv_bin)
    history = new_model.fit(
        X_train, y_train_bin,
        epochs=epochs_num,
        validation_data=Validation,
        verbose=2)

    ## save the model
    model_dir_path = os.path.join("artifacts","models") 
    model_file_path = os.path.join(model_dir_path, "even_odd_model.h5")
    new_model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation matrics {new_model.evaluate(X_test, y_test_bin)}")             


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