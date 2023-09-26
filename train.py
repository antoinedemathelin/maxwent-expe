import os
import shutil
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy
from joblib import Parallel, delayed
from networks import DenseNet
from utils import results_regression, results_classification, GaussianNegativeLogLikelihood, entropy, SaveModel
from methods import AnchoredNetwork, BaseEnsemble, DeepEnsemble, MOD, RDE, NegativeCorrelation, BNN, MCDropout, MaxWEnt, MaxWEntSVD
from datasets import reg1d, classif2d, citycam_weather, citycam_bigbus, citycam_cameras, load_uci

CLASSIFICATION_DATASET = ["2d_classif"]

base_dict = {
    "AnchoredNetwork": AnchoredNetwork,
    "BaseEnsemble": BaseEnsemble,
    "DeepEnsemble": DeepEnsemble,
    "MOD": MOD,
    "RDE": RDE,
    "NegativeCorrelation": NegativeCorrelation,
    "BNN": BNN,
    "MCDropout": MCDropout,
    "MaxWEnt": MaxWEnt,
    "MaxWEntSVD": MaxWEntSVD
}

network_dict = {
    "DenseNet": DenseNet
}

optimizer_dict = {
    "SGD": SGD,
    "Adam": Adam
}

loss_dict = {
    "SparseCategoricalCrossentropy": SparseCategoricalCrossentropy(from_logits=True),
    "MeanSquaredError": MeanSquaredError(),
    "BinaryCrossentropy": BinaryCrossentropy(from_logits=True),
    "GaussianNegativeLogLikelihood": GaussianNegativeLogLikelihood()
}

dataset_dict = {
    "2d_classif": classif2d,
    "1d_reg": reg1d,
    "citycam_weather": citycam_weather,
    "citycam_cameras": citycam_cameras,
    "citycam_bigbus": citycam_bigbus,
    'yacht_extrapol': load_uci(dataset='yacht', setting='extrapol'),
    'energy_extrapol': load_uci(dataset='energy', setting='extrapol'),
    'concrete_extrapol': load_uci(dataset='concrete', setting='extrapol'),
    'wine_extrapol': load_uci(dataset='wine', setting='extrapol'),
    'power_extrapol': load_uci(dataset='power', setting='extrapol'),
    'naval_extrapol': load_uci(dataset='naval', setting='extrapol'),
    'protein_extrapol': load_uci(dataset='protein', setting='extrapol'),
    'kin8nm_extrapol': load_uci(dataset='kin8nm', setting='extrapol'),
    'yacht_interpol': load_uci(dataset='yacht', setting='interpol'),
    'energy_interpol': load_uci(dataset='energy', setting='interpol'),
    'concrete_interpol': load_uci(dataset='concrete', setting='interpol'),
    'wine_interpol': load_uci(dataset='wine', setting='interpol'),
    'power_interpol': load_uci(dataset='power', setting='interpol'),
    'naval_interpol': load_uci(dataset='naval', setting='interpol'),
    'protein_interpol': load_uci(dataset='protein', setting='interpol'),
    'kin8nm_interpol': load_uci(dataset='kin8nm', setting='interpol'),
}

score_dict = {
    "entropy": entropy,
}

callbacks_dict = {
    "SaveModel": SaveModel,
}


def build_model(config, num=None):
    
    metrics = config.get("metrics", None)
    optimizer = config.get("optimizer", "SGD")
    schedule = config.get("schedule", None)
    
    n_estimators = config.get("n_estimators", 1)
    method = config.get("method")
    network = config.get("network")
    loss = config.get("loss")
    
    optimizer_params = config.get("optimizer_params", {})
    network_params = config.get("network_params", {})
    schedule_params = config.get("schedule_params", {})
    method_params = config.get("params", {})
    pretrained_weights = config.get("pretrained_weights", None)
    
    if schedule is not None:
        schedule = schedule_dict[schedule](**schedule_params)
        optimizer_params.pop("lr", None)
        optimizer_params.pop("learning_rate", None)
        optimizer = optimizer_dict[optimizer](schedule, **optimizer_params)
    else:
        optimizer = optimizer_dict[optimizer](**optimizer_params)
        
    loss = loss_dict[loss]
        
    if BaseEnsemble in base_dict[method].__bases__: 
        method_params["n_estimators"] = n_estimators
        model = base_dict[method](network_dict[network], **network_params, **method_params)
    else:
        network = network_dict[network](**network_params)
        if pretrained_weights is not None and num is not None:
            pretrained_weights = os.path.join(pretrained_weights, "net_%i.hdf5"%num)
            network.load_weights(pretrained_weights)
        model = base_dict[method](network, **method_params)
        
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def fit_network(num, state, data, config):
    np.random.seed(state)
    tf.random.set_seed(state)
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    save_path = config.get("save_path", "")
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    steps_per_epoch = config.get("steps_per_epoch", 1)
    verbose = config.get("verbose", 1)
    callbacks = config.get("callbacks", None)
    callbacks_params = config.get("callbacks_params", {})
    state = config.get("state", 0)
    method = config.get("method")
    rescale_output = config.get("rescale_output", False)
    
    copy_config = copy.deepcopy(config)
    if method == "MaxWEntSVD":
        copy_config["params"]["X_train"] = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        
    if method == "BNN":
        copy_config["params"]["num_data"] = x.shape[0]
    
    model = build_model(copy_config, num)
    
    train_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x),
                                    tf.data.Dataset.from_tensor_slices(y)))
    val_ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(xval),
                                  tf.data.Dataset.from_tensor_slices(yval)))
    repeat_train = int(epochs * steps_per_epoch * batch_size / x.shape[0]) + 1
    
    train_ds = train_ds.repeat(repeat_train).shuffle(10000).batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    
    callbaks_list = []
    if callbacks is not None:
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
            callbacks_params = [callbacks_params]
        for callback, params in zip(callbacks, callbacks_params):
            if callback == "SaveModel":
                callbaks_list.append(callbacks_dict[callback](validation_data=val_ds, **params))
            else:
                callbaks_list.append(callbacks_dict[callback](**params))
    
    model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
              callbacks=callbaks_list)
    
    if num is None:
        save_path_net = save_path
    else:
        save_path_net = os.path.join(save_path, "net_%i.hdf5"%num)
    
    model.save_weights(save_path_net)


def train(config):
    
    for k, v in config.items():
        print(k+": ", v)
    
    state = config.get("state", 0)
    dataset = config.get("dataset")
    method = config.get("method")
    n_jobs = config.get("n_jobs", None)
    n_estimators = config.get("n_estimators", 1)
    params = config.get("params", {})
    save_path = config.get("save_path", "")
    network = config.get("network")
    network_params = config.get("network_params", {})
    batch_size = config.get("batch_size", 1)
    ood_score_func = config.get("ood_score_func", None)
    rescale_output = config.get("rescale_output", False)
    val_score = config.get("val_score", "loss")
    
    data = dataset_dict[dataset](state=state)
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    input_shape = data[0].shape[1:]
    config["network_params"]["input_shape"] = input_shape
    if rescale_output:
        config["network_params"]["scale"] = [y.std(), 1.]
        config["network_params"]["offset"] = [y.mean(), 0.]
    
    if not isinstance(params, list):
        params = [params]
    
    np.random.seed(state)
    tf.random.set_seed(state)
    states = np.random.choice(2**20, n_estimators)
    
    best_score = np.inf
    for p in params:
        config_p = copy.deepcopy(config)
        config_p["params"] = p
        config_p["save_path"] = "temp_model"
        os.makedirs("temp_model", exist_ok=True)
        if not BaseEnsemble in base_dict[method].__bases__:
            if n_jobs is None:
                for s, num in zip(states, range(n_estimators)):
                    fit_network(num, s, data, config_p)
            else:
                Parallel(n_jobs=n_jobs)(delayed(fit_network)(num, s, data, config_p)
                                        for s, num in zip(states, range(n_estimators)))
        else:
            fit_network(None, states[0], data, config_p)
        
        if BaseEnsemble in base_dict[method].__bases__: 
            model = build_model(config_p)
        else:
            model = BaseEnsemble(build_model, config=config_p, n_estimators=n_estimators)
        model.load_weights("temp_model")
        shutil.rmtree("temp_model")
        
        model.compile(loss=loss_dict[config.get("loss")])
        
        # Log weight
        if val_score == "weights":
            weight_list = []
            for i in range(model.n_estimators):
                weight_loss = 0.
                count = 0.
                train_vars = getattr(model, "network_%i"%i).trainable_variables 
                for j in range(len(train_vars)):
                    w = train_vars[j]
                    w = tf.math.log(1. + tf.exp(w))
                    weight_loss += tf.reduce_sum(w)
                    count += tf.reduce_sum(tf.ones_like(w))
                weight_loss /= count
                weight_list.append(weight_loss.numpy())
            config["weights_%s"%str(p)] = weight_list
            score = -np.mean(weight_list)
        else:
            score = model.evaluate(xval, yval)
        
        # Save best model
        print(method, str(p), "Score: %.4f"%score)
        if score <= best_score:
            model.save_weights(save_path)
            best_score = score
            config["best_params"] = p
            config["best_score"] = score
        
        # Log results for params = p
        if dataset in CLASSIFICATION_DATASET:
            scores = results_classification(model, data)
        else:
            scores = results_regression(model, data)
        scores = {k+"_%s"%str(p): v for k, v in scores.items()}
        config.update(scores)
        
    config_p = copy.deepcopy(config)
    config_p["params"] = config["best_params"]
    
    if BaseEnsemble in base_dict[method].__bases__: 
        model = build_model(config_p)
    else:
        model = BaseEnsemble(build_model, config=config_p, n_estimators=n_estimators)
        
    model.load_weights(save_path)
    model.compile(loss=loss_dict[config.get("loss")])
    
    if dataset in CLASSIFICATION_DATASET:
        scores = results_classification(model, data)
    else:
        scores = results_regression(model, data)
            
    logs_save_path = os.path.join("logs", save_path + ".csv")
    os.makedirs(os.path.dirname(logs_save_path), exist_ok=True)
    config.update(scores)
    pd.Series(config).to_csv(logs_save_path)