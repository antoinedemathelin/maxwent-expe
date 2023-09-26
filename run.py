import os
import copy
import time
import yaml
import ast
import pandas as pd
from train import train
from train import base_dict, dataset_dict, CLASSIFICATION_DATASET
import sys, getopt


def read_args(argv):
    args_dict_default = dict(
        configfile = None,
        dataset = None,
        method = None,
        params = {},
        savedir = ".",
        name = "",
        n_runs = 1,
        n_jobs = None,
        n_estimators = 1,
        epochs = 100,
        batch_size = 32,
        steps_per_epoch = 100,
        network = "DenseNet",
        network_params = dict(
            layers=3,
            last_units=1,
            activation="relu",
            last_activation=None
        ),
        optimizer = "Adam",
        optimizer_params = dict(
            lr=0.001
        ),
        callbacks = "SaveModel",
        callbacks_params = {},
        loss = "MeanSquaredError",
        rescale_output = False,
        pretrained_weights = None
    )
    
    abbrv_dict = dict(
        c = "configfile",
        d = "dataset",
        m = "method",
        p = "params",
        s = "savedir",
        n = "name",
        e = "epochs",
        b = "batch_size",
        o = "optimizer",
        l = "loss"
    )
    
    try:
        opts, args = getopt.getopt(argv,"hc:d:m:p:s:n:e:b:o:l:",
        ["config=", "dataset=", "method=", "params=", "savedir=", "name=",
         "epochs=", "batch_size=", "optimizer=", "loss=", "n_runs=",
         "n_jobs=", "n_estimators=", "steps_per_epoch=", "network=",
         "network_params=", "optimizer_params=", "callbacks=", "callbacks_params=",
         "rescale_output=", "pretrained_weights="])
    except getopt.GetoptError as error:
        print(error)
        print("Available options:")
        print("script.py -c <configfile> -d <dataset> -m <method>"
              " -p <params> -s <savedir> -n <name> -e <epochs>"
              " -b <batch_size> -o <optimizer> -l <loss>"
              " --n_runs --n_jobs --n_estimators --steps_per_epoch --network"
              " --network_params --optimizer_params --callbacks --callbacks_params"
              " --rescale_output --pretrained_weights")
        sys.exit(2)
    
    print(getopt.getopt(argv,"hc:d:m:p:s:n:e:b:o:l:",
        ["config=", "dataset=", "method=", "params=", "savedir=", "name=",
         "epochs=", "batch_size=", "optimizer=", "loss=", "n_runs=",
         "n_jobs=", "n_estimators=", "steps_per_epoch=", "network=",
         "network_params=", "optimizer_params=", "callbacks=", "callbacks_params=",
         "rescale_output=", "pretrained_weights="]))
    
    args_dict = {}
    for opt, arg in opts:
        if opt == '-h':
            print("script.py -c <configfile> -d <dataset> -m <method>"
              " -p <params> -s <savedir> -n <name> -e <epochs>"
              " -b <batch_size> -o <optimizer> -l <loss>"
              " --n_runs --n_jobs --n_estimators --steps_per_epoch --network"
              " --network_params --optimizer_params --callbacks --callbacks_params"
              " --rescale_output --pretrained_weights")
            sys.exit()
        elif opt.replace("-", "") in abbrv_dict:
            opt = "--" + abbrv_dict[opt.replace("-", "")]

        if opt.replace("--", "") in args_dict_default:
            args_dict[opt.replace("--", "")] = arg
    
    config = {}
    if "configfile" in args_dict:
        file = open(args_dict["configfile"], "r")
        config.update(yaml.safe_load(file))
        file.close()
    else:
        config.update(args_dict_default)
    for key, value in args_dict.items():
        if key in ["network_params", "params", "optimizer_params", "callbacks_params"]:
            config[key] = ast.literal_eval(value)
        elif key in ["n_runs", "n_jobs", "n_estimators",
                     "epochs", "batch_size", "steps_per_epoch"]:
            config[key] = int(value)
        elif key in ["rescale_output"]:
            config[key] = bool(value)
        else:
            config[key] = value
    return config


if __name__ == "__main__":
    config = read_args(sys.argv[1:])
    script = config.get("name", "")
    dirname = config.get("savedir", ".")
    n_runs = config.get("n_runs", 1)

    for state in range(n_runs):
        config["state"] = state
        
        if "runs" in config:
        
            for run in config["runs"]:
                config_run = copy.deepcopy(config)
                config_run.update(config_run["runs"][run])
                del config_run["runs"]

                config_run["save_path"] = "{dirname:}/{script:}/{run:}_{state:}".format(
                dirname=dirname, script=script, run=run, state=state)

                if "pretrained_weights" in config_run and config_run["pretrained_weights"] is not None:
                    config_run["pretrained_weights"] = config_run["save_path"].replace(
                    run, "DeepEnsemble")
                    
                train(config_run)
                
        else:
            if "datasets" not in config:
                config["datasets"] = [config["dataset"]]
            if "methods" not in config:
                config["methods"] = {config["method"]: {"method": config["method"]}}
            
            for dataset in config["datasets"]:
                config["dataset"] = dataset
                for method in config["methods"]:
                    run = dataset + "_" + method

                    config_run = copy.deepcopy(config)
                    config_run.update(config_run["methods"][method])
                    del config_run["methods"]
                    del config_run["datasets"]

                    config_run["save_path"] = "{dirname:}/{script:}/{run:}_{state:}".format(
                    dirname=dirname, script=script, run=run, state=state)

                    if "pretrained_weights" in config_run and config_run["pretrained_weights"] is not None:
                        config_run["pretrained_weights"] = config_run["save_path"].replace(
                        method, "DeepEnsemble")
                    
                    train(config_run)