import os
import pandas as pd
import urllib
import numpy as np
import pathlib
from io import BytesIO, StringIO
from zipfile import ZipFile
from scipy.io import arff


UCI_WEBSITE = "https://archive.ics.uci.edu/static/public/"


data_url_dict = dict(
    yacht=UCI_WEBSITE+"243/yacht+hydrodynamics.zip",
    energy=UCI_WEBSITE+"242/energy+efficiency.zip",
    concrete=UCI_WEBSITE+"165/concrete+compressive+strength.zip",
    wine=UCI_WEBSITE+"186/wine+quality.zip",
    power=UCI_WEBSITE+"294/combined+cycle+power+plant.zip",
    naval=("https://raw.githubusercontent.com/trewaite/Naval-Vessel-Maintenance/"
    "15bb3ed4ba49c9b82e6648e73218232cbe1138a8/UCI%20CBM%20Dataset/data.txt"),
    protein=UCI_WEBSITE+"265/physicochemical+properties+of+protein+tertiary+structure.zip",
    kin8nm="https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff"
)

filename_dict = dict(
    yacht="yacht_hydrodynamics.data",
    energy="ENB2012_data.xlsx",
    concrete="Concrete_Data.xls",
    power="CCPP/Folds5x2_pp.xlsx",
    wine="winequality-red.csv",
    protein="CASP.csv",
)

columns_dict = dict(

    yacht=[
        "Longitudinal position of the center of buoyancy",
        "Prismatic coefficient",
        "Length-displacement ratio",
        "Beam-draught ratio",
        "Length-beam ratio",
        "Froude number",
        "Residuary resistance per unit weight of displacement"],
    
    naval = [
        "Lever position (lp) [ ]",
        "Ship speed (v) [knots]",
        "Gas Turbine (GT) shaft torque (GTT) [kN m]",
        "GT rate of revolutions (GTn) [rpm]",
        "Gas Generator rate of revolutions (GGn) [rpm]",
        "Starboard Propeller Torque (Ts) [kN]",
        "Port Propeller Torque (Tp) [kN]",
        "Hight Pressure (HP) Turbine exit temperature (T48) [C]",
        "GT Compressor inlet air temperature (T1) [C]",
        "GT Compressor outlet air temperature (T2) [C]",
        "HP Turbine exit pressure (P48) [bar]",
        "GT Compressor inlet air pressure (P1) [bar]",
        "GT Compressor outlet air pressure (P2) [bar]",
        "GT exhaust gas pressure (Pexh) [bar]",
        "Turbine Injecton Control (TIC) [%]",
        "Fuel flow (mf) [kg/s]",
        "GT Compressor decay state coefficient",
        "GT Turbine decay state coefficient",],
    
    kin8nm = [
        'theta1',
        'theta2',
        'theta3',
        'theta4',
        'theta5',
        'theta6',
        'theta7',
        'theta8',
        'y'],
)

target_dict = dict(
    energy = ["Y1"],
    naval = ["GT Turbine decay state coefficient"],
    protein = "RMSD",
)


exclude_dict = dict(
    energy = ["Y2"],
    naval = ["GT Compressor decay state coefficient"],
)


option_dict = dict(
    yacht = dict(header=None, usecols=[i for i in range(7)], delim_whitespace=True),
    wine = dict(delimiter=";"),
    kin8nm = dict(header=None, skiprows=23),
    naval = dict(delimiter="  ", header=None, engine='python')
)


custom_opening_dict = dict()


def open_uci_dataset(dataset, online=True, path=None):
    if not dataset in data_url_dict:
        raise ValueError("Dataset `%s` is not available. Available datasets are: "
                         "%s"%(dataset, str(list(data_url_dict.keys()))))
    
    if online:
        path = data_url_dict[dataset]
    
    elif path is None:
        dirname = os.path.dirname(__file__)
        if (not os.path.isdir(os.path.join(dirname, "datasets")) or
            not os.path.isdir(os.path.join(dirname, "datasets", "uci"))):
            raise ValueError("No UCI datasets have been downloaded yet."
                             " Use argument `online=True` or download the dataset with"
                             " the function `download_uci(dataset)`.")
        
        list_files = os.listdir(os.path.join(dirname, "datasets", "uci"))
        no_file_found = True
        for file in list_files:
            if dataset == file.split(".")[0]:
                no_file_found = False
                filename = file
        if no_file_found:
            raise ValueError("The dataset `%s` has not been downloaded yet."
                             " Use argument `online=True` or download the dataset with"
                             " the function `download_uci(dataset)`."%dataset)
        
        path = os.path.join(dirname, "datasets", "uci", filename)
        path = pathlib.Path(path).as_uri()
        
    else:
        path = pathlib.Path(path).as_uri()
        
    if dataset in custom_opening_dict:
        df = custom_opening_dict[dataset](path)
    else:
        if dataset in option_dict:
            kwargs = option_dict[dataset]
        else:
            kwargs = {}
        
        extension = data_url_dict[dataset].split(".")[-1]
        
        if extension == "zip":
            resp = urllib.request.urlopen(path)
            zipfile = ZipFile(BytesIO(resp.read()))
            filename = filename_dict[dataset]
            if filename.split(".")[-1] in ["xls", "xlsx"]:
                df = pd.read_excel(zipfile.open(filename), **kwargs)
            else:
                df = pd.read_csv(zipfile.open(filename), **kwargs)
        elif extension in ["xls", "xlsx"]:
            df = pd.read_excel(path, **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)
        
    if dataset in columns_dict:
        df.columns = columns_dict[dataset]    
    
    if dataset in target_dict:
        target = target_dict[dataset]
    else:
        target = df.columns[-1]
        
    if dataset in exclude_dict:
        df = df.drop(exclude_dict[dataset], axis=1)
    
    if not isinstance(target, list):
        target = [target]
    
    y = df[target]
    X = df.drop(target, axis=1)
    
    return X, y


def download_uci(dataset):
    if not dataset in data_url_dict:
        raise ValueError("Dataset `%s` is not available. Available datasets are: "
                         "%s"%(dataset, str(list(data_url_dict.keys()))))
        
    print("Downloading...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(dirname, "datasets")
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, "uci_datasets")
    if not os.path.isdir(path):
        os.mkdir(path)
    
    list_files = os.listdir(path)
    
    for file in list_files:
        if dataset == file.split(".")[0]:
            print("Dataset `%s` already downloaded."%dataset)
            return True
    
    url = UCI_WEBSITE + data_url_dict[dataset]
    filename = os.path.join(path, dataset+"."+url.split(".")[-1])
    urllib.request.urlretrieve(url, filename)
    
    print("Done! The dataset is stored at `%s`"%filename)
    return True