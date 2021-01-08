from sklearn.datasets import fetch_rcv1
import pandas as pd
import numpy as np
import os
import importlib

from src.utils import get_project_root

root = get_project_root()


def class_for_name(module_name, class_name):
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


class DataLoader:
    x = None
    y = None

    def __init__(self, dataset, file_ext=None):
        data = globals()[dataset.capitalize()]
        if dataset == 'sonar':
            data = data(dataset, file_ext)
        else:
            data = data()

        self.x = data.x
        self.y = data.y


class Reuters:
    x = None
    y = None

    def __init__(self):
        rcv1 = fetch_rcv1()
        self.x = rcv1["data"]
        self.y = rcv1.target[:, 87]


class Sonar:
    x = None
    y = None

    def __init__(self, file, extension):
        filename = '.'.join([file, extension])

        sonar = pd.read_csv(os.path.join(root, 'data', filename))
        self.x = np.array(sonar.iloc[:, 0:-1])
        y = np.array(sonar.iloc[:, -1])
        y[y == 'Rock'] = -1
        y[y == 'Mine'] = 1
        self.y = y


class Mnist:
    x = None
    y = None
