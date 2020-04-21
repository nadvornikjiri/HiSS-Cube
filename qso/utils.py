from os import path

import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
import torch
from torch import nn
from torch.utils.data import TensorDataset


WAVEMIN, WAVEMAX = 3.5843, 3.9501
N_WAVES = 3659
WAVES = np.logspace(WAVEMIN, WAVEMAX, N_WAVES)


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


def preprocessing(X):
    return minmax_scale(X, feature_range=(-1, 1), axis=1)


def get_dataset(X, y):
    return TensorDataset(*list(map(
        torch.from_numpy, [X.reshape(-1, 1, N_WAVES), y.astype("f4")]
    )))


def load_ds(ds_file, grp, va=False):
    X_key, y_key = ("X_tr", "y_tr") if not va else ("X_va", "y_va")
    with h5py.File(ds_file, "r") as ds:
        dom = ds[grp]
        X, y = dom[X_key][...], dom[y_key][:]
    X = preprocessing(X)
    return get_dataset(X, y)


def predict(model, dl, dev):
    bs = dl.batch_size
    trues = torch.zeros(dl.dataset.tensors[0].size(0))
    preds = torch.zeros_like(trues)
    for i, (xb, yb) in enumerate(dl):
        start = i * bs
        end = start + bs
        trues[start:end] = yb
        preds[start:end] = model(xb.to(dev)).squeeze()
    return trues, preds
