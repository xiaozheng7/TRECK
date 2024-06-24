import numpy as np
import pandas as pd


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MAPE_100(pred, true):
    df_pred = pd.DataFrame(pred.flatten())
    df_trues = pd.DataFrame(true.flatten())
    df_results = pd.concat ([df_trues,df_pred],axis=1)
    names = ['trues','pred']
    df_results.columns = names
    df_over = df_results[df_results['trues']>100]
    preds_in = df_over['pred'].values
    trues_in = df_over['trues'].values

    return np.mean(np.abs((preds_in - trues_in) / trues_in))



def metric(pred, true):
    pred = np.clip(pred, 0, 999999)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape_100 = MAPE_100(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape_100, mape, mspe, rse, corr
