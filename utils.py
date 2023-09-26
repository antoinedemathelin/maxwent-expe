import time
import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from methods import BaseEnsemble


class SaveModel(tf.keras.callbacks.Callback):
    
    def __init__(self, validation_data, monitor="loss", threshold=False, verbose=1):
        super().__init__()
        rand = np.random.choice(2**20)
        t = int(time.time()*1000)
        filename = "net_%i_%i"%(rand, t)
        if not isinstance(self.model, BaseEnsemble):
            filename += ".hdf5"
        self.filename = filename
        self.threshold = threshold
        self.last_save = 0
        if isinstance(validation_data, tuple):
            xval, yval = validation_data
            self.validation_data = tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(xval),
                tf.data.Dataset.from_tensor_slices(yval))).batch(128)
        else:
            self.validation_data = validation_data
        self.monitor = monitor
        if self.monitor == "acc":
            self.best = 0.
        else:
            self.best = np.inf
        self.verbose = verbose
        
    
    def on_train_begin(self, logs=None):
        self.model.save_weights(self.filename)
        
        if self.threshold:
            if self.monitor == "loss":
                config = self.model.loss.get_config()
                config["reduction"] = tf.keras.losses.Reduction.NONE
                loss_fn = self.model.loss.__class__.from_config(config)
            elif self.monitor == "acc":
                loss_fn = tf.keras.metrics.get(self.monitor)
            else:
                raise ValueError("monitor should be loss or acc")
            losses = []
            for data in self.validation_data:
                x, y = data
                yp = self.model(x)
                if len(yp.shape) > 2:
                    yp = tf.reduce_mean(yp, axis=-1)
                if self.monitor == "acc":
                    if yp.shape[1] == 1:
                        yp = tf.concat((-yp, yp), axis=1)
                    yp = tf.argmax(yp, axis=1)
                    yp = tf.expand_dims(yp, axis=-1)
                    y = tf.reshape(y, yp.shape)
                    losses.append(loss_fn(y, yp))
                else:
                    losses.append(loss_fn(y, yp))
            losses = tf.concat(losses, axis=0)
            
            if self.monitor == "loss":
                self.threshold = (tf.reduce_mean(losses) +
                2. * tf.math.reduce_std(losses) / float(len(losses)))
            else:
                self.threshold = (tf.reduce_mean(losses) -
                2. * tf.math.reduce_std(losses) / float(len(losses)))
            
            print("Start Training, threshold: %.3f"%self.threshold)
            
        
    def on_epoch_end(self, epoch, logs=None):
        monitor = self.model.evaluate(self.validation_data, verbose=0, return_dict=True)[self.monitor]
        if not self.threshold:
            if self.monitor == "acc":
                condition = (monitor >= self.best)
            else:
                condition = (monitor <= self.best)                
        
        else:
            if self.monitor == "acc":
                condition = (monitor >= self.threshold) #self.best - 
            else:
                condition = (monitor <= self.threshold) #self.best + 
            
        message = "Val loss %.3f  "%monitor
        
        if condition:
            self.best = monitor
            self.model.save_weights(self.filename)
            self.last_save = epoch
            message += "Model saved! Epoch %i"%epoch
        if self.verbose:
            print(message)
    
    
    def on_train_end(self, logs=None):
        print("Restore Weights : Epoch %i  Val loss %.3f"%(self.last_save, self.best))
        self.model.load_weights(self.filename)
        if isinstance(self.model, BaseEnsemble):
            shutil.rmtree(self.filename)
        else:
            os.remove(self.filename)



class GaussianNegativeLogLikelihood(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        mu = tf.reshape(y_pred[:, 0], tf.shape(y_true))
        log_sigma_2 = tf.reshape(y_pred[:, 1], tf.shape(y_true))
        error = tf.square(y_true - mu)
        log_sigma_2 = tf.clip_by_value(log_sigma_2, -20., 10000.)
        inv_sigma_2 = tf.exp(-log_sigma_2)
        return 0.5*(inv_sigma_2*error + log_sigma_2)


def gaussian_nll(y_true, y_mu, y_sigma):
    y_sigma = np.clip(y_sigma, 1e-12, np.inf)
    return np.mean(0.5 * (np.log(y_sigma) + (y_true-y_mu)**2 / y_sigma))


def softmax(y):
    if y.shape[1] == 1:
        y = np.clip(y, -20., 10000.)
        y = 1/(1+np.exp(-y))
        y = np.concatenate((1-y, y), axis=1)
        return y
    else:
        return tf.keras.activations.softmax(tf.identity(y), axis=1).numpy()


def entropy(y):
    ys = softmax(y)
    if ys.ndim < 3:
        ys = ys[:, :, np.newaxis]
    ys = ys.mean(-1)
    return -np.sum(ys*np.log(ys+1e-12), 1)


def mu_sigma(y):
    if y.shape[1] == 1:
        return y.mean(-1).ravel(), y.std(-1).ravel()
    else:
        if y.ndim < 3:
            y = y[:, :, np.newaxis]
        mu = y[:, 0, :]
        mean_mu = np.mean(mu, axis=-1)
        sigma2 = np.exp(np.clip(y[:, 1, :], -np.inf, 20.))
        mean_sigma2 = np.var(mu, axis=-1) + np.mean(sigma2, axis=-1)
    return mean_mu, mean_sigma2


def calibration_curve_regression(y_true, y_pred_mu, y_pred_sigma, n_bins=10):
    mu = y_pred_mu; sigma = y_pred_sigma
    alphas = np.linspace(1/(n_bins+1), 1, n_bins, endpoint=False)
    pred_alphas = []
    for alpha in alphas:
        length = norm.ppf(1-alpha/2., scale=sigma)
        in_interval = (y_true >= mu - length) & (y_true <= mu + length)
        pred_alphas.append(np.mean(in_interval))
    alphas = 1 - alphas
    pred_alphas = np.array(pred_alphas)
    return alphas[::-1], pred_alphas[::-1]


def results_regression(model, data):
    scores = {}
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    yptest = model.predict(xtest, batch_size=256, verbose=1)
    ypood = model.predict(xood, batch_size=256, verbose=1)
    
    mu, sigma_test = mu_sigma(yptest)
    
    mae_score = np.abs(ytest.ravel() - mu.ravel()).mean()
    
    nll_score = gaussian_nll(ytest.ravel(), mu, sigma_test)
    cal_curve = calibration_curve_regression(ytest.ravel(), mu, sigma_test)
    cal_score = np.mean(np.abs(cal_curve[0] - cal_curve[1]))
    calup_score = np.mean(np.clip(cal_curve[0] - cal_curve[1], 0., np.inf))
    length = norm.ppf(1-0.01/2., scale=sigma_test)
    in_interval = (ytest.ravel() >= mu - length) & (ytest.ravel() <= mu + length)
    q99_score = np.mean(in_interval)
    
    print("Test: MAE %.4f  NLL %.4f  Cal %.4f  CalUp %.4f  Q99 %.4f"%(mae_score,
                                                                      nll_score,
                                                                      cal_score,
                                                                      calup_score,
                                                                      q99_score))

    scores["mae_id"] = mae_score
    scores["nll_id"] = nll_score
    scores["cal_id"] = cal_score
    scores["calup_id"] = calup_score
    scores["q99_id"] = q99_score
    
    mu, sigma_ood = mu_sigma(ypood)
    mae_score = np.abs(yood.ravel() - mu.ravel()).mean()
    
    nll_score = gaussian_nll(yood.ravel(), mu, sigma_ood)
    cal_curve = calibration_curve_regression(yood.ravel(), mu, sigma_ood)
    cal_score = np.mean(np.abs(cal_curve[0] - cal_curve[1]))
    calup_score = np.mean(np.clip(cal_curve[0] - cal_curve[1], 0., np.inf))
    length = norm.ppf(1-0.01/2., scale=sigma_ood)
    in_interval = (yood.ravel() >= mu - length) & (yood.ravel() <= mu + length)
    q99_score = np.mean(in_interval)
    
    print("OOD: MAE %.4f  NLL %.4f  Cal %.4f  CalUp %.4f  Q99 %.4f"%(mae_score,
                                                                      nll_score,
                                                                      cal_score,
                                                                      calup_score,
                                                                      q99_score))
    scores["mae_ood"] = mae_score
    scores["nll_ood"] = nll_score
    scores["cal_ood"] = cal_score
    scores["calup_ood"] = calup_score
    scores["q99_ood"] = q99_score
    
    in_dist = sigma_test
    ood_dist = sigma_ood
    
    auc = auroc_score(in_dist, ood_dist)
    fpr = 1 - np.mean(ood_dist >= np.quantile(in_dist, 0.95))
    
    scores["auc_ood"] = auc
    scores["fpr_ood"] = fpr
    return scores


def auroc_score(id_score, ood_score):
    gt = np.zeros(len(id_score) + len(ood_score))
    gt[len(id_score):] = 1
    y_pred = np.concatenate((id_score, ood_score))
    auc = roc_auc_score(gt, y_pred)
    return auc


def results_classification(model, data):
    scores = {}
    
    x, y, xval, yval, xtest, ytest, xood, yood = data
    
    yptest = model.predict(xtest, batch_size=256, verbose=1)
    ypood = model.predict(xood, batch_size=256, verbose=1)
    
    in_dist = entropy(yptest)
    ood_dist = entropy(ypood)
    
    auc = auroc_score(in_dist, ood_dist)
    fpr = 1 - np.mean(ood_dist >= np.quantile(in_dist, 0.95))
    
    yptest_soft = softmax(yptest)
    if yptest_soft.ndim < 3:
        yptest_soft = yptest_soft[:, :, np.newaxis]
    yptest_soft = yptest_soft.mean(-1)
    
    acc_score = np.mean(yptest_soft.argmax(1) == ytest)
    nll_test = tf.keras.losses.SparseCategoricalCrossentropy()(ytest,
                                                               yptest_soft).numpy().mean()
    
    print("Test: Acc %.4f  NLL: %.4f"%(acc_score, nll_test))
    
    print("OOD: AUC: %.4f  FPR: %.4f"%(auc, fpr))
    
    scores["acc_id"] = acc_score
    scores["nll_id"] = nll_test
    scores["auc_ood"] = auc
    scores["fpr_ood"] = fpr
    return scores