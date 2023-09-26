import os
import inspect
import tensorflow as tf
from tensorflow.keras import Model


class BaseEnsemble(Model):

    def __init__(self,
                 build_network,
                 n_estimators=5,
                 lambda_=1.,
                 **params):
        
        super().__init__()
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        
        for i in range(self.n_estimators):
            net = build_network(**params)
            net._name = "network_%i"%i
            setattr(self, "network_%i"%i, net)
            
            
    def call(self, inputs, training=False):
        preds = []
        for i in range(self.n_estimators):
            yp = getattr(self, "network_%i"%i)(inputs)
            preds.append(yp)
        if len(preds[-1].shape) > 2:
            return tf.concat(preds, axis=-1)
        else:
            return tf.stack(preds, axis=-1)
    
    
    def individual_call(self, num, inputs, training=False):
        net = getattr(self, "network_%i"%num)
        if not training:
            preds = []
            for i in range(self.n_pred):
                preds.append(net(inputs, training=training))
            return tf.stack(preds, axis=-1)
        else:
            return net(inputs, training=training)
    
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        # Compute predictions
        y_pred = self(x, training=False)
        
        if self.loss.__class__.__name__ == 'GaussianNegativeLogLikelihood':
            y_pred = self.gaussian_nll_preprocessing(y_pred)
        
        y_pred = tf.reduce_mean(y_pred, axis=-1)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        return logs
    
    
    def gaussian_nll_preprocessing(self, y_pred):
        mu = tf.reduce_mean(y_pred[:, 0, :], axis=-1, keepdims=True)
        sigma_2 = tf.exp(tf.clip_by_value(y_pred[:, 1, :], -10000., 20.))
        sigma_2 = tf.reduce_mean(sigma_2, axis=-1, keepdims=True)
        var_mu = tf.math.reduce_variance(y_pred[:, 0, :], axis=-1, keepdims=True)
        log_new_sigma_2 = tf.math.log(tf.clip_by_value(sigma_2 + var_mu, 1e-12, 1e12))
        new_y_pred = tf.concat((mu, log_new_sigma_2), axis=1)
        print(new_y_pred.shape)
        return tf.expand_dims(new_y_pred, axis=-1)
    
    
    def save_weights(self, file):
        os.makedirs(file, exist_ok=True)
        for num in range(self.n_estimators):
            file_path = os.path.join(file, "net_%i.hdf5"%num)
            getattr(self, "network_%i"%num).save_weights(file_path)
            
            
    def load_weights(self, file):
        for num in range(self.n_estimators):
            file_path = os.path.join(file, "net_%i.hdf5"%num)
            getattr(self, "network_%i"%num).load_weights(file_path)