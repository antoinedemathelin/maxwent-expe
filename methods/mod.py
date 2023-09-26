import tensorflow as tf
from .base import BaseEnsemble

    
class MOD(BaseEnsemble):   
    
    def train_step(self, data):
        X, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        X_ood = tf.random.uniform(
            tf.shape(X),
            minval=tf.reduce_min(X, axis=0),
            maxval=tf.reduce_max(X, axis=0),
            dtype=X.dtype)
        
        losses = []
        
        if self.loss.__class__.__name__ == 'GaussianNegativeLogLikelihood':
            nll_loss = True
        else:
            nll_loss = False
        
        with tf.GradientTape() as tape:
            
            preds = self(X, training=True)
            preds_ood = self(X_ood, training=False)
            
            if nll_loss:
                var_ood = tf.math.reduce_variance(preds_ood[:, 0, :], axis=-1)
                var_ood = tf.reduce_mean(var_ood)
            else:
                var_ood = tf.math.reduce_variance(preds_ood, axis=-1)
                var_ood = tf.reduce_mean(var_ood)
            
            loss = 0.
            for i in range(self.n_estimators):
                yp = preds[:, :, i]
                loss += self.compiled_loss(y, yp)
            loss /= self.n_estimators

            loss += -self.lambda_ * var_ood
            loss += sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, tf.reduce_mean(preds, axis=-1))
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        logs["reg"] = var_ood
        return logs