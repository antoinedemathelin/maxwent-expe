import tensorflow as tf
from .base import BaseEnsemble


class NegativeCorrelation(BaseEnsemble):

    def train_step(self, data):
        X, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
            
        if self.loss.__class__.__name__ == 'GaussianNegativeLogLikelihood':
            nll_loss = True
        else:
            nll_loss = False
        
        with tf.GradientTape() as tape:
            
            preds = self(X, training=True)
            
            if nll_loss:
                mean_preds = tf.stop_gradient(tf.reduce_mean(preds[:, :1, :], axis=-1))
            else:
                mean_preds = tf.stop_gradient(tf.reduce_mean(preds, axis=-1))
            
            loss = 0.
            corr = 0.
            for i in range(self.n_estimators):
                if nll_loss:
                    yp_all = tf.concat((preds[:, :1, :i], preds[:, :1, i+1:]), axis=-1)
                else:
                    yp_all = tf.concat((preds[:, :, :i], preds[:, :, i+1:]), axis=-1)
                yp = preds[:, :, i]
                
                loss += self.compiled_loss(y, yp)
                corr += tf.reduce_mean((yp-mean_preds) * tf.stop_gradient(
                tf.reduce_sum(yp_all-tf.expand_dims(mean_preds, axis=-1), axis=-1)))
            
            loss /= self.n_estimators
            corr /= self.n_estimators
            
            loss += self.lambda_ * corr
            loss += sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, mean_preds)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        logs["corr"] = corr
        return logs