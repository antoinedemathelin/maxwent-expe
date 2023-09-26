import tensorflow as tf
from .base import BaseEnsemble


class RDE(BaseEnsemble):    
    
    def train_step(self, data):
        x, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        with tf.GradientTape() as tape:
            
            preds = self(x, training=True)
            
            all_dist = []
            for i in range(self.n_estimators):
                for j in range(self.n_estimators):
                    weights_i = getattr(self, "network_%i"%i).trainable_variables
                    weights_j = getattr(self, "network_%i"%j).trainable_variables
                    
                    int_reg_loss = 0.
                    for w1, w2 in zip(weights_i, weights_j):
                        int_reg_loss += tf.reduce_sum(tf.square(w1 - w2))
                    all_dist.append(int_reg_loss)
                    
            sort_dist = tf.sort(tf.stack(all_dist))
            median = sort_dist[int(self.n_estimators**2 / 2)]
            gamma = tf.math.log(self.n_estimators * 1.) / median
            
            loss = 0.
            reg_loss = 0.
            for i in range(self.n_estimators):
                yp = preds[:, :, i]
                loss += self.compiled_loss(y, yp)
                
                reg_loss_i = 0.
                for j in range(self.n_estimators):
                    reg_loss_i += tf.exp(-gamma * all_dist[i * self.n_estimators + j])
                reg_loss_i /= tf.stop_gradient(reg_loss_i)
                reg_loss += reg_loss_i
                
            loss /= self.n_estimators
            reg_loss /= self.n_estimators
            
            loss += reg_loss
            loss += sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, tf.reduce_mean(preds, axis=-1))
        logs = {m.name: m.result() for m in self.metrics}
        logs["reg"] = reg_loss
        return logs