import tensorflow as tf
from tensorflow.keras import Model

    
class AnchoredNetwork(Model):
    
    def __init__(self,
                 network,
                 sigma=1.,
                 lambda_=1.):
        
        super().__init__(network.inputs, network.outputs)
        self.sigma = sigma
        self.lambda_ = lambda_
        
        self.anchors_ = []
        for i in range(len(self.trainable_variables)):
            self.anchors_.append(tf.random.normal(self.trainable_variables[i].shape) * self.sigma)
        
    
    def train_step(self, data):
        X, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))

        with tf.GradientTape() as tape:

            y_pred = self(X, training=True)
            loss = self.compiled_loss(y, y_pred)

            reg_loss = 0.
            count = 0.
            for j in range(len(self.anchors_)):
                reg_loss += tf.reduce_sum(
                    tf.square(self.trainable_variables[j] - self.anchors_[j]))
                count += tf.reduce_sum(tf.ones_like(self.trainable_variables[j]))

            reg_loss /= count

            loss += self.lambda_ * reg_loss
            loss += sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["reg"] = reg_loss
        return logs