import tensorflow as tf
from tensorflow.keras import Model


class DeepEnsemble(Model):
    
    def __init__(self,
                 network):
        super().__init__(network.inputs, network.outputs)
    
    def train_step(self, data):
        X, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))

        with tf.GradientTape() as tape:

            y_pred = self(X, training=True)
            loss = self.compiled_loss(y, y_pred)
            loss += sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        return logs