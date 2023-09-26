import tensorflow as tf
import re
from tensorflow_probability.python.layers import DenseReparameterization
import tensorflow_probability.python.distributions as tfd


def replace_layers(model, layer_names="dense",
                   new_layer_constructors=DenseReparameterization,
                   num_data=1,
                   lambda_=1.):
    
    kernel_divergence_fn=(lambda q, p, ignore: (lambda_/num_data) * tfd.kl_divergence(q, p))
    bias_divergence_fn=(lambda q, p, ignore: (lambda_/num_data) * tfd.kl_divergence(q, p))
    
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
        new_layer_constructors = [new_layer_constructors]
    
    layer_inputs_dict = {}
    inputs_dict = {}

    for layer in model.layers:
        if not isinstance(layer.input, list):
            layer_inputs = [layer.input]
        else:
            layer_inputs = layer.input
        layer_inputs_dict[layer.name] = [l.name for l in layer_inputs]

        for inpt in layer_inputs:
            if inpt.name not in inputs_dict:
                inputs_dict[inpt.name] = inpt

    outputs = []
    output_names = [out.name for out in model.outputs]
    for layer in model.layers:
        layer_input = [inputs_dict[name] for name in layer_inputs_dict[layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]
        
        match = False
        for name, constructor in zip(layer_names, new_layer_constructors):
            if re.match(name, layer.name):
                units = layer.units
                activation = layer.activation
                new_layer = constructor(units=units,
                                        activation=activation,
                                        kernel_divergence_fn=kernel_divergence_fn,
                                        bias_divergence_fn=bias_divergence_fn)
                x = new_layer(layer_input)
                match = True
        if not match:
            x = layer(layer_input)
        inputs_dict[layer.output.name] = x

        if layer.output.name in output_names:
            outputs.append(x)
        if len(outputs) == 1:
            outputs = outputs[0]

    new_model = tf.keras.Model(model.input, outputs)
    return new_model


class BNN(tf.keras.Model):
    
    def __init__(self, network, num_data=1, lambda_=1., n_pred=50):
        new_network = replace_layers(network, num_data=num_data, lambda_=lambda_)
        self.n_pred = n_pred
        self.lambda_ = lambda_
        self.num_data = num_data
        super().__init__(new_network.inputs, new_network.outputs)
    
    
    def call(self, inputs, training=False):
        if not training:
            preds = []
            for i in range(self.n_pred):
                preds.append(super().call(inputs, training=training))
            return tf.stack(preds, axis=-1)
        else:
            return super().call(inputs, training=training)
    
    
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
        logs["loss2"] = tf.reduce_sum(loss)
        return logs
    
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        # Compute predictions
        y_pred = self(x, training=False)
        y_pred = tf.reduce_mean(y_pred, axis=-1)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        return logs

