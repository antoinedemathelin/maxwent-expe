import tensorflow as tf
import numpy as np
from keras import backend
from keras.engine.input_spec import InputSpec
import re
import scipy


class MaxEntDense(tf.keras.layers.Dense):
    
    
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_noise_type="uniform",
        bias_noise_type="uniform",
        kernel_noise_shape=None,
        bias_noise_shape=None,
        kernel_noise_initializer=-10.,
        bias_noise_initializer=-10.,
        rotation=False,
        clip=None,
        **kwargs,
    ):
        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         **kwargs)
        
        self.kernel_noise_initializer = kernel_noise_initializer
        self.bias_noise_initializer = bias_noise_initializer
        self.kernel_noise_type = kernel_noise_type
        self.bias_noise_type = bias_noise_type
        self.kernel_noise_shape = kernel_noise_shape
        self.bias_noise_shape = bias_noise_shape
        self.rotation = rotation
        self.clip = clip
    
    
    def build(self, input_shape):
        
        if isinstance(self.kernel_noise_initializer, float):
            self.kernel_noise_initializer = tf.keras.initializers.Constant(
                value=self.kernel_noise_initializer)
        else:
            self.kernel_noise_initializer = tf.keras.initializers.get(
                self.kernel_noise_initializer)
            
        if isinstance(self.bias_noise_initializer, float):
            self.bias_noise_initializer = tf.keras.initializers.Constant(
                value=self.bias_noise_initializer)
        else:
            self.bias_noise_initializer = tf.keras.initializers.get(
                self.bias_noise_initializer)
        
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False,
        )
        self.kernel_noise = self.add_weight(
            "kernel_noise",
            shape=[last_dim, self.units],
            initializer=self.kernel_noise_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_rotation = self.add_weight(
            "kernel_rotation",
            shape=[last_dim, last_dim],
            dtype=self.dtype,
            trainable=False,
        )
        self.kernel_rotation.assign(tf.eye(last_dim))
        if self.kernel_noise_shape is None:
            self.kernel_noise_shape_ = self.kernel_noise.shape
        else:
            self.kernel_noise_shape_ = self.kernel_noise_shape
        
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=False,
            )
            self.bias_noise = self.add_weight(
                "bias_noise",
                shape=[
                    self.units,
                ],
                initializer=self.bias_noise_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
            if self.bias_noise_shape is None:
                self.bias_noise_shape_ = self.bias_noise.shape
            else:
                self.bias_noise_shape_ = self.bias_noise_shape
        else:
            self.bias = None
        self.built = True

    
    def z_sample(self, kind, shape):
        if kind == "normal":
            z = tf.random.normal(shape)
        elif kind == "uniform":
            z = tf.random.uniform(shape)*2. - 1.
        elif kind == "bernouilli":
            z = tf.random.uniform(shape)
            z = tf.cast(tf.math.greater(0.5, z), self.kernel.dtype)*2. - 1.
        else:
            raise ValueError("Unknow noise distribution")
        return z
    
    
    def call(self, inputs):
        z = self.z_sample(self.kernel_noise_type, self.kernel_noise_shape_)
        
        kernel_noise = tf.math.log(1. + tf.exp(self.kernel_noise))
        if self.rotation:
            if self.clip is None:
                kernel = self.kernel + tf.matmul(self.kernel_rotation, z * kernel_noise)
            else:
                kernel = self.kernel + tf.matmul(self.kernel_rotation, 
                                                 tf.clip_by_value(z * kernel_noise,
                                                                  -self.clip, self.clip))
        else:
            if self.clip is None:
                kernel = self.kernel + z * kernel_noise
            else:
                kernel = self.kernel + tf.clip_by_value(z * kernel_noise, -self.clip, self.clip)
        
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, tf.SparseTensor):
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)

                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            z = self.z_sample(self.bias_noise_type, self.bias_noise_shape_)
            
            bias_noise = tf.math.log(1. + tf.exp(self.bias_noise))
            if self.clip is None:
                bias = self.bias + z * bias_noise
            else:
                bias = self.bias + tf.clip_by_value(z * bias_noise, -self.clip, self.clip)
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs


def replace_layers(model, layer_names="dense", new_layer_constructors=MaxEntDense, **kwargs):
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
                new_layer = constructor.from_config(layer.get_config())
                for k, v in kwargs.items():
                    setattr(new_layer, k, v)
                x = new_layer(layer_input)
                new_layer.kernel.assign(tf.identity(layer.kernel))
                if layer.use_bias:
                    new_layer.bias.assign(tf.identity(layer.bias))
                match = True
        if not match:
            layer.trainable = False
            x = layer(layer_input, training=False)
        inputs_dict[layer.output.name] = x

        if layer.output.name in output_names:
            outputs.append(x)
        if len(outputs) == 1:
            outputs = outputs[0]

    new_model = tf.keras.Model(model.input, outputs)
    return new_model


def replace_layers_svd(model, X_train, layer_names="dense", new_layer_constructors=MaxEntDense, **kwargs):    
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
                
                if X_train is not None:
                    print("Compute SVD")
                    layer_input_rpz = tf.keras.Model(model.inputs, layer.input)
                    XTX = None
                    for batch in X_train:
                        X_train_rpz = layer_input_rpz(batch).numpy()
                        if XTX is None:
                            XTX = X_train_rpz.transpose().dot(X_train_rpz)
                        else:
                            XTX += X_train_rpz.transpose().dot(X_train_rpz)
                    _, Vl = scipy.linalg.eig(XTX)
                    Vl = Vl.astype(np.float32)
                    print("Done!", Vl.shape)
                
                new_layer = constructor.from_config(layer.get_config())
                for k, v in kwargs.items():
                    setattr(new_layer, k, v)
                x = new_layer(layer_input)
                new_layer.kernel.assign(tf.identity(layer.kernel))
                
                if X_train is not None:
                    new_layer.kernel_rotation.assign(tf.identity(Vl)) #.transpose()
                    
                if layer.use_bias:
                    new_layer.bias.assign(tf.identity(layer.bias))
                match = True
        if not match:
            layer.trainable = False
            x = layer(layer_input, training=False)
        inputs_dict[layer.output.name] = x

        if layer.output.name in output_names:
            outputs.append(x)
        if len(outputs) == 1:
            outputs = outputs[0]

    new_model = tf.keras.Model(model.input, outputs)
    return new_model


class MaxWEnt(tf.keras.Model):
    
    def __init__(self, network, lambda_=1., n_pred=50, **kwargs):
        new_network = replace_layers(network, **kwargs)
        super().__init__(new_network.inputs, new_network.outputs)
        self.lambda_ = lambda_
        self.n_pred = n_pred

        
    def call(self, inputs, training=False):
        if not training:
            preds = []
            for i in range(self.n_pred):
                preds.append(super().call(inputs, training=training))
            return tf.stack(preds, axis=-1)
        else:
            return super().call(inputs, training=training)
    
    
    def train_step(self, data):
        x, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        with tf.GradientTape() as tape:
            
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            
            weight_loss = 0.
            count = 0.
            for i in range(len(self.trainable_variables)):
                w = self.trainable_variables[i]
                w = tf.math.log(1. + tf.exp(w))
                weight_loss += tf.reduce_sum(w)
                count += tf.reduce_sum(tf.ones_like(w))

            weight_loss /= count
            
            loss += -self.lambda_ * weight_loss
            loss += sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["weight"] = weight_loss
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
    
    
class MaxWEntSVD(tf.keras.Model):
    
    def __init__(self, network, X_train=None, lambda_=1., n_pred=50, **kwargs):
        kwargs["rotation"] = True
        new_network = replace_layers_svd(network, X_train=X_train, **kwargs)
        super().__init__(new_network.inputs, new_network.outputs)
        self.lambda_ = lambda_
        self.n_pred = n_pred

        
    def call(self, inputs, training=False):
        if not training:
            preds = []
            for i in range(self.n_pred):
                preds.append(super().call(inputs, training=training))
            return tf.stack(preds, axis=-1)
        else:
            return super().call(inputs, training=training)
    
    
    def train_step(self, data):
        x, y = data
        
        if len(y.shape) < 2:
            y = tf.reshape(y, (-1, 1))
        
        with tf.GradientTape() as tape:
            
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            
            weight_loss = 0.
            count = 0.
            for i in range(len(self.trainable_variables)):
                w = self.trainable_variables[i]
                w = tf.math.log(1. + tf.exp(w))
                weight_loss += tf.reduce_sum(w)
                count += tf.reduce_sum(tf.ones_like(w))

            weight_loss /= count
            
            loss += -self.lambda_ * weight_loss
            loss += sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["weight"] = weight_loss
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


