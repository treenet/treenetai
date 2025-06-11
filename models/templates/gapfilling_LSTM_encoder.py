import tensorflow as tf


class gapfilling_LSTM_encoder:
    def __init__(
        self,
        inputs,
        aug_model,
        reg_model,
        dropouts=[0.4],
        **kwargs  # allows us to ignore other keywords
    ):
        super(gapfilling_LSTM_encoder, self).__init__()
        self.inputs = inputs
        self.aug_model = aug_model
        self.reg_model = reg_model

        self.dropouts = dropouts

        self.preprocessing_function = None

        # set base learner information
        # ref.: https://gitlab.ethz.ch/lukovicm/datacentricml/-/blob/main/lamella/notebooks/results-comparisons-dev.ipynb
        self.kernel_sizes = [3, 3, 5, 7, 7]
        self.num_filters = [16, 64, 256, 64, 128]
        self.pool_sizes = [2, 2, 3, 2, 3]
        self.stride_sizes = [1, 1, 1, 1, 1]
        self.dense_final_0 = 256

        self.regularizer_function = tf.keras.regularizers.l1
        self.weight_decay = 0.0005


def _add_regularization(
        self, model, regularizer_function=tf.keras.regularizers.l2, weight_decay=0.0005
):
    if weight_decay is None or weight_decay <= 0.0 or regularizer_function is None:
        return

    regularizer = regularizer_function(weight_decay / 2)

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            self._add_regularization(
                layer,
                regularizer_function=regularizer_function,
                weight_decay=weight_decay,
            )
        for attr in ["kernel_regularizer", "bias_regularizer"]:
            if hasattr(layer, attr) and layer.trainable:
                setattr(layer, attr, regularizer)
    return


def model_name(self):
    return "{}".format(self.__class__.__name__)


def make_model(self):
    x = self.inputs

    if self.aug_model:
        x = self.aug_model(x)

    for i in range(5):
        kernel_size = self.kernel_sizes[i]
        num_filter = self.num_filters[i]
        pool_size = self.pool_sizes[i]
        stride_size = self.stride_sizes[i]
        x = tf.keras.layers.Conv2D(
            filters=num_filter,
            kernel_size=kernel_size,
            strides=stride_size,
            activation="relu",
            padding="same",
            use_bias=False,
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(self.dense_final_0, activation="relu")(x)
    x = tf.keras.layers.Dropout(self.dropouts[0])(x)

    outputs = self.reg_model(x)

    model = tf.keras.Model(self.inputs, outputs)

    self._add_regularization(model, self.regularizer_function, self.weight_decay)

    return model
