import tensorflow as tf


class autoencoder2D:
    def __init__(
        self,
        inputs,
        aug_model,
        reg_model,
        dropouts=[0.4],
        **kwargs  # NOTE: allows us to ignore other keywords
    ):
        super(autoencoder2D, self).__init__()
        self.inputs = inputs
        self.aug_model = aug_model
        self.reg_model = reg_model
        self.dropouts = dropouts
        self.preprocessing_function = None
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

        segment_length = x.shape[1]
        channels = x.shape[2]
        # NOTE: x.shape = ( , segment_length, channels)

        # NOTE: Encoder
        x = tf.keras.layers.Reshape((segment_length, channels, 1))(x)  # NOTE: x.shape = ( , segment_length, channels)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)

        # NOTE: Decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 1))(x)

        x = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        x = tf.keras.layers.Reshape((segment_length, channels))(x)

        # outputs = self.reg_model(x)
        outputs = x
        model = tf.keras.Model(self.inputs, outputs)

        # self._add_regularization(model, self.regularizer_function, self.weight_decay)

        return model
