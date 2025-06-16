import tensorflow as tf


# NOTE: see at the last part of
#  https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
"""
The CNN LSTM model will read subsequences of the main sequence in as blocks, extract features from each block, 
then allow the LSTM to interpret the features extracted from each block.

One approach to implementing this model is to split each window of 128 time steps into subsequences for the CNN model to 
process. For example, the 128 time steps in each window can be split into four subsequences of 32 time steps.

# reshape data into time steps of sub-sequences
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

We can then define a CNN model that expects to read in sequences with a length of 32 time steps and nine features.

The entire CNN model can be wrapped in a TimeDistributed layer to allow the same CNN model to read in each of the four 
subsequences in the window. The extracted features are then flattened and provided to the LSTM model to read, extracting 
its own features before a final mapping to an activity is made.
"""

class climate_processing_LSTM_CNN:
    def __init__(
        self,
        inputs,
        aug_model,
        reg_model,
        dropouts=[0.5, 0.2],
        **kwargs  # NOTE: allows us to ignore other keywords
    ):
        super(climate_processing_LSTM_CNN, self).__init__()
        self.input_shape = inputs
        self.aug_model = aug_model
        self.reg_model = reg_model
        self.dropouts = dropouts
        self.preprocessing_function = None
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
        #x = self.input_shape # TODO: implement the shape information in the pipeline.

        # NOTE: optional augmentation
        if self.aug_model:
            x = self.aug_model(x)

        inputs = tf.keras.Input(shape=(720, 6))
        x = inputs
        n_timesteps = 720
        n_input_features = 6 # TODO: this information should be passed from a different fucntion
        n_output_features = 2
        # NOTE: x.shape = ( , segment_length, channels)

        # TODO: find a more general way to introduce the two variables below.
        n_steps = 10
        n_length = 72

        print('x: ', x.shape)

        x = tf.keras.layers.Reshape((n_steps, n_length, n_input_features))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                                            input_shape=(n_length, n_input_features))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Dropout(self.dropouts[0])(x)
        x = tf.keras.layers.LSTM(
            input_shape=(n_timesteps, n_input_features),
            return_sequences=True,
            units=n_timesteps*2,
        )(x)
        x = tf.keras.layers.Dropout(self.dropouts[1])(x)
        x = tf.keras.layers.LSTM(
            units=n_timesteps,
            return_sequences=True,
        )(x)

        x = tf.keras.layers.Dropout(self.dropouts[1])(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_timesteps))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(n_timesteps, activation='relu')(x)
        x = tf.keras.layers.Dense(n_timesteps * n_output_features, activation='relu')(x)
        x = tf.keras.layers.Reshape((n_timesteps, n_output_features))(x)

        # TODO: the output of the regression model has a different shape than the label series.
        #  This has to be solved or the regression model should not be used when comparing multi-channel series.
        # outputs = self.reg_model(x)
        outputs = x

        model = tf.keras.Model(inputs, outputs)

        # TODO: fix the regularization
        # self._add_regularization(model, self.regularizer_function, self.weight_decay)

        return model
