"""
Temporal Convolutional Network (TCN) model for multi-task learning.
"""

from __future__ import annotations
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TCNBlock(layers.Layer):
    """
    Single TCN block with dilated causal convolutions.
    
    A TCN block consists of:
    1. Dilated causal 1D convolution
    2. Batch normalization
    3. Activation (ReLU)
    4. Dropout
    5. Another dilated causal conv + normalization
    6. Residual connection
    """
    
    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        """
        Initialize TCN block.
        
        Args:
            n_filters: Number of convolutional filters
            kernel_size: Size of convolution kernel
            dilation_rate: Dilation rate for temporal receptive field
            dropout_rate: Dropout probability
        """
        super().__init__(**kwargs)
        
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # First conv block
        self.conv1 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        
        # Second conv block
        self.conv2 = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation('relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Residual connection (1x1 conv to match dimensions if needed)
        self.match_dims = None
    
    def build(self, input_shape):
        """Build layer components."""
        # Check if we need to match dimensions for residual
        if input_shape[-1] != self.n_filters:
            self.match_dims = layers.Conv1D(
                filters=self.n_filters,
                kernel_size=1,
                padding='same'
            )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor of shape (batch, timesteps, channels)
            training: Boolean flag for training mode
            
        Returns:
            Output tensor of shape (batch, timesteps, n_filters)
        """
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        if self.match_dims is not None:
            residual = self.match_dims(inputs)
        else:
            residual = inputs
        
        return layers.Add()([x, residual])
    
    def get_config(self):
        """Get config for serialization."""
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for attention.
    
    Adds position information to embeddings so attention can distinguish
    different timesteps. Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
    
    def build(self, input_shape):
        """Build positional encoding matrix."""
        d_model = input_shape[-1]
        
        # Create position encoding matrix
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                         -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((self.max_len, d_model))
        
        # Use tf.tensor_scatter_nd_update for assignment
        indices_sin = tf.stack([
            tf.repeat(tf.range(self.max_len), d_model // 2),
            tf.tile(tf.range(0, d_model, 2), [self.max_len])
        ], axis=1)
        indices_cos = tf.stack([
            tf.repeat(tf.range(self.max_len), d_model // 2),
            tf.tile(tf.range(1, d_model, 2), [self.max_len])
        ], axis=1)
        
        sin_values = tf.reshape(tf.sin(position * div_term), [-1])
        cos_values = tf.reshape(tf.cos(position * div_term), [-1])
        
        pe = tf.tensor_scatter_nd_update(pe, indices_sin, sin_values)
        pe = tf.tensor_scatter_nd_update(pe, indices_cos, cos_values)
        
        self.pe = tf.Variable(pe, trainable=False, name='positional_encoding')
        
        super().build(input_shape)
    
    def call(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len})
        return config


class TCNModel:
    """
    Multi-task TCN for gap filling and hourly prediction.
    
    Architecture:
    1. Encoder: Stack of TCN blocks with increasing dilation
    2. Optional: Multi-head attention for global context
    3. Two decoder branches:
       a) Reconstruction: Predict masked 10-min inputs
       b) Hourly prediction: Predict cleaned hourly targets
    """
    
    def __init__(
        self,
        n_input_channels: int = 11,
        n_target_channels: int = 3,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout_rate: float = 0.2,
        input_length: int = 4320,
        output_length: int = 720,
        use_attention: bool = False,
        n_attention_heads: int = 4,
        attention_key_dim: int = 32
    ):
        """
        Initialize TCN model.
        
        Args:
            n_input_channels: Number of input channels
            n_target_channels: Number of target channels
            n_filters: Number of filters per TCN block
            kernel_size: Convolution kernel size
            n_blocks: Number of TCN blocks
            dropout_rate: Dropout probability
            input_length: Length of input sequence (10-min steps)
            output_length: Length of output sequence (hourly steps)
            use_attention: Whether to add attention after TCN encoder
            n_attention_heads: Number of attention heads (if use_attention=True)
            attention_key_dim: Key dimension per attention head
        """
        self.n_input_channels = n_input_channels
        self.n_target_channels = n_target_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate
        self.input_length = input_length
        self.output_length = output_length
        self.use_attention = use_attention
        self.n_attention_heads = n_attention_heads
        self.attention_key_dim = attention_key_dim
        
        self.model = None
    
    def build(self) -> keras.Model:
        """
        Build the TCN model architecture.
        
        Returns:
            Keras Model with two outputs:
            - recon_output: Reconstructed 10-min inputs (batch, 4320, 11)
            - hourly_output: Predicted hourly targets (batch, 720, 3)
        """
        # Input layers
        input_x = layers.Input(
            shape=(self.input_length, self.n_input_channels),
            name='input_x'
        )
        input_mask = layers.Input(
            shape=(self.input_length, self.n_input_channels),
            name='input_mask'
        )
        
        # Combine input and mask
        x = layers.Concatenate()([input_x, input_mask])
        
        # TCN encoder: Stack of blocks with exponentially increasing dilation
        for i in range(self.n_blocks):
            dilation = 2 ** i
            x = TCNBlock(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation,
                dropout_rate=self.dropout_rate,
                name=f'tcn_block_{i}'
            )(x)
        
        # Optional attention layer for global context
        if self.use_attention:
            # Add positional encoding for attention
            x = PositionalEncoding(max_len=self.input_length, name='pos_encoding')(x)
            
            # Multi-head self-attention
            # For efficiency on long sequences, we use a strided approach
            # Downsample -> Attention -> Upsample
            
            # Downsample for efficient attention (factor of 6)
            x_downsampled = layers.AveragePooling1D(
                pool_size=6, strides=6, name='attention_downsample'
            )(x)
            
            # Apply multi-head attention on downsampled sequence
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_attention_heads,
                key_dim=self.attention_key_dim,
                dropout=self.dropout_rate,
                name='multi_head_attention'
            )(x_downsampled, x_downsampled)
            
            # Add & Norm (standard transformer pattern)
            x_downsampled = layers.Add(name='attention_residual')([x_downsampled, attn_output])
            x_downsampled = layers.LayerNormalization(name='attention_layer_norm')(x_downsampled)
            
            # Upsample back to original resolution
            x_upsampled = layers.UpSampling1D(size=6, name='attention_upsample')(x_downsampled)
            
            # Merge attention context with original features
            x = layers.Concatenate(name='merge_attention')([x, x_upsampled])
            
            # Project back to n_filters dimensions
            x = layers.Conv1D(
                filters=self.n_filters,
                kernel_size=1,
                activation='relu',
                name='attention_projection'
            )(x)
        
        # Branch 1: Reconstruction (10-min resolution)
        recon = layers.Conv1D(
            filters=self.n_filters // 2,
            kernel_size=1,
            activation='relu',
            name='recon_intermediate'
        )(x)
        recon = layers.Dropout(self.dropout_rate)(recon)
        
        recon_output = layers.Conv1D(
            filters=self.n_input_channels,
            kernel_size=1,
            activation='linear',
            name='recon_output'
        )(recon)
        
        # Branch 2: Hourly prediction
        # Downsample from 10-min to hourly (factor of 6)
        hourly = layers.AveragePooling1D(
            pool_size=6,
            strides=6,
            name='downsample_to_hourly'
        )(x)
        
        hourly = layers.Conv1D(
            filters=self.n_filters // 2,
            kernel_size=1,
            activation='relu',
            name='hourly_intermediate'
        )(hourly)
        hourly = layers.Dropout(self.dropout_rate)(hourly)
        
        hourly_output = layers.Conv1D(
            filters=self.n_target_channels,
            kernel_size=1,
            activation='linear',
            name='hourly_output'
        )(hourly)
        
        # Create model
        self.model = keras.Model(
            inputs=[input_x, input_mask],
            outputs=[recon_output, hourly_output],
            name='TCN_MultiTask'
        )
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = 3e-4,
        recon_masked_weight: float = 1.0,
        recon_unmasked_weight: float = 0.05,
        hourly_weight: float = 1.0
    ):
        """
        Compile the model with custom loss weights.
        
        The reconstruction loss is weighted differently for masked vs unmasked regions.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            recon_masked_weight: Weight for masked reconstruction loss
            recon_unmasked_weight: Weight for unmasked reconstruction loss
            hourly_weight: Weight for hourly prediction loss
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        # Create custom loss for weighted reconstruction
        def weighted_recon_loss(y_true, y_pred):
            """
            Weighted MSE loss that emphasizes masked regions.
            
            The mask is passed through model inputs and used here.
            """
            # Get mask from model input (assuming it's available in training)
            # For now, use standard MSE
            # TODO: Implement proper masked loss
            return tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'recon_output': 'mse',
                'hourly_output': 'mse'
            },
            loss_weights={
                'recon_output': recon_masked_weight,
                'hourly_output': hourly_weight
            },
            metrics={
                'recon_output': ['mae'],
                'hourly_output': ['mae']
            }
        )
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model must be built first")
        return self.model.summary()
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model must be built first")
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str) -> keras.Model:
        """Load model from file."""
        # Placeholder loss function for constrained models
        def constrained_hourly_loss(y_true, y_pred):
            import tensorflow as tf
            return tf.reduce_mean(tf.abs(y_true - y_pred))
        
        return keras.models.load_model(
            filepath,
            custom_objects={
                'TCNBlock': TCNBlock,
                'PositionalEncoding': PositionalEncoding,
                'constrained_hourly_loss': constrained_hourly_loss
            }
        )


def create_tcn_model(
    n_filters: int = 64,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout_rate: float = 0.2,
    learning_rate: float = 3e-4,
    use_attention: bool = False,
    n_attention_heads: int = 4,
    attention_key_dim: int = 32
) -> keras.Model:
    """
    Convenience function to create and compile a TCN model.
    
    Args:
        n_filters: Number of filters per block
        kernel_size: Convolution kernel size
        n_blocks: Number of TCN blocks
        dropout_rate: Dropout probability
        learning_rate: Learning rate
        use_attention: Whether to add attention after TCN encoder
        n_attention_heads: Number of attention heads (if use_attention=True)
        attention_key_dim: Key dimension per attention head
        
    Returns:
        Compiled Keras model
    """
    tcn = TCNModel(
        n_filters=n_filters,
        kernel_size=kernel_size,
        n_blocks=n_blocks,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        n_attention_heads=n_attention_heads,
        attention_key_dim=attention_key_dim
    )
    
    model = tcn.build()
    tcn.compile_model(learning_rate=learning_rate)
    
    return model
