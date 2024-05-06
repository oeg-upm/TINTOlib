import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

class FFNNComponent:
    def __init__(
        self,
        filters_ffnn: [int],
        dropout_rate: float,
    ):
        self.filters_ffnn = filters_ffnn
        self.dropout_rate = dropout_rate
        self.ffnn = tf.keras.Sequential()
        for filters in self.filters_ffnn:
            self.ffnn.add(Dense(filters, activation='relu'))
            self.ffnn.add(BatchNormalization())
            self.ffnn.add(Dropout(self.dropout_rate))

    def __call__(
        self,
        x: tf.Tensor,
    ):
        x = self.ffnn(x)
        return x

class AddNorm(layers.Layer):
    """Layer normalization and dropout applied to the sum of inputs (residual connection)."""
    def __init__(
        self,
        dropout_rate: float = 0.1, 
        epsilon: float = 1e-12,
        **kwargs,
    ):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon)

    def call(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
    ):
        y = self.dropout(y)
        # Add & Norm (residual connection followed by layer normalization)
        return self.layer_norm(x + y)

class PositionWiseFFN(layers.Layer):
    """Position-wise feed-forward network."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        **kwargs,
    ):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # First dense layer increases dimensionality from input_size to hidden_size and relu activation
        self.dense1 = layers.Dense(hidden_size, activation='relu')
        # Second dense layer projects back to input_size dimensions
        self.dense2 = layers.Dense(input_size)

    def call(
        self,
        x: tf.Tensor,
    ):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


# Define the TransformerBlock layer

class TransformerBlock(layers.Layer):
    """Encoder block of the Transformer model"""
    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        dropout: float,
        ffn_hidden_size: int,
        use_bias: bool =False,
    ):
        super(TransformerBlock, self).__init__()
        self.attention_head = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=output_dim,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.add_norm_1 = AddNorm(dropout)
        self.feed_forward = PositionWiseFFN(key_dim, ffn_hidden_size)
        self.add_norm_2 = AddNorm(dropout)

    def call(
        self,
        x: tf.Tensor,
    ):
        # Define the model's forward pass
        attention = self.attention_head(x, x, x)
        x = self.add_norm_1(x, attention)
        x = self.add_norm_2(x, self.feed_forward(x))
            
        return x
    
class Encoder(layers.Layer):
    """Transformer block of the Transformer model"""
    def __init__(
        self,
        attention_num_heads: int,
        attention_key_dim: int,
        attention_value_dim: int,
        attention_output_dim: int,
        attention_dropout: float,
        ffn_hidden_size: int,
        num_layers: int,
        attention_use_bias: bool = False,
    ):
        super(Encoder, self).__init__()
        
        # Create a Sequential model to hold the encoder blocks
        self.encoder_blocks = tf.keras.Sequential()
        self.transformer_block = TransformerBlock

        for i in range(num_layers):
            # Add an EncoderBlock as a layer to the Sequential model
            self.encoder_blocks.add(
                self.transformer_block(
                    num_heads=attention_num_heads,
                    key_dim=attention_key_dim,
                    value_dim=attention_value_dim,
                    output_dim=attention_output_dim,
                    dropout=attention_dropout,
                    use_bias=attention_use_bias,
                    ffn_hidden_size=ffn_hidden_size,
                )
            )

    def call(
        self,
        x: tf.Tensor,
    ):
        # Define the model's forward pass
        x = self.encoder_blocks(x)
        return x



class ResidualBlock(layers.Layer):
    """ResNet Residual block"""
    def __init__(
        self,
        filters: int,
        use_1x1conv=False,
        strides: int = 1,
        use_bias: bool = False,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same", use_bias=use_bias)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same", use_bias=use_bias)
        self.bn2 = layers.BatchNormalization()
        if use_1x1conv:
            self.conv3 = layers.Conv2D(filters, kernel_size=1, strides=3)
        else:
            self.conv3 = None
    def call(
        self,
        x: tf.Tensor,
    ):
        # Define the model's forward pass
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = self.conv3(x)
        return self.relu(x + y)

