import tensorflow as tf
from keras import layers, Model
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.models import Sequential
from HyNN.layers import Encoder

class PatchEmbeddings(layers.Layer):
    """
    Patch embeddings layer
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch. The output shape is [batch_size, h, w, d_model]
        self.conv = tf.keras.layers.Conv2D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            input_shape=(None, None, in_channels)
        )


    def __call__(self, x):
        """
        * `x` is the input image of shape `[batch_size, height, width, channels]`
        """
        # Apply the convolution layer to get the transformed "patches" [batch_size, h, w, c]
        x = self.conv(x)

        # Rearrange to shape `[batch_size, patches, d_model]` [n, h*w, c] "[batch_size, patch_w, patch_h, feature_dimension]"
        # Each patch 
        n, h, w, c = x.shape
        x = tf.reshape(x, [-1, h * w, c])

        return x


class LearnedPositionalEmbeddings(layers.Layer):
    """
    This adds learned positional embeddings to patch embeddings. In this case patches positions in image.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = tf.Variable(tf.zeros((1, max_len, d_model)), trainable=True)

    def __call__(self, x):
        """
        * `x` is the patch embeddings of shape `[batch_size, patches, d_model]`
        """
        # Get the positional embeddings for the given patches x.shape[1] = patch size
        pe = self.positional_encodings[:,:x.shape[1],:]
        # Element-wise addition of positional embeddings to the patch embeddings to encode spatial information.
        return x + pe
    
class CLSTokenLayer(tf.keras.layers.Layer):
    """
    This layer introduces a trainable `[CLS]` token to the beginning of a sequence of embeddings. 
    """

    def __init__(self, d_model, **kwargs):
        """
        * `d_model` is the dimensionality of the model, matching the dimensionality of the input embeddings.
        """
        super(CLSTokenLayer, self).__init__(**kwargs)
        # Initialize the `[CLS]` token as a trainable variable. This token is shared across all instances in a batch.
        self.cls_token_emb = tf.Variable(initial_value=tf.random.uniform([1, 1, d_model]), trainable=True)

    def call(self, x):
        """
        * `x` is the input embeddings of shape `[batch_size, sequence_length, d_model]`.
        """
        # Get the batch size from the input feature shape
        batch_size = tf.shape(x)[0]
        # Duplicate the `[CLS]` token for each item in the batch
        cls_tokens = tf.tile(self.cls_token_emb, [batch_size, 1, 1])
        # Concatenate the `[CLS]` token with the input features
        return tf.concat([cls_tokens, x], axis=1)

class VisionTransformer(Model):
    """
    ## Vision Transformer

    This combines the [patch embeddings],
    [positional embeddings],
    transformer and the [classification head].
    """
    def __init__(self, d_model: int, patch_size: int, in_channels: int, n_heads: int,n_layers: int, bias: bool = False, dropout: float = 0.1):
        """
        * `transformer_layer` is a copy of a single [transformer layer].
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers].
        * `patch_emb` is the [patch embeddings layer].
        * `pos_emb` is the [positional embeddings layer].
        * `encoder` is the [transformer layer].
        * `cls_token_emb` is the `[CLS]` token embedding.
        * `mlp` is the multi-layer perceptron that takes the `[CLS]` token output.
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = PatchEmbeddings(
            d_model,
            patch_size,
            in_channels
        
        )
        self.pos_emb = LearnedPositionalEmbeddings(d_model)

        # Transformer layers
        self.encoder = Encoder(
            attention_num_heads=n_heads,
            attention_key_dim= d_model,
            attention_value_dim= d_model,
            attention_output_dim= d_model,
            attention_dropout= dropout,
            ffn_hidden_size=d_model*4,
            num_layers=n_layers,
            attention_use_bias=bias,
        )

        # `[CLS]` token embedding
        self.cls_token_layer = CLSTokenLayer(d_model)
        # Final normalization layer, stabilize the training process...
        self.ln = tf.keras.layers.LayerNormalization()

        self.mlp = Sequential([
            Dense(128, input_shape=(d_model,)),  # Start with your input dimension
            BatchNormalization(),  # Apply batch normalization
            Activation('relu'),  # Then apply the activation function
            Dropout(dropout),  # Apply dropout with a rate of 0.5 (adjust as necessary)
            
            Dense(64),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout),  # Adjust dropout rate as necessary
            
            Dense(32),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout),  # Adjust dropout rate as necessary
            
            Dense(16, activation='relu')  # Final layer with activation
        ])

    def __call__(self, x):
        """
        * `x` is the input image of shape `[batch_size, height, width, channels]`
        """
        # Get patch embeddings. This gives a tensor of shape `[batch_size, patches, d_model]`
        x = self.patch_emb(x)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        x = self.cls_token_layer(x)
        # Add positional embeddings
        x = self.pos_emb(x)
        # Pass through transformer layers with no attention masking
        x = self.encoder(x)
        # Apply layer normalization
        # x = self.ln(x)
        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[:,0]
        # Apply further processing on the `[CLS]` token
        mlp_output = self.mlp(x)
        return mlp_output

# Create a VisionTransformer instance
vit_model = VisionTransformer(d_model=512, patch_size=16, in_channels=3, n_heads=8, n_layers=6)

# Compile the model
vit_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Generate dummy data to simulate an input image batch (channel-last format)
import numpy as np
dummy_input = np.random.random((10, 256, 256, 3))  # 10 images of size 256x256 with 3 color channels

# Run a forward pass (you can wrap this in a try-except block to handle potential errors)
try:
    output = vit_model(dummy_input)
    print("Output shape:", output.shape)
except Exception as e:
    print("Error during model execution:", e)