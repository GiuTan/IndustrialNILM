import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



num_layers = 2
d_model = 128
num_heads = 4
dff = 512
rate = 0.1

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calcola l'attenzione Scaled Dot-Product.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class NILMTransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, window_size, rate=0.1):
        super(NILMTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.window_size = window_size

        self.input_projection = layers.Dense(d_model, name="input_projection")

        self.position_embedding = layers.Embedding(window_size, d_model, name="position_embedding")

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)  # Normalizzazione finale

    def call(self, inputs, training, mask=None):



        x = self.input_projection(inputs)  # (batch_size, window_size, d_model)


        positions = tf.range(self.window_size, dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.position_embedding(positions)
        x += position_embeddings

        x = self.dropout(x, training=training)
        x = self.layernorm(x)


        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

def create_BERT(num_layers, d_model, num_heads, dff,
                                  window_size, num_appliances=2, rate=0.1):

    input_aggregate = keras.Input(shape=(window_size, 1), dtype=tf.float32, name="aggregate_power_input")



    nilm_encoder = NILMTransformerEncoder(num_layers, d_model, num_heads, dff, window_size, rate)

    # L'output dell'encoder è (batch_size, window_size, d_model)
    encoder_output = nilm_encoder(input_aggregate, training=True)


    output_disaggregation = layers.Dense(num_appliances, activation='linear', name="disaggregated_power_output")(
        encoder_output)


    model = keras.Model(inputs=input_aggregate,
                        outputs=output_disaggregation,
                        name="NILM_Transformer_Model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])
    return model


