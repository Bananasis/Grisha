import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

BATCHSIZE = 64
SEQLEN = 100
EPOCHS = 30
RNNSIZE = 256
EMBSIZE = 1024
MODELDIR = "my_model"


def preprocess(comments):
    chrs = sorted(set(comments))
    print(list(chrs))
    ids_from_chars = \
        preprocessing.StringLookup(vocabulary=list(chrs), mask_token=None)
    chars_from_ids = \
        preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    ids = ids_from_chars(list(comments))
    return tf.data.Dataset.from_tensor_slices(ids), ids_from_chars, chars_from_ids


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def prepare_batches(comments, batch_size, seq_len):
    ds, ifc, cfi = preprocess(comments)
    sequences = ds.batch(seq_len + 1, drop_remainder=True)
    ds = sequences.map(split_input_target)
    ds = ds.shuffle(10000) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds, ifc, cfi


class Network(tf.keras.Model):
    def __init__(self, inp_size, emb_size, rnn_size, drop_rate):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(inp_size, emb_size)
        self.gru = tf.keras.layers.GRU(rnn_size,
                                       return_sequences=True,
                                       return_state=True)
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.dense = tf.keras.layers.Dense(inp_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        if training:
            x = self.dropout(x)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def load(self, file_weight):
        loaded = self.load_weights(file_weight)
        loaded.assert_consumed()

    def compile_and_train(self, ds, epochs=20, checkpoints=True):
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer='adam', loss=loss)
        checkpoint_callback = []
        if checkpoints:
            checkpoint_dir = './training_checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_prefix,
                save_weights_only=True)
        self.fit(ds, epochs=epochs, callbacks=[checkpoint_callback])

    class TextGenerator(tf.keras.Model):
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')] * len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        @tf.function
        def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                  return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits / self.temperature
            # Apply the prediction mask: prevent "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_p = sys.argv[1]
        df = pd.read_csv(file_p)
        text = "\n".join(map(" ".join, df[["author", "content"]].values.tolist()))
        ds, ifc, cfi = prepare_batches(text, BATCHSIZE, SEQLEN)
        model = Network(len(ifc.get_vocabulary()), EMBSIZE, RNNSIZE)
        model.compile_and_train(ds, epochs=EPOCHS, checkpoints=True)

        model.save_weights(MODELDIR + "/weight")
        df = pd.DataFrame(columns=["seq_len", "init_seq", "inp_size", "emb_size", "rnn_size", "chrs"])
        df.append(
            {
                "seq_len": SEQLEN
                , "init_seq": ds[SEQLEN + 1]
                , "inp_size": len(ifc.get_vocabulary())
                , "emb_size": EMBSIZE
                , "rnn_size": RNNSIZE
                , "chrs": "".join(ifc.get_vocabulary())
            },
            ignore_index=True,
        )
        df.to_csv(MODELDIR + "/index")
