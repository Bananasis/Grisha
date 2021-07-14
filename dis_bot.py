import discord
import pandas as pd
import numpy as np
import os
import sys
import time
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
import tensorflow as tf
batch_size = 64
seq_length = 100

from tensorflow.keras.layers.experimental import preprocessing
chrs = ['\n', ' ', '!', '"', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'V', 'W', 'Z', '\\', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'А', 'Б', 'В', 'Д', 'К', 'Л', 'М', 'Н', 'О', 'П', 'С', 'Т', 'Х', 'Э', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', '🌸', '🚽']
ids_from_chars =\
    preprocessing.StringLookup(vocabulary=list(chrs), mask_token=None)
chars_from_ids =\
    preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def preprocess(text):
    ids = ids_from_chars(list(text))
    return tf.data.Dataset.from_tensor_slices(ids),ids_from_chars,chars_from_ids

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

class theModel(tf.keras.Model):
    def __init__(self,inp_size,emb_size, rnn_size):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(inp_size,emb_size)
        self.gru = tf.keras.layers.GRU(rnn_size,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(inp_size)
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
          return x, states
        else:
          return x


class OneStep(tf.keras.Model):
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
            values=[-float('inf')]*len(skip_ids),
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
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


client = discord.Client()
guild = discord.Guild


if len(sys.argv) > 1:
    file_p = sys.argv[1]
    df = pd.read_csv(file_p)
    text = "\n".join(map(" ".join, df[["content",  "author"]].values.tolist()))[0:101*64]

    examples_per_epoch = len(text)//(seq_length+1)
    ds,ids_from_chars ,chars_from_ids= preprocess(text)
    sequences = ds.batch(seq_length+1, drop_remainder=True)
    ds = sequences.map(split_input_target) 
    ds = ds.shuffle(10000)\
    .batch(batch_size, drop_remainder=True)\
    .prefetch(tf.data.experimental.AUTOTUNE)
    model = theModel(len(chrs)+1,256,2048)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(\
    filepath=checkpoint_prefix,\
    save_weights_only=True)
    model.fit(ds, epochs=1, callbacks=[checkpoint_callback])
    loaded = model.load_weights("training_checkpoints\ckpt_32")
    loaded.assert_consumed()
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif message.content.startswith("_"):

        cmd = message.content.split()[0].replace("_", "")
        if len(message.content.split()) > 1:
            parameters = message.content.split()[1:]


        if cmd == "talk":
            states = None
            next_char = tf.constant([" ".join(parameters[1:])])
            result = [next_char]

            for n in range(int(parameters[0])):
                next_char, states = one_step_model.generate_one_step(next_char, states=states)
                result.append(next_char)
            await message.channel.send(tf.strings.join(result)[0].numpy().decode("utf-8"))
        

            
print("begin")
client.run("ODYyNDMyNDk0NDAyMDExMTM2.YOYQ2Q.Nj5tiRCPXck92C_3yCGIzG7dz4s")
print("done")