
import pandas as pd
import numpy as np
import os
import sys
import time
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
batch_size = 64
seq_length = 100
EPOCHS = 40


def preprocess(text):
	chrs = sorted(set(text))
	print(list(chrs))
	ids_from_chars =\
	preprocessing.StringLookup(vocabulary=list(chrs), mask_token=None)
	chars_from_ids =\
	preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
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
		self.dropout = tf.keras.layers.Dropout(rate=0.2)
		self.dense = tf.keras.layers.Dense(inp_size)
	def call(self, inputs, states=None, return_state=False, training=False):
	    x = inputs
	    x = self.embedding(x, training=training)
	    if states is None:
	      states = self.gru.get_initial_state(x)
	    x, states = self.gru(x, initial_state=states, training=training)
	    x = self.dropout(x)
	    x = self.dense(x, training=training)

	    if return_state:
	      return x, states
	    else:
	      return x



if len(sys.argv) > 1:
	file_p = sys.argv[1]
	df = pd.read_csv(file_p)
	text = "\n".join(map(" ".join, df[[ "author","content"]].values.tolist()))

	examples_per_epoch = len(text)//(seq_length+1)
	ds,ids_from_chars ,chars_from_ids= preprocess(text)
	sequences = ds.batch(seq_length+1, drop_remainder=True)
	ds = sequences.map(split_input_target) 
	ds = ds.shuffle(10000)\
	.batch(batch_size, drop_remainder=True)\
	.prefetch(tf.data.experimental.AUTOTUNE)
	model = theModel(len(ids_from_chars.get_vocabulary()),256,2048)
	loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer='adam', loss=loss)
	# Directory where the checkpoints will be saved
	checkpoint_dir = './training_checkpoints'
	# Name of the checkpoint files
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(\
	filepath=checkpoint_prefix,\
	save_weights_only=True)
	model.fit(ds, epochs=EPOCHS, callbacks=[checkpoint_callback])

	model.save_weights("my_model")




