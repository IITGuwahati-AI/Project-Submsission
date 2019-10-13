import tensorflow as tf

# made only the model for LSTM with vocab 1000 and embedding vector dimension 32

v_size = 1000
em_dim = 32

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(v_size,em_dim)
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))
	tf.keras.layers.Dense(10,activation="relu")
	tf.keras.layers.Dense(1,activation="sigmoid")
	])