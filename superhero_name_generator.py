import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Bidirectional, Dense
import matplotlib.pyplot as plt

with open('superheroes.txt', 'r') as f:
  data = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
    split='\n',
)

tokenizer.fit_on_texts(data)

char_to_index = tokenizer.word_index
index_to_char = dict((v, k) for k, v in char_to_index.items())

names = data.splitlines()

tokenizer.texts_to_sequences(names[0])

def name_to_seq(name):
  return [tokenizer.texts_to_sequences(c)[0][0] for c in name]

def sequence_to_name(sequence):
  return ''.join([index_to_char[i] for i in sequence if i != 0])

sequences = []

for name in names:
  seq = name_to_seq(name)
  if len(seq) >= 2:
    sequences  += [seq[:i] for i in range(2, len(seq)+1)]

max_len = max([len(seq) for seq in sequences])

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, padding='pre', maxlen=max_len
)

padded_sequences.shape

x, y = padded_sequences[:, :-1], padded_sequences[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y)

num_chars = len(char_to_index.keys()) + 1

model = Sequential([
                    Embedding(num_chars, 8, input_length=max_len-1),
                    Conv1D(64, 5, strides=1, activation='relu', padding='causal'),
                    MaxPool1D(2),
                    LSTM(32),
                    Dense(num_chars, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

h = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
    ]
)

epochs_ran = len(h.history['loss'])

plt.plot(range(1, epochs_ran+1), h.history['val_accuracy'], label='Validation')
plt.plot(range(1, epochs_ran+1), h.history['accuracy'], label='Training')
plt.legend()
plt.show()

def generate_names(seed):
  for i in range(40):
    seq = name_to_seq(seed)
    padded = tf.keras.preprocessing.sequence.pad_sequences([seq], padding='pre',
                                                           maxlen=max_len-1,
                                                           truncating='pre')
    prediction = model.predict(padded)[0]
    pred_char = index_to_char[tf.argmax(prediction).numpy()]
    seed += pred_char

    if pred_char == '\t':
      break
  return seed

print(generate_names('gr'))

