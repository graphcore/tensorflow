import argparse
import tensorflow as tf

from tensorflow.python import ipu

from tensorflow.python.ipu.keras.layers import Embedding
from tensorflow.python.ipu.keras.layers import LSTM

from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizer_v2.adam import Adam

max_features = 20000


# Define the dataset
def get_dataset():
  (x_train, y_train), (_, _) = imdb.load_data(num_words=max_features)

  x_train = sequence.pad_sequences(x_train, maxlen=80)

  ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  ds = ds.repeat()
  ds = ds.map(lambda x, y: (x, tf.cast(y, tf.int32)))
  ds = ds.batch(32, drop_remainder=True)
  return ds


# Define the model
def get_model():
  return ipu.keras.SequentialPipelineModel(
      [[Embedding(max_features, 128)],
       [LSTM(128, dropout=0.2),
        Dense(1, activation='sigmoid')]],
      gradient_accumulation_count=16)


#
# Main code
#

# Parse command line args
parser = argparse.ArgumentParser("Config Parser", add_help=False)
parser.add_argument('--steps-per-epoch',
                    type=int,
                    default=768,
                    help="Number of steps in each epoch.")
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    help="Number of epochs to run.")
args = parser.parse_args()

# Configure IPUs
cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 2)
ipu.utils.configure_ipu_system(cfg)

# Set up IPU strategy
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():

  model = get_model()

  model.compile(loss='binary_crossentropy', optimizer=Adam(0.005))

  model.fit(get_dataset(),
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs)
