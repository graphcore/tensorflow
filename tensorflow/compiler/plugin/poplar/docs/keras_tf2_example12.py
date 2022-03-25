import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.datasets import mnist

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()


# Create a dataset for the model.
def create_dataset():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


# An existing model with no pipeline stage assignments.
class ExistingModel(tf.keras.Model):
  def __init__(self):
    super(ExistingModel, self).__init__(self)
    self.dense_layer_1 = tf.keras.layers.Dense(8)
    self.dense_layer_2 = tf.keras.layers.Dense(8)
    self.concat_layer = tf.keras.layers.Concatenate()
    self.dense_layer_3 = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = self.dense_layer_1(inputs)
    x1 = self.dense_layer_2(x)
    x2 = self.dense_layer_2(x)
    x1 = self.dense_layer_2(x1)
    x2 = self.dense_layer_2(x2)
    x = self.concat_layer([x1, x2])
    x = self.dense_layer_3(x)

    return x


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Construct the model inside the strategy.
  model = ExistingModel()

  # Call build to trace the graph generated by the call function.
  # This step is required before getting or setting pipeline stage assignments.
  model.build((28, 28))

  # Get a blank set of pipeline stage assignments.
  assignments = model.get_pipeline_stage_assignment()

  # Modify the assignments by setting pipline stages.
  for assignment in assignments:
    if assignment.layer == model.dense_layer_1:
      assignment.pipeline_stage = 0
    elif assignment.layer == model.dense_layer_2 and assignment.node_index < 2:
      assignment.pipeline_stage = 1
    elif assignment.layer == model.dense_layer_2 and assignment.node_index < 4:
      assignment.pipeline_stage = 2
    elif assignment.layer == model.concat_layer:
      assignment.pipeline_stage = 2
    elif assignment.layer == model.dense_layer_3:
      assignment.pipeline_stage = 3

  # Apply the modified assignments back to the model.
  model.set_pipeline_stage_assignment(assignments)

  # Compile the model for training.
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(),
                metrics=["accuracy"],
                steps_per_execution=256)

  model.set_pipelining_options(gradient_accumulation_steps_per_replica=16,
                               device_mapping=[0, 1, 1, 0])
  model.fit(dataset, epochs=2, steps_per_epoch=128)
