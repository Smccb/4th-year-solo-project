import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))