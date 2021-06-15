# Add a model to the toolbox

Any TF2 (keras) model filling the following specifications can be evaluated with the toolbox.

## Required specifications

### Model : model definition
A valid Keras model, see https://keras.io/api/models/model/ for more details. The model must return the loss and the accuracy when called.

Use losses from *utils.metrics* (see below for an example) or any loss from TF/Keras with a mask

### Parameters : model parameters
A class containing all the parameters of the model. It must contain al least the four following parameters: 
- **num_epochs**: (int) number of epochs for the training of the model
- **stop_window_size**: (int or *None*) size of the window for early stopping (when validation accuracy) doesn't increase anymore. Set to *None* to disable it.
- **batch_size**: (int) size of each batch
- **learning_rate**: (float) learning rate of the optimizer

### utils.py : utilities functions
A file including (at least) the following function:

- model_preprocessing():

```python
def model_preprocessing(parameters, adj, features):
    # 1. preprocess the features and the adjacency matrix (and eventually make them sparse)
    support = adj
    
    # 2. convert adj and features to a Tensor/SparseTensor 
    #    if sparse features, define num_features_nnz
    num_features_nnz = ...
    
    # 3. add some eventual data for the model initialization
    feed_data = dict()
    feed_data[...] = ...
    
    return support, features, feed_data, num_features_nnz
```
Note that the **adj** element can contain any data

## Example

Define the Model and its parameters:
```python
import tensorflow as tf
from tensorflow import keras

from toolbox.metrics import *

class ModelExample(keras.Model):

    def __init__(self, parameters, input_dim, output_dim, num_features_nonzero, **kwargs):
        super(ModelExample, self).__init__(**kwargs)
        
        self.params = parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.loss_function = parameters.loss_function

        self.layers_ = []
        self.layers_.append(keras.layers.Flatten())
        self.layers_.append(keras.layers.Dense(output_dim, activation=lambda x: x))


    def call(self, inputs):
        x, label, mask, ajd = inputs
        outputs = [x]

        for layer in self.layers:
            hidden = layer(outputs[-1])
            outputs.append(hidden)

        self.outputs = outputs[-1]

        loss = self.loss_function(outputs[-1], label, mask)
        
        acc = masked_accuracy(label, self.outputs, mask)
        
        return loss, acc

    def predict(self):
        return tf.nn.softmax(self.outputs)


class Params(object):
    
    def __init__(self, stop_window_size=8, num_epochs=200, batch_size=64, learning_rate=0.05, activation=tf.nn.relu, features_normalization=2, loss_function="softmax_cross_entropy", loss_parameters=None):
        self.num_epochs = num_epochs
        self.stop_window_size = stop_window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.activation = activation
        self.feats_norm = features_normalization

        self.loss_function = CustomLosses(loss_function, loss_parameters=loss_parameters)
```

Also define the preprocessing function:

```python
import tensorflow as tf
from numpy import linalg as LA

def model_preprocessing(parameters, adj, features):
    features = features / LA.norm(features, ord=parameters.feats_norm)
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    adj = tf.convert_to_tensor(adj)

    feed_data = dict() # or None
    
    return adj, features, feed_data, None
```
