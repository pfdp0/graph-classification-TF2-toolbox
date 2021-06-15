# Graph Classification TF2 toolbox

Graph Classification TF2 is a Python toolbox for comparing 
different semi-supervised classification model on graph-structured data.

The toolbox includes four CNN-based models (GCN, DCNN, GWNN and M-GWNN) as well as three comparison models (SVM, AutoSVM and MLP).
It is also possible to add new models, these models must follow the specifications detailed in [ADD_MODEL.md](./ADD_MODEL.md).

## Usage

The following code runs GCN on two datasets (with 10 runs on 5 folds) and saves the results in the **results/** folder
```python
from toolbox.main import run_experiments

# Import the desired models
from methods.GCN.models import GCN
import methods.GCN.params as GCNparams
import methods.GCN.utils as GCNutils

# Define the models to run
models_to_run = {
    'GCN': {
        'function': GCN,
        'params_fct': GCNparams,
        'utils_fct': GCNutils,
        'params_range': {'learning_rate': [0.001, 0.01],
                         'hidden_layer_sizes': [16, 32, 64]}
    },
}

# Select the datasets
datasets_list = ['myciteseerA100Xsym', 'mycoraA100Xsym']

# Run the experiments (10 times 5 fold validation)
run_experiments(models_to_run, datasets_list)
```

