# Configurable Transformer library

PyTorch code for basic functionalities. The code is intended to be used as the basis for further experiments.

For unit testing we will reproduce parts of the [Decision Transformer](https://arxiv.org/abs/2106.01345) and [Vision Transformer](https://arxiv.org/abs/2010.11929) papers.

## Package Description

`arguments.py` contains all available hyperparameters.

`core.py` contains 
*   a self attention layer class `SelfAttention` that can be used as causal and standard layer.
*   a transformer core class `Core` that can be used as a transformer encoder, decoder, or another configurable custom module.

`ops.py` contains code for embedder on predictor modules.
*   The `EmbedderBank` is a module that tokenizes different types of data into a numerical embedding flexibly, as defined by the EmbedderCore configuration. It produces an dictionary of vectors that can be flexibly queried by the downstream models, as defined by the transformer configuration or by the model configuration of the model that contains the transformer core (e.g. vision transformer). This flexibility will allow to tokenize various data and feed them selectively to specialized downstream models.
*   The `PredictorBank` takes the model output and produces a dictionary of predictions, as defined by the model configuration.

`utils.py` contains code for various helper functions.

`decision_transformer.py` contains the code for the decision transformer model.

`vision_transformer.py` contains the code for the vision transformer model.

### Training

Basic usage is

```bash
$ python __test__.py
```

See [arguments.py](https://github.com/sebvoigtlaender/state_rl_basics/blob/main/arguments.py) for available parameters. 

### Tests

Unit tests will follow.

### Lessons learned