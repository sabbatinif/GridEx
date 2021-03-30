# GridEx

This repository contains an implementation of the ITER and GridEx algorithms and useful code to test these procedures on different data sets.

## Classes

#### Dataset.py
Class for modeling a data set. This class also splits the whole data set into the training and test sets.

#### ITER.py

An implementation of the ITER algorithm.

#### Gridex.py

An implementation of the GridEx Algorithm.

#### Regressor.py

Class for training a neural network solving regression tasks.

#### utils.py

A collection of useful functions.

## Examples

The algorithms can be simply tested on a collection of predefined data sets (for more details see the References section).

With the following instrucion the functions for reproducing the experiments are loaded:

```python
from utils import *
```

To split the data sets into train and test sets and to build and train artificial neural network predictors, this instruction can be used:

```python
trainAndSave()
```

The following instruction applies ITER to all the available data sets and displays the results:

```python
testIter()
```

Finally, the same can be executed for GridEx with this instruction:

```python
testGridex()
```
