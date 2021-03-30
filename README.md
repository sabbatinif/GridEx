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

## References

1. Airfoil Self-Noise Data Set. https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise (2014), [Online; last accessed 19 Jan 2021].
2. Combined Cycle Power Plant Data Set. https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant (2014), [Online; last accessed 19 Jan 2021].
3. Energy Efficiency Data Set. https://archive.ics.uci.edu/ml/datasets/Energy+efficiency (2012), [Online; last accessed 19 Jan 2021].
4. Gas Turbine CO and NOx Emission Data Set. https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set (2019), [Online; last accessed 19 Jan 2021].
5. Huysmans, J., Baesens, B., Vanthienen, J.: Iter: an algorithm for predictive regression rule extraction. In: International Conference on Data Warehousing and Knowledge Discovery. pp. 270{279. Springer (2006).
6. Wine Quality Data Set. https://archive.ics.uci.edu/ml/datasets/Wine+Quality (2009), [Online; last accessed 19 Jan 2021].
