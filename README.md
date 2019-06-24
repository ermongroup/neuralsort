# Stochastic Optimization of Sorting Networks via Continuous Relaxations

This repository provides a reference implementation for learning NeuralSort-based models as described in the paper:

> Stochastic Optimization of Sorting Networks via Continuous Relaxations  
> [Aditya Grover](https://aditya-grover.github.io), [Eric Wang](https://ericjwang.com), Aaron Zweig and [Stefano Ermon](https://cs.stanford.edu/~ermon/).  
> International Conference on Learning Representations (ICLR), 2019.  
> Paper: https://openreview.net/pdf?id=H1eSS3CcKX

## Requirements

The codebase is implemented in Python 3.7. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
```

## Datasets

The scripts for downloading and loading the MNIST and CIFAR10 datasets are included in the `datasets_loader` folder. These scripts will be called automatically the first time the `main.py` script is run.

## Options

Learning and inference of differentiable kNN models is handled by the `pytorch/run_dknn.py` script which provides the following command-line arguments:

```
  --k INT                 number of nearest neighbors
  --tau FLOAT             temperature of sorting operator
  --nloglr FLOAT          negative log10 of learning rate
  --method STRING         one of 'deterministic', 'stochastic'
  --dataset STRING        one of 'mnist', 'fashion-mnist', 'cifar10'
  --num_train_queries INT number of queries to evaluate during training.
  --num_train_neighbors INT number of neighbors to consider during training.
  --num_samples INT       number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  -resume                 start a new model, instead of loading an older one
```

Learning and inference of quantile-regression models is handled by the `tf/run_median.py` script, which provides the following command-line arguments:

```
  --M INT                 minibatch size
  --n INT                 number of elements to compare at a time
  --l INT                 number of digits in each multi-mnist dataset element
  --tau FLOAT             temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         one of 'vanilla', 'sinkhorn', 'gumbel_sinkhorn', 'deterministic_neuralsort', 'stochastic_neuralsort'
  --n_s INT               number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  --lr FLOAT              initial learning rate
```

Learning and inference of sorting models is handled by the `tf/run_sort.py` script, which provides the following command-line arguments:

```
  --M INT                 minibatch size
  --n INT                 number of elements to compare at a time
  --l INT                 number of digits in each multi-mnist dataset element
  --tau FLOAT             temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         one of 'vanilla', 'sinkhorn', 'gumbel_sinkhorn', 'deterministic_neuralsort', 'stochastic_neuralsort'
  --n_s INT               number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  --lr FLOAT              initial learning rate

```

## Examples

_Training dKNN model to classify CIFAR10 digits_

```
cd pytorch
python run_dknn.py --k=9 --tau=64 --nloglr=3 --method=deterministic --dataset=cifar10
```

_Training quantile regression model to predict the median of sets of nine 5-digit numbers_

```
cd tf
python run_median.py --M=100 --n=9 --l=5 --method=deterministic_neuralsort
```

_Training sorting model to sort sets of five 4-digit numbers_

```
cd tf
python run_sort.py --M=100 --n=5 --l=4 --method=deterministic_neuralsort
```

## Citing

If you find NeuralSort useful in your research, please consider citing the following paper:

> @inproceedings{   
> grover2018stochastic,   
> title={Stochastic Optimization of Sorting Networks via Continuous Relaxations},  
> author={Aditya Grover and Eric Wang and Aaron Zweig and Stefano Ermon},  
> booktitle={International Conference on Learning Representations},  
> year={2019},  
> url={https://openreview.net/forum?id=H1eSS3CcKX},  
> }
