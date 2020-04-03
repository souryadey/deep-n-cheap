# deep-n-cheap ![DnC logo](/dnc_logo.png)

## Welcome
This repository implements _Deep-n-Cheap_ – an AutoML framework to search for deep learning models. Features include:
- **Complexity oriented**: Get models with good performance and low training time or parameter count
- Cuttomizable search space for both architecture and training hyperparameters
- Supports CNNs and MLPs

**Highlight**: State-of-the-art performance on benchmark and custom datasets with training time orders of magnitude lower than competing frameworks and NAS efforts.

**Research paper** available on [arXiv](https://arxiv.org/abs/2004.00974). Please consider citing it if you use or benefit from this work.

## How to run?
- Install Python 3
- Install [Pytorch](https://pytorch.org/)
```
$ pip install sobol_seq tqdm
$ git clone https://github.com/souryadey/deep-n-cheap.git
$ cd deep-n-cheap
$ python main.py
```

For **help**:
```
$ python main.py -h
```

## Complexity customization
Set `wc` to high values to penalize complexity at the cost of performance:
- `--wc=0`: Performance oriented search
- `--wc=0.1`: Balance performance and complexity
- `--wc=10`: Complexity oriented search
- Any non-negative value of `wc` is supported!

## Datasets (including custom)
Set `dataset` to either:
- `--dataset=torchvision.datasets.<dataset>`. Currently supported values of `<dataset>` = MNIST, FashionMNIST, CIFAR10, CIFAR100
- `--dataset='<dataset>.npz'`, where `<dataset>` is a `.npz` file with 4 keys:
	- `xtr`: numpy array of shape (num_train_samples, num_features...), example (50000,3,32,32) or (60000,784). Image data should be in _channels_first_ format.
	- `ytr`: numpy array of shape (num_train_samples,)
	- `xte`: numpy array of shape (num_test_samples, num_features...)
	- `yte`: numpy array of shape (num_test_samples,)
- Some datasets can be downloaded from the links in `dataset_links.txt`. Alternatively, define your own **custom datasets**.

## Examples
1. Search for CNNs between 4-16 layers on CIFAR-10, train each for 100 epochs, run Bayesian optimization for 15 prior points and 15 steps. Optimize for performance only. Estimated search cost: 30 GPU hours
```
python main.py --network 'cnn' --dataset torchvision.datasets.CIFAR10 --input_size 3 32 32 --output_size 10 --wc 0 --numepochs 100 --bo_prior_states 15 --bo_steps 15 --num_conv_layers 4 16
```

2. Search for CNNs between 5-10 layers on Fashion MNIST without augmentation, max channels in any conv layer is 256, search for batch sizes from 64 to 128 and dropout drop probabilities in [0.1,0.2]. Optimize for fast training by using a high `wc`. Download the dataset to the parent directory.
```
python main.py --network 'cnn' --data_folder '../' --dataset torchvision.datasets.FashionMNIST --input_size 1 28 28 --output_size 10 --augment False --wc 1 --num_conv_layers 5 10 --channels_upper 256 --batch_size 64 128 --drop_probs_cnn 0.1 0.2
```

3. Search for MLPs on the custom [Reuters RCV1 dataset](https://ieeexplore.ieee.org/document/8689061), which has 2000 input features and 50 output classes. Search between 0-2 hidden layers, moderately penalize parameter count, search for initial learning rates from 1e-1 to 1e-4, run each model for 20 epochs. Use half data for validation.
```
python main.py --network 'mlp' --dataset 'rcv1_2000.npz' --val_split 0.5 --input_size 2000 --output_size 50 --wc 0.05 --penalize 'numparams' --numepochs 20 --num_hidden_layers 0 2 --lr -4 -1
```

4. Search for CNNs on the custom [Reuters RCV1 dataset](https://ieeexplore.ieee.org/document/8689061) reshaped to 5 channels of 20x20 pixels each and saved as 'rcv1_2000_reshaped.npz'. Use default CNN search settings.
```
python main.py --network 'cnn' --dataset 'rcv1_2000_reshaped.npz' --input_size 5 20 20 --output_size 50
```

## Some results from the research paper
- &gt;91% accuracy on CIFAR-10 in 9 hrs of searching with a model taking 3 sec/epoch to train.
- &gt;95% accuracy on Fashion-MNIST in 17 hrs of searching with a model taking 5 sec/epoch to train.
- &gt;91% accuracy on Reuters RCV1 in 2 hrs of searching with a model with 1M parameters.

## Contact
Deep-n-Cheap is developed and maintained by the [USC HAL team](https://hal.usc.edu/)
