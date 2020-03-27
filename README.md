# deep-n-cheap
This repository implements _Deep-n-Cheap_ – an AutoML framework to search for deep learning models. Features include:
- **Complexity oriented**: Get models with good performance and low training time or parameter count
- Cuttomizable search space for both architecture and training hyperparameters
- Supports CNNs and MLPs

**Highlight**: State-of-the-art performance on benchmark and custom datasets with training time orders of magnitude lower than competing frameworks and NAS efforts. Research paper link coming soon!

### How to run?
- Install Python 3
- Install [Pytorch](https://pytorch.org/)
```
$ pip install sobol_seq tqdm
$ git clone https://github.com/souryadey/predefinedsparse-nnets.git
$ cd deep-n-cheap
$ python main.py
```

### Customization options and help
`$ python main.py -h`

Set `wc` to high values to penalize complexity at the cost of performance:
- `--wc=0`: Performance oriented search
- `--wc=0.1`: Balance performance and complexity
- `--wc=10`: Complexity oriented search
- Any non-negative value of `wc` is supported!

### Examples

1. Search for CNNs between 4-16 layers on CIFAR-10, train each for 100 epochs, run Bayesian optimization for 15 prior points and 15 steps. Optimize for performance only. Estimated search cost: 30 GPU hours
```
python main.py --network 'cnn' --dataset 'cifar10' --wc 0 --numepochs 100 --bo_prior_states 15 --bo_steps 15 --num_conv_layers 4 16
```

2. Search for CNNs between 5-10 layers on Fashion MNIST without augmentation, max channels in any conv layer is 256, search for batch sizes from 64 to 128 and dropout drop probabilities in [0.1,0.2]. Optimize for fast training by using a high `wc`. Download the dataset to the parent directory.
```
python main.py --network 'cnn' --dataset 'fmnist' --data_folder '../' --wc 1 --num_conv_layers 5 10 --channels_upper 256 --batch_size 64 128 --drop_probs_cnn 0.1 0.2
```

3. Search for MLPs on the custom [Reuters RCV1 dataset](https://ieeexplore.ieee.org/document/8689061) between 0-2 hidden layers, moderately penalize parameter count, search for initial learning rates from 1e-1 to 1e-4, run each model for 20 epochs.
```
python main.py --network 'mlp' --dataset 'rcv1_2000' --wc 0.05 --penalize 'numparams' --numepochs 20 --num_hidden_layers 0 2 --lr -4 -1
```

Dataset notes:
- To run CNNs, datasets are downloaded automatically into `data_folder`
- To run MLPs, put your datasets in .npz format in `data_folder`. Some datasets are provided, see `mlp_dataset_links.txt` for downloading

### Contact
Deep-n-Cheap is developed and maintained by the [USC HAL team](https://hal.usc.edu/)

