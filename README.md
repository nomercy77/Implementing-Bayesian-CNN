**Bayesian convolutional neural networks with variational inference**, a variant of convolutional neural networks (CNNs), in which the intractable posterior probability distributions over weights are inferred by **Bayes by Backprop** and use variational inference.
---------------------------------------------------------------------------------------------------------


### Filter weight distributions in a Bayesian Vs Frequentist approach

![Distribution over weights in a CNN's filter.](experiments/figures/BayesCNNwithdist.png)

---------------------------------------------------------------------------------------------------------

### Fully Bayesian perspective of an entire CNN

![Distributions must be over weights in convolutional layers and weights in fully-connected layers.](experiments/figures/CNNwithdist_git.png)

---------------------------------------------------------------------------------------------------------



### Layer types

This repository contains two types of bayesian lauer implementation:  
* **BBB (Bayes by Backprop):**  
  Based on [this paper](https://arxiv.org/abs/1505.05424). This layer samples all the weights individually and then combines them with the inputs to compute a sample from the activations.

* **BBB_LRT (Bayes by Backprop w/ Local Reparametrization Trick):**  
  This layer combines Bayes by Backprop with local reparametrization trick from [this paper](https://arxiv.org/abs/1506.02557). This trick makes it possible to directly sample from the distribution over activations.
---------------------------------------------------------------------------------------------------------




### Directory Structure:
`layers/`:  Contains `ModuleWrapper`, `FlattenLayer`, `BBBLinear` and `BBBConv2d`.  
`models/BayesianModels/`: Contains standard Bayesian models (BBBLeNet, BBBAlexNet, BBB3Conv3FC).  
`models/NonBayesianModels/`: Contains standard Non-Bayesian models (LeNet, AlexNet).  
`checkpoints/`: Checkpoint directory: Models will be saved here.  
`tests/`: Basic unittest cases for layers and models.  
`main_bayesian.py`: Train and Evaluate Bayesian models.  
`config_bayesian.py`: Hyperparameters for `main_bayesian` file.  
`main_frequentist.py`: Train and Evaluate non-Bayesian (Frequentist) models.  
`config_frequentist.py`: Hyperparameters for `main_frequentist` file.  

---------------------------------------------------------------------------------------------------------



### Uncertainty Estimation:  
There are two types of uncertainties: **Aleatoric** and **Epistemic**.  
Aleatoric uncertainty is a measure for the variation of data and Epistemic uncertainty is caused by the model.  
Here, two methods are provided in `uncertainty_estimation.py`, those are `'softmax'` & `'normalized'` and are respectively based on equation 4 from [this paper](https://openreview.net/pdf?id=Sk_P2Q9sG) and equation 15 from [this paper](https://arxiv.org/pdf/1806.05978.pdf).  
Also, `uncertainty_estimation.py` can be used to compare uncertainties by a Bayesian Neural Network on `MNIST` and `notMNIST` dataset. You can provide arguments like:     
1. `net_type`: `lenet`, `alexnet` or `3conv3fc`. Default is `lenet`.   
2. `weights_path`: Weights for the given `net_type`. Default is `'checkpoints/MNIST/bayesian/model_lenet.pt'`.  
3. `not_mnist_dir`: Directory of `notMNIST` dataset. Default is `'data\'`. 
4. `num_batches`: Number of batches for which uncertainties need to be calculated.  

**Notes**:  
1. You need to download the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset from [here](http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz).  
2. Parameters `layer_type` and `activation_type` used in `uncertainty_etimation.py` needs to be set from `config_bayesian.py` in order to match with provided weights. 

---------------------------------------------------------------------------------------------------------



This is an add-on to the original implementation by:

```
@article{shridhar2019comprehensive,
  title={A comprehensive guide to bayesian convolutional neural network with variational inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1901.02731},
  year={2019}
}
```

```
@article{shridhar2018uncertainty,
  title={Uncertainty estimations by softplus normalization in bayesian convolutional neural networks with variational inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1806.05978},
  year={2018}
}
}
```

--------------------------------------------------------------------------------------------------------
