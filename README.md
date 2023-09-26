# Maximum Weight Entropy

This repository provides the source code of the Maximum Weight Entropy (**MaxWEnt**) experiments.

## Overview

MaxWEnt is an **uncertainty quantification** method, particularly suited for **Out-Of-Distribution detection**. The method is designed as an application of the **Maximum Entropy** principle to stochastic neural networks. Given a set of observations $\mathcal{S} = \\{ (x_1, y_1), ..., (x_n, y_n) \\} \subset \mathcal{X} \times \mathcal{Y}$ and a neural network $h_w: \mathcal{X} \to \mathcal{Y}$, parameterized by the weight vector $w \in \mathcal{W}$, the method learns the parameters $\phi \in \mathbb{R}^D$ of a distribution $q_{\phi}$ defined over $\mathcal{W}$. MaxWEnt fosters the **weight diversity** of $q_{\phi}$ by penalizing the average empirical risk of the distribution by the **weight entropy**. Formally, the MaxWEnt optimization is expressed as follows:

$$\min_{\phi \in \mathbb{R}^D} \\; \mathbb{E}\_{q_{\phi}}\[\mathcal{L}\_{\mathcal{S}}(w)\] - \lambda \mathbb{E}\_{q_{\phi}}\[-\log(q_{\phi}(w))\].$$

Where:
 - $\mathbb{E}\_{q_{\phi}}\[\mathcal{L}\_{\mathcal{S}}(w)\]$ is the average empirical risk over $q_{\phi}$
 - $\mathbb{E}\_{q_{\phi}}\[-\log(q_{\phi}(w))\]$ is the entropy of $q_{\phi}$
 - $\lambda$ is a trade-off parameter

MaxWEnt significantly improves the out-of-distribution uncertainty estimation in comparison to the main baselines: DeepEnsemble and the standard Bayesian Neural Network (BNN) with centered gaussian prior. The figure below presents the results obtained on two synthetic datasets:

<table>
  <tr valign="top">
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_cla_de_5.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_cla_mcd_1.png">
    </td>
     <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_cla_bnn_1.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_cla_mwe_1.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_cla_mwesvd_1.png">
    </td>
  </tr>
  <tr valign="top">
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_reg_de_5.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_reg_mcd_1.png">
    </td>
     <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_reg_bnn_1.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_reg_mwe_1.png">
    </td>
    <td width="20%">
         <img src="https://github.com/antoinedemathelin/maxwent-expe/blob/da4662d52753c2130d412aaa08e5b58876cde215/images/toy_reg_mwesvd_1.png">
    </td>
  </tr>
  <tr valign="top">
    <td width="20%" align="center">
         <b>Deep Ensemble</b>
    </td>
    <td width="20%" align="center">
         <b>MCDropout</b>
    </td>
     <td width="20%" align="center">
         <b>BNN</b>
    </td>
    <td width="20%" align="center">
         <b>MaxWEnt</b>
    </td>
    <td align="center" width="20%">
         <b>MaxWEnt-SVD</b>
    </td>
  </tr>
 <tr valign="top">
  <td width="100%" colspan="5">
   <b>Uncertainty Estimation Comparison</b>. Above: "two-moons" 2D classification dataset. Below: 1D-regression. For classification, uncertainty estimates, in shades of blue, are computed with the average of prediction entropy over multiple predictions (darker areas correspond to higher uncertainty). For regression, the ground-truth is represented in black and the predicted confidence intervals of length $4 \sigma_w(x)$  in light blue, with $\sigma_w(x)$ computed as the standard deviation over multiple predictions. The two eperiments can be run in the respective notebooks `Synthetic_Classification.ipynb` and `Synthetic_Regression.ipynb`.
   </td>
  </tr>
</table>

## Experiments

To run the experiments, first clone the repository with:
```
git clone https://github.com/antoinedemathelin/maxwent-expe.git
```
or
```
git clone git@github.com:antoinedemathelin/maxwent-expe.git
```
On your labtop, create a conda environment (with python 3.11 preferably). Run, for instance,
```
conda create -n maxwent python=3.11
conda activate maxwent
```
Then, install the requirements from the `requirements.txt` file:
```
pip install -r requirements.txt
```

The experiments can then be conducted using one of the following config files: `1d_reg_all`, `2d_classif_all`, `uci_all`, `citycam_all` in the `configs/` folder. For instance:
```
python run.py --config configs/1d_reg_all.yml
```
A single experiment can also be conducted for a particular dataset and method:
```
python run.py -c configs/uci.yml -d yacht_interpol -m MCDropout -p {'rate':0.1}
```
The models are stored in the `results/` folder and the scores in the `logs/` folder.

**Note** : the CityCam dataset should first be downloaded from this [website](https://www.citycam-cmu.com/), then preprocessed with the following command line:
```
python preprocessing_citycam.py <path_to_the_citycam_dataset_on_your_labtop>
```
