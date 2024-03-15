# What's in a Prior? Learned Proximal Networks for Inverse Problems

  

This is the official implementation of the paper [What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/forum?id=kNPcOaqC5r&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)) @ [ICLR 2024](https://iclr.cc/Conferences/2024)

  

by [Zhenghan Fang](), [Sam Buchanan](https://sdbuchanan.com/), and [Jeremias Sulam](https://sites.google.com/view/jsulam)

  

--------------------

  

We propose *learned proximal networks* (LPN), a new class of deep neural networks that *exactly implement the proximal operator* of a general learned function. Such an LPN implicitly learns a regularization function for inverse problems that can be characterized and evaluated, shedding light onto what has been learned from data and improving the interpretability of learning-based solutions. In turn, we present a new training problem, dubbed *proximal matching*, that provably promotes the recovery of the correct regularization term (i.e., the log of the data distribution).

Moreover, we show convergence for PnP reconstruction algorithms using LPN with minimal and verifiable assumptions.

  
  
  

![Learning Laplacian](assets/laplacian_compact.png)

The proximal operator $f_\theta$ and log-prior $R_\theta$ learned by LPN for the Laplacian distribution using the $\ell_2$, $\ell_1$, or proximal matching ($\mathcal{L}_{PM}$) loss.

  

![Learned prior for MNIST](assets/mnist_gaussian.png)

Learned prior for hand-written digit images in MNIST.

  

## Installation

  

The code is implemented with Python 3.9.16 and PyTorch 1.12.0. Install the conda environment by

```
conda env create -f environment.yml
```

  

Install the `lpn` package

```
pip install -e .
```

  

## Dataset Preparation

The datasets are placed in `data/` folder.

  

### MNIST

The dataset is already in `data/mnist` and has the following structure:

```
data/
└── mnist
    ├── labels.npy
    └── mnist.npy
```

  
  

### CelebA

Download files of the CelebA dataset, as defined in the filelist in torchvision's CelebA class:

  

`img_align_celeba.zip, list_attr_celeba.txt, identity_CelebA.txt, list_bbox_celeba.txt, list_landmarks_align_celeba.txt, list_eval_partition.txt`

  

directly from the authors' [google drive link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=drive_link), and place them in `data/celeba/celeba`. Unzip `img_align_celeba.zip`.

  

The resulting directory should have the following structure:

```

data/
└── celeba
    └── celeba
        ├── img_align_celeba
        ├── identity_CelebA.txt
        ├── list_attr_celeba.txt
        ├── list_bbox_celeba.txt
        ├── list_eval_partition.txt
        └── list_landmarks_align_celeba.txt
```

  

### MayoCT

Download the dataset from the authors' [google drive link](https://drive.google.com/drive/folders/1gKytBtkTtGxBLRcNInx2OLty4Gie3pCX?usp=sharing), and place it in `data/mayoct`. See the authors' [github repo](https://github.com/Subhadip-1/unrolling_meets_data_driven_regularization) and [paper](https://arxiv.org/abs/2106.03538) for more details.

  

The resulting directory should have the following structure:

```

data/
└── mayoct
    └── mayo_data_arranged_patientwise
        ├── test
        │   ├── FBP
        │   ├── Phantom
        │   └── Sinogram
        └── train
            ├── FBP
            ├── Phantom
            └── Sinogram
```

  
  

## How to Run the Code

Code of main functionalities of LPN is placed in the `lpn` folder.



Code for repoducing the experiments in the paper is placed in the `exps` folder.

  

### Laplacian Experiment

  

For reproducing the Laplacian experiment, use code in `exps/laplacian/`.

  

1. Train: `laplacian_train.ipynb`

  

2. Test: `laplacian_test.ipynb`

  

3. Visualize results

- Plot Fig. 1 in the paper: `viz_compact.ipynb`
- Plot Fig. 6 in the supplementary of the paper: `viz_supp.ipynb`

  

Outputs (figures, models, and results) will be saved in `exps/laplacian/experiments/`.

  
  
  

### MNIST Experiment

Code for reproducing the MNIST experiment is in `exps/mnist/`.

  

1. Train:

```
bash exps/mnist/train_mnist.sh
```

  

- Model will be saved at `exps/mnist/experiments/mnist/model.pt`.

  

- We also provide the [pretrained model](#pretrained-checkpoints).

  

2. Compute prior:

  

```
bash exps/mnist/prior_mnist.sh
```

  

- Results will be saved in `exps/mnist/experiments/mnist/prior`.

  
  

3. Visualize results (Figures 3 and 7 in the paper)

  

- Learned prior at example images: `notebooks/viz_img_and_prior.ipynb`

  

- Violin plot: `notebooks/viz_violin.ipynb`

  

- Set `perturb_mode` in the notebooks to `gaussian`, `convex`, or `blur` for different perturbation modes.

  

- Figures will be saved in `exps/mnist/experiments/mnist/figures`.

  
  
  

### CelebA Experiment

Code for reproducing the CelebA experiment is in `exps/celeba/`.

  

1. Train:

```
bash exps/celeba/train.sh
```

- Two models will be trained with two noise levels (0.05 and 0.1), and saved in `exps/celeba/models/lpn/s={0.05, 0.1}/model.pt`.

  

- We also provide the [pretrained models](#pretrained-checkpoints).

  
  

2. Run deblurring:

```
python exps/celeba/test.py --sigma_blur [BLUR LEVEL] --sigma_noise [NOISE LEVEL]
```

- E.g., `python test.py --sigma_blur 1.0 --sigma_noise 0.02` will run deblurring using LPN with Gaussian blur kernel standard deviation $\sigma_{blur}=1.0$ and noise standard deviation $\sigma_{noise}=0.02$.

  

- Results will be saved in `exps/celeba/results/inverse/deblur/blur=[BLUR LEVEL]_noise=[NOISE LEVEL]/admm/lpn/{x,y,xhat}`. `x` and `y` contain the clean images and blurred observation, respectively. `xhat` contains the deblurred images.

  

<!-- 3. Visualize

- Figures: `notebooks/reports/celeba/viz_deblur.ipynb`

- Numerical results: `notebooks/reports/celeba/table_deblur.ipynb` -->

  
  

### MayoCT Experiment

Code for reproducing the MayoCT experiment is in `exps/mayoct/`.

  

1. Train:

```
bash exps/mayoct/train.sh
```

  

- Model will be saved in `exps/mayoct/experiments/mayoct/`
- We also provide the [pretrained model](#pretrained-checkpoints).

  

2. Run tomography:

  

```
python exps/mayoct/inverse_mayoct_tomo.py
```

  

- Results will be saved in `exps/mayoct/results/inverse/mayoct/tomo`.

  

3. Run compressed sensing (CS):

  

```
bash exps/mayoct/test_cs.sh
```

  

- Results will be saved in `exps/mayoct/results/inverse/mayoct/cs`.

  
  
  

<!-- 4. Visualize

- Figures: `experiments_code/ct/notebooks/vis_ct_{cs, ct}.ipynb`

- Numerical results: `experiments_code/ct/notebooks/table.ipynb` -->

  
  
  

## Pretrained checkpoints

All checkpoints are provided in this [Google drive](https://drive.google.com/drive/folders/1qtOra7EDas8gDXGHMsCfSjjIqdvnnb5E?usp=sharing).

  
  

## Acknowledgements

- [scico](https://github.com/lanl/scico)
- [Prox-PnP](https://github.com/samuro95/Prox-PnP)
- [unrolling_meets_data_driven_regularization](https://github.com/Subhadip-1/unrolling_meets_data_driven_regularization)
- [odl](https://odlgroup.github.io/odl/)


## References

  

If you find the code useful, please consider citing

```bib
@inproceedings{
    fang2023whats,
    title={What's in a Prior? Learned Proximal Networks for Inverse Problems},
    author={Zhenghan Fang and Sam Buchanan and Jeremias Sulam},
    booktitle={International Conference on Learning Representations},
    year={2024}
}
```