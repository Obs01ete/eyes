## Few-shot eye classification

Dmitrii Khizbullin, 2021

The solution is trained in two stages. Stage one is completely unsupervised variational autoencoder trained on all 3600 images. Reconstruction loss is L2 error, there is extra L2 norm loss imposed on the latent vector of size 50. The purpose of VAE is to reduce dimentionality of images to avoid overfit during classification.

![](assets/vae1.png)

![](assets/vae2.png)

The second stage is supervised classification on a manually annotated 40 samples (20 open and 20 closed eyes). 26 samples go to training, 14 go to validation. Validation accuracy reaches 89% suggesting that VAE-based pretraining helps to boost accuracy even for an extremely small dataset.

To create conda environment:
```bash
conda env create -f environment.yml
```

To run training:
```bash
python train.py
```
It produces `vae.pth` and `classifier.pth` artifacts.

To run inference:
```bash
python inference.py test/
```