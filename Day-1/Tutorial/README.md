# Training a Multimodal Contrastive Embedding Model 

This tutorial trains a *small cross-attention head* on top of frozen
[AION](https://github.com/PolymathicAI/AION) embeddings, so that the image
of a galaxy and its spectrum end up close together in a shared 768-d space.
It is the same idea as [AstroCLIP](https://arxiv.org/abs/2310.03024) — but
instead of training two large encoders, we reuse a multimodal foundation
model's embeddings and only learn a small pooling module (~10M parameters,
trains in minutes per epoch on a single consumer GPU).


### Learning objectives

- **MLOps**: Practical experience training a foundation model with torch, Lightning, 
and Weights and Biases on a national supercomputer.
- **Contrastive Learning**: Understand the InfoNCE/CLIP objective — how simply pairing
an image with its own spectrum against in-batch negatives aligns two modalities in a
shared embedding space, the role of the learned temperature and of batch size, and how
physical structure (redshift, stellar populations) emerges without ever being supervised.
- **Adapting Foundation Models for Downstream Tasks**: Instead of fine-tuning a large
encoder, keep the foundation model frozen and train a lightweight attentive probe
(a V-JEPA 2-style cross-attention pooler, ~10M parameters) on top of its embeddings —
the standard, compute-efficient way to specialize a foundation model.
- **Probabilistic Inference from Foundation Model Outputs**: Use frozen embeddings as
learned summary statistics for estimating galaxy properties (redshift, stellar mass)
with a simple k-NN regressor, and learn to judge predictions by their full error
distribution — typical precision vs. catastrophic outliers — rather than a single
squared-error score like $R^2$.


It is also a compact, idiomatic example of a
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) (2.6+)
codebase: a `LightningDataModule`, a `LightningModule`, a YAML config, and a
`LightningCLI` entrypoint with wandb logging.


## 1. Preparing the project 

Once you are logged in on Leonardo, you can run the following commands to prepare
the project:
```bash
git clone https://github.com/Sera91/SMR4228.git
cd SMR4228/Day-1/Tutorial

module load python
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Once this is done, you should have all the dependencies installed, and the project
is ready to run. To be able to see the results of your runs online, please also 
log into Weights and Biases:

```bash
wandb login
```

## 2. Understanding the model

AION gives us a *sequence* of tokens per galaxy per modality; contrastive
learning needs *one vector* per galaxy per modality. The reduction is learned
by a single cross-attention block (`CrossAttentionPool` in
`aion_contrastive/model.py`):

```
learned queries [Q, 768]  ──┐
                            ├──> [depth-1 self-attn blocks over tokens]
AION tokens   [L, 768]    ──┘    ──> cross-attention + MLP ──> mean over Q
                                                             ──> Linear ──> R^768
```

The pooler follows the **attentive pooler of V-JEPA 2**
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2),
`src/models/attentive_pooler.py`).

### The contrastive (InfoNCE / CLIP) loss

Both pooled vectors are L2-normalised, so their dot product is a cosine
similarity. For a batch of $N$ galaxies we build the $N \times N$ similarity
matrix between all images and all spectra, and treat matching as an
$N$-way classification problem in both directions:

$$
\mathcal{L} = \tfrac{1}{2}\left[
\underbrace{-\tfrac{1}{N}\sum_i \log
  \frac{e^{z^{img}_i \cdot z^{spec}_i / \tau}}{\sum_j e^{z^{img}_i \cdot z^{spec}_j / \tau}}}_{\text{image} \to \text{spectrum}}
+ \underbrace{(\dots)}_{\text{spectrum} \to \text{image}}
\right]
$$

The temperature $\tau$ is learned (initialised at 0.07, as in CLIP). Every other galaxy in the batch acts as 
a negative example, which is why contrastive learning likes **large batches**.

Nothing about galaxy physics is hand-coded: the only supervision is
*"this image and this spectrum belong to the same object"*, and the shared
structure (redshift, stellar populations, morphology…) emerges by itself.

## 3. Running it

```bash
sbatch slurm.job
```

Once the run is done, you can upload the weights like so:

```bash
# from a Leonardo login node (has internet)
wandb sync $SCRATCH/wandb_logs/wandb/offline-run-*
```