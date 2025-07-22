# Understanding Vector Quantized Variational AutoEncoder (VQ-VAE)

Vector Quantized Variational AutoEncoders (VQ-VAE) are powerful generative models that learn **discrete latent representations**. Unlike standard VAEs that rely on continuous, probabilistic encodings and a prior (usually Gaussian), VQ-VAEs operate **non-probabilistically** using a learned *codebook* of embeddings. This structure enables better performance in domains like **image compression**, **speech synthesis**, and **discrete representation learning**.


<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_01.png" alt="Slide 1 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>


## Encoder and Discretization Process

The encoder transforms the input data into a high-dimensional continuous representation \\( z_e \\). However, instead of using \\( z_e \\) directly, VQ-VAE **quantizes** it by replacing it with the closest entry from a learned **codebook** \\( \{e_k\}_{k=1}^{K} \\).

<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_02.png" alt="Slide 2 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>


## Finding the Closest Codebook Vector

Each output \\( z_e \\) of the encoder is compared with the entries in the codebook, and the closest vector (in Euclidean space) is selected:

\\[
z_q = e_k, \quad \text{where } k = \arg\min_j \| z_e - e_j \|_2
\\]


<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_03.png" alt="Slide 3 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>

---

<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_04.png" alt="Slide 4 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>

---


<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_05.png" alt="Slide 5 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>

---

## Quantization and Reconstruction

The selected codebook vector \( z_q \) replaces the original encoder output \( z_e \) and is passed to the decoder for reconstruction. This mechanism enforces a **discrete bottleneck**, enabling compression and interpretability.

<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_06.png" alt="Slide 6 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>

---

<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_07.png" alt="Quantized Representation" style="width:80%; display:block; margin:auto;">
</div>

---
## Loss Functions in VQ-VAE

The total loss in VQ-VAE consists of three components:

1. **Reconstruction Loss** \\( \mathcal{L}_{\text{rec}} \\): measures the distance between the input and the output
2. **Codebook Loss** \\( \mathcal{L}_{\text{cb}} = \| \text{sg}[z_e] - z_q \|_2^2 \\): updates codebook vectors
3. **Commitment Loss** \\( \mathcal{L}_{\text{com}} = \| z_e - \text{sg}[z_q] \|_2^2 \\): encourages encoder outputs to stay near the codebook

<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_08.png" alt="Slide 8 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>

---

## The Straight-Through Estimator (STE)

Since the quantization step is non-differentiable, VQ-VAE employs a **Straight-Through Estimator (STE)**:
<div>
$$\frac{\partial \mathcal{L}_{\text{rec}}}{\partial z_e} := \frac{\partial \mathcal{L}_{\text{rec}}}{\partial z_q}$$
</div>


<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_09.png" alt="Slide 9 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>
This heuristic trick allows gradients to flow from the decoder through the quantized vector \( z_q \) and update the encoder as if \( z_q = z_e \).

---

## Codebook and commitment losses

The stop-gradient (\\( \text{sg}[\cdot] \\)) operator ensures that gradients don't flow into the wrong components during backpropagation.


<div style="border:3px solid #ccc; padding:20px; border-radius:10px;">
    <img src="assets/blogs/vq-vae_resources/slide_10.png" alt="Slide 10 - VQ-VAE" style="width:80%; display:block; margin:auto;">
</div>


## Not a Probabilistic Model

Unlike classical VAEs, VQ-VAEs do **not** define a prior or use a KL-divergence penalty. There is no sampling of latent variables. The entire process is **deterministic**, making it more stable and interpretable in certain applications.

---

## Conclusion

VQ-VAE bridges the gap between **continuous** representation learning and **discrete** latent spaces. Its innovations—quantization, codebooks, commitment losses, and STE—make it uniquely suited for a variety of generative tasks. Future extensions like **VQ-VAE-2** build on this foundation with hierarchical codebooks and improved training schemes.
