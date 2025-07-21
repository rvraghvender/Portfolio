
# Variational AutoEncoder (VAE)

<div style="border:2px solid #ccc; padding:15px; border-radius:10px; background-color:#f4f4f4;">
  <h3>Table of Contents</h3>
  <ol style="line-height:1.8;">
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#objective">Objective</a></li>
    <li><a href="#kl-divergence">Kullback-Leibler Divergence</a></li>
    <li><a href="#loss-function">Loss Function</a></li>
    <li><a href="#reparameterization">Reparameterization Trick</a></li>
    <li><a href="#generator">Generator</a></li>
  </ol>
</div>


<h2 id="introduction">1. Introduction</h2>
<div style="border:3px solid #ccc; padding:20px; border-radius:5px; background-color: lightblue;">
  <strong>Introduction to VAE:</strong> 

-   A Variational Autoencoder (VAE) is a generative model

-   It learns to encode input data into a lower-dimensional space

-   VAE maximizes the likelihood of the data while introducing
    regularization

-   The encoder and decoder are trained together using backpropagation

-   VAEs are useful for generating new, similar data based on learned
    distributions
</div>



<h2 id="architecture">2. Architecture</h2>
<img src="assets/blogs/vae_resources/VAE.png" alt="Architecture of VAE" style="width:80%; display:block; margin:auto;">


>
<strong>Basic Terminology:</strong>

$$p_\theta(\mathbf{z} | \mathbf{x}) = \dfrac{p_\theta(\mathbf{x} | \mathbf{z}) p_\theta(\mathbf{z})}{p_\theta(\mathbf{x})}$$

<ol>
  <li><strong>Prior</strong>: \( p_\theta(\mathbf{z}) \)</li>
  <li><strong>Posterior</strong>: \( p_\theta(\mathbf{z} | \mathbf{x}) \)</li>
  <li><strong>Likelihood</strong>: \( p_\theta(\mathbf{x} | \mathbf{z}) \) </li>
  <li><strong>Marginal Likelihood</strong>: \( p_\theta(\mathbf{x}) \) </li>
</ol>


<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color: lightgrey;">
<strong>For example: </strong>
If, we're asked to imagine a animal:
    <ol>
        <li>\( \mathbf{x} \): data that we want to model a.k.a the animal.</li>
        <li>\( \mathbf{z} \): latent variable a.k.a our imagination.</li>
        <li>\( p(\mathbf{x}) \): probability distribution of the data, i.e. thatanimal kingdom.</li>
        <li>\( p(\mathbf{z}) \): probability distribution of latent variable, i.e.our brain, the source of our imagination.</li>
        <li>\( p(\mathbf{x} | \mathbf{z}) \): distribution of generating data givenlatent variable, e.g. turning imagination into real animal.</li>
    </ol>
</div>

<h2 id="objective">3. Objective</h2>
<!-- ------------------------------------------ -->
Our goal is to model the data, hence we want to find \\( p(\mathbf{x}) \\) (the marginal likelihood, or evidence) in order to
generate data that looks like real data distribution.

<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color: lightyellow;">
  Marginal Likelihood: Using the law of probability, we could find its relation with <span>\( \mathbf{z} \)</span>::
  $$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x | z})p_\theta(\mathbf{z})d\mathbf{z}$$
  or,
  $$\textcolor{blue}{\log p(\mathbf{x}) = \log \int p(\mathbf{x | z})p(\mathbf{z})d\mathbf{z}}$$
  That is, we marginalize out <span>\( \mathbf{z} \)</span> from the joint probability distribution <span>\( p(\mathbf{x, z}) \)</span>.
</div>

Instead of maximizing \\( p_\theta(\mathbf{x}) \\), we maximize
\\( \log p_\theta(\mathbf{x}) \\), since both are always increasing functions.
[We need to find \\( \theta^* \\) that maximizes the
\\( \log p_\theta(\mathbf{x}) \\)].

For example, if \\( p_\theta(\mathbf{x})$ = $10^{-50} \\), the
\\( \log p_\theta(\mathbf{x}) \\) = -115.13, and this is much easier to work
with and also log transforms the multiplications into addition.

<!-- ---------------------------------------- -->
<h2 id="kl-divergence">4. Kullback-Leibler Divergence</h2>

<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color: coral;">
<strong>Intractable problem</strong>
<ol>
    <li>We cannot find solution of <span> \( p(\mathbf{x | z}) \) </span> due to huge
        computational demand to sum over all the possible value of
        <span>\( \mathbf{z} \)</span> in above equation, thus making this problem
        intractable.</li>

<li>Therefore, we need to approximate this with other probablity density
        function <span> \( q_\phi(\mathbf{x} | \mathbf{z}) \) </span> known as approximate
        posterior distribution</li>
</div>

We introduce a new term in the loss function called the Kullback-Leibler
(KL) divergence \\( D_{KL} \\) and we call this method **variation inference**

<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color: lightyellow;">
KL divergence <span> \( D_{KL} \) </span> measures how much information is lost if the
distribution <span> \(P(z|X) \) </span>  is used to represent <span> \( Q(z|X) \) </span> and is expressed as:
$$D_{KL} [Q(z|X) \parallel P(z|X)] = \sum_z Q(z|X) \log \frac{Q(z|X)}{P(z|X)}$$
</div>

<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color:#f9f9f9;">
  <p><strong>Kullback-Leibler divergence:</strong> Let's now expand the KL term:</p>
  $$
  \begin{aligned}
  D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z} | \mathbf{x})) 
  &= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z} | \mathbf{x})} \, d\mathbf{z} \\
  &= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x}) \cdot p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} \, d\mathbf{z}
  \quad \text{[Because } p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{z}, \mathbf{x})}{p(\mathbf{x})} \text{]} \\
  &= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \left( \log p_{\theta}(\mathbf{x}) + \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} \right) d\mathbf{z} \\
  &= \log p_{\theta}(\mathbf{x}) + \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} \, d\mathbf{z}
  \quad \text{[Since } \int q(\mathbf{z} | \mathbf{x}) d\mathbf{z} = 1 \text{]} \\
  &= \log p_{\theta}(\mathbf{x}) + \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{x} | \mathbf{z}) \cdot p_{\theta}(\mathbf{z})} \, d\mathbf{z}
  \quad \text{[Since } p(\mathbf{z}, \mathbf{x}) = p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) \text{]} \\
  &= \log p_{\theta}(\mathbf{x}) + \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z})} - \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]
  \end{aligned}
  $$


  <p><strong>Kullback-Leibler divergence:</strong></p>
  $$
  \begin{aligned}
  D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z} | \mathbf{x}))
  &= \log p_{\theta}(\mathbf{x}) + D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z})) \\
  &\quad - \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]
  \end{aligned}
  $$

  <p>Once we rearrange the left and right-hand sides of the equation, we obtain:</p>
  $$
  \begin{aligned}
  \log p_{\theta}(\mathbf{x}) &= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]
  - D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z})) \\
  &\quad + D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z} | \mathbf{x}))
  \end{aligned}
  $$

  <p>Since the last KL term is always \( \geq 0 \), we obtain the ELBO inequality:</p>
  $$
  \log p_{\theta}(\mathbf{x}) \geq \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]
  - D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) \parallel p_{\theta}(\mathbf{z}))
  $$

  <p><strong>Interpretation:</strong> Instead of maximizing the left-hand side (which is intractable), we maximize the right-hand side.</p>

  <p><strong>Conclusion:</strong> Maximizing the likelihood \( \log p_{\theta}(\mathbf{x}) \) is equivalent to minimizing the KL divergence and maximizing the expected log-likelihood.</p>

  <p>In variational Bayesian methods, this right-hand side is known as the <strong>variational lower bound</strong>, or the <strong>evidence lower bound (ELBO)</strong>:</p>
  $$
  \mathcal{L}(\theta, \phi, \mathbf{x}, \mathbf{z})
  $$
</div>



<h2 id="loss-function">5. Loss Function</h2>
<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color:#f9f9f9;">


<p><strong>First Term in the ELBO:</strong> To derive the final VAE loss function, we begin by assuming the likelihood of the data \( \mathbf{x} \) given the latent variable \( \mathbf{z} \) is Gaussian:</p>

$$
  p(\mathbf{x} | \mathbf{z}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp \left(-\frac{1}{2\sigma^2} \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right)
  $$

  <p>where:</p>
  <ol>
    <li>\( \hat{\mathbf{x}} \) is the reconstruction produced by the decoder</li>
    <li>\( \sigma^2 \) is the variance of the Gaussian likelihood (usually fixed to 1 or learned)</li>
  </ol>


  <p>Taking the logarithm of the likelihood:</p>
  $$
  \log p(\mathbf{x} | \mathbf{z}) = - \frac{1}{2} \left[ d \log(2\pi\sigma^2) + \frac{1}{\sigma^2} \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right]
  $$

  <p>where \( d \) is the dimensionality of \( \mathbf{x} \).</p>



  <p>This log-likelihood contains two components:</p>

  <ol>
    <li><strong>Constant term:</strong> \( - \frac{d}{2} \log(2\pi\sigma^2) \), which is independent of the data and doesn’t influence optimization</li>
    <li><strong>Quadratic term:</strong> \( - \frac{1}{2\sigma^2} \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \), which corresponds to squared reconstruction error</li>
  </ol>

  <p>The first term in the ELBO is the expected log-likelihood under the approximate posterior:</p>
  $$
    \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log p(\mathbf{x} | \mathbf{z}) \right]
  $$

  <p>Substituting the log-likelihood expression:</p>
  $$
  \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[
    - \frac{1}{2} \left( d \log(2\pi\sigma^2) + \frac{1}{\sigma^2} \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right)
  \right]
  $$

  <p>Splitting the expectation over the constant and variable terms:</p>
  $$
  - \frac{d}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \, \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right]
  $$

  <p>Since the first term is constant, it does not affect the gradients. Therefore, <strong>maximizing the ELBO</strong> is equivalent to <strong>minimizing the expected reconstruction error</strong>:</p>
  $$
  \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right]
  $$

  <p>The reconstruction loss term in the ELBO is:</p>
  $$
  \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \| \mathbf{x} - \hat{\mathbf{x}} \|^2 \right]
  = -\frac{1}{N} \sum_{i=1}^N \| \mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)} \|^2
  $$

  <p>This is exactly the <strong>mean squared error (MSE)</strong>. In practice, we use Monte Carlo sampling to estimate this expectation.</p>

</div>
<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color:#f9f9f9;">

  <p><strong>KL Loss and the Second Term in the ELBO:</strong></p>


  <p>Now, let's consider the <strong>second term</strong> in the ELBO, the KL divergence. Assume:</p>

  <ol>
    <li><strong>Approximate posterior:</strong> \( q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mu_z, \sigma^2_z I) \)</li>
    <li><strong>Prior:</strong> \( p(\mathbf{z}) = \mathcal{N}(0, I) \)</li>
  </ol>

  <p>This allows us to use the closed-form solution for the KL divergence between two multivariate Gaussians. The KL divergence is defined as:</p>
  $$
  D_{KL}(q(\mathbf{z} | \mathbf{x}) \parallel p(\mathbf{z})) =
  \mathbb{E}_{q(\mathbf{z} | \mathbf{x})} \left[
    \log \frac{q(\mathbf{z} | \mathbf{x})}{p(\mathbf{z})}
  \right]
  $$

  <p><strong>Step 1: Gaussian PDFs</strong></p>

  <p>For a multivariate Gaussian:</p>
  $$
  p(\mathbf{z}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{z} - \mu)^T \Sigma^{-1} (\mathbf{z} - \mu) \right)
  $$

  <p>With \( \Sigma_p = I \) and \( \mu_p = 0 \), the prior becomes:</p>
  $$
  p(\mathbf{z}) = \frac{1}{(2\pi)^{d/2}} \exp \left( -\frac{1}{2} \mathbf{z}^T \mathbf{z} \right)
  $$

  <p>Similarly, the approximate posterior is:</p>
  $$
  q(\mathbf{z} | \mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma_q|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{z} - \mu_z)^T \Sigma_q^{-1} (\mathbf{z} - \mu_z) \right)
  $$

  <p><strong>Step 2: Log-Ratio</strong></p>
  $$
  \log \frac{q(\mathbf{z} | \mathbf{x})}{p(\mathbf{z})} =
  -\frac{1}{2} (\mathbf{z} - \mu_z)^T \Sigma_q^{-1} (\mathbf{z} - \mu_z)
  + \frac{1}{2} \mathbf{z}^T \mathbf{z}
  - \frac{1}{2} \log |\Sigma_q|
  $$

  <p>Since \( \Sigma_q = \mathrm{diag}(\sigma_z^2) \), we get:</p>
  $$
  \log |\Sigma_q| = \sum_{i=1}^d \log \sigma_{z,i}^2
  $$

  <p><strong>Step 3: Expectation</strong></p>
  $$
  \begin{aligned}
  D_{KL} &= \mathbb{E}_{q(\mathbf{z} | \mathbf{x})} \left[
    -\frac{1}{2} (\mathbf{z} - \mu_z)^T \Sigma_q^{-1} (\mathbf{z} - \mu_z)
    + \frac{1}{2} \mathbf{z}^T \mathbf{z}
    - \frac{1}{2} \log |\Sigma_q|
  \right]
  \end{aligned}
  $$

  <p>Using the identities:</p>
  <ol>
  <li><span>\( \mathbb{E}_{q} [(\mathbf{z} - \mu_z)^T \Sigma_q^{-1} (\mathbf{z} - \mu_z)] = \mathrm{Tr}(I) = d \) </span> </li>
  <li><span>\( \mathbb{E}_{q}[\mathbf{z}^T \mathbf{z}] = \mu_z^T \mu_z + \sum_{i=1}^d \sigma_{z,i}^2 \) </span></li>
</ol>
  <p><strong>Final closed-form KL divergence:</strong></p>
  $$
  D_{KL}(q(\mathbf{z} | \mathbf{x}) \parallel p(\mathbf{z})) =
  \frac{1}{2} \sum_{i=1}^d \left( \sigma_{z,i}^2 + \mu_{z,i}^2 - 1 - \log \sigma_{z,i}^2 \right)
  $$

  <p><strong>Total Loss — The ELBO:</strong></p>

  <p>The Evidence Lower Bound (ELBO) for a Variational Autoencoder is given by:</p>
  $$
  \begin{aligned}
  \mathcal{L}(\theta, \phi, \mathbf{x}, \mathbf{z}) &=
  \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})}
  \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) \right]
  - D_{KL}\left( q_\phi(\mathbf{z} | \mathbf{x}) \parallel p_\theta(\mathbf{z}) \right)
  \end{aligned}
  $$

  <p>Substituting the reconstruction loss (MSE) and KL divergence from earlier steps, we obtain:</p>
  $$
  \begin{aligned}
  \mathcal{L}(\theta, \phi, \mathbf{x}, \mathbf{z}) &=
  - \frac{1}{N} \sum_{i=1}^N \| \mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)} \|^2
  - \frac{1}{2} \sum_{j=1}^d \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
  \end{aligned}
  $$

  <p><strong>VAE Loss Function:</strong></p>

  <p>The negative ELBO defines the loss function to minimize:</p>
  $$
  \begin{aligned}
  L_{\text{VAE}}(\theta, \phi) &=
  -\mathcal{L}(\theta, \phi, \mathbf{x}, \mathbf{z}) \\
  &=
  \frac{1}{N} \sum_{i=1}^N \| \mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)} \|^2
  + \frac{1}{2} \sum_{j=1}^d \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
  \end{aligned}
  $$

  <p>Therefore, by minimizing this loss, we are equivalently maximizing the lower bound of the probability of generating real data samples.</p>

  <p><strong>Optimization Goal:</strong></p>
  $$
  \theta^*, \phi^* = \arg \min_{\theta, \phi} L_{\text{VAE}}(\theta, \phi)
  $$

</div>


<h2 id="reparameterization">6. Reparameterization Trick</h2>
<ol>
<li>   The expectation term in the loss function invokes generating samples
    from  <span> \( E_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \) </span></li>

<li>   Sampling is a stochastic process and therefore we cannot
    backpropagate the gradient.</li>

<li>   Using Reparameterization trick: it is possible to express the random
    variables <span> \( \mathbf{z} \) </span> as a deterministic variable
    <span> \( \mathbf{z} \sim q_\phi( \mathbf{z|x^{(i)} }) = \mathcal{N}(\mathbf{z}; \mathbf{\mu^{(i)}, \sigma^{2(i)}I}) \) </span></li>

<li>
  <span>\( \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \)</span>,
  where <span>\( \odot \)</span> denotes the element-wise (Hadamard) product.
</li>


<li>   VAEs are useful for generating new, similar data based on learned
    distributions </li>
</ol>


<img src="assets/blogs/vae_resources/reparameterization-trick.png" alt="Illustration of how the reparameterization trick makes the sampling process trainable" style="width:80%; display:block; margin:auto;">


<h2 id="generator">7. Generator</h2>

<div style="border:3px solid #ccc; padding:20px; border-radius:10px; background-color:#f9f9f9;">

  <p><strong>Latent Space Visualization:</strong></p>

  <p>After training the VAE model, we can visualize the learned latent space. This helps us understand how the model organizes and encodes input data in a lower-dimensional representation.</p>

  <img src="assets/blogs/vae_resources/LatentSpace.png" alt="Latent space visualization" style="width:80%; display:block; margin:auto; margin-top:10px; margin-bottom:10px;">

  <p><strong>Data Generation Process:</strong></p>

  <ol>
  <li>
      Sample a latent vector \( \mathbf{z}^{(i)} \) from the prior distribution:
      $$
      p(\mathbf{z}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
      $$
    </li>

<li>
      Generate a sample \( \mathbf{x}^{(i)} \) from the decoder using the conditional likelihood:
      $$
      p_\theta(\mathbf{x} \mid \mathbf{z} = \mathbf{z}^{(i)})
      $$
    </li>

<li>
      Continuously vary the components of the latent vector \( \mathbf{z} \) and observe how the output \( \hat{\mathbf{x}} \) changes.
      This reveals how different dimensions in latent space control generative factors.
    </li>
  </ol>

</div>

**The End**
