::: frame
:::

::: frame
Overview
:::

# Introduction

::: frame
Introduction to VAE

-   A Variational Autoencoder (VAE) is a generative model

-   It learns to encode input data into a lower-dimensional space

-   VAE maximizes the likelihood of the data while introducing
    regularization

-   The encoder and decoder are trained together using backpropagation

-   VAEs are useful for generating new, similar data based on learned
    distributions
:::

::: frame
Architecture

![Architecture of VAE](images/VAE.png){#vae-architecture
width="0.8\\linewidth"}
:::

::: frame
Basic Terminology

::: theorem
$$p_\theta(\mathbf{z} | \mathbf{x}) = \dfrac{p_\theta(\mathbf{x} | \mathbf{z}) p_\theta(\mathbf{z})}{p_\theta(\mathbf{x})}$$
:::

::: columns
**Terminology**

1.  **Prior**: $p_\theta(\mathbf{z})$

2.  **Posterior**: $p_\theta(\mathbf{z} | \mathbf{x})$

3.  **Likelihood**: $p_\theta(\mathbf{x} | \mathbf{z})$

4.  **Marginal Likelihood**: $p_\theta(\mathbf{x})$

::: examples
-   $\mathbf{x}$: data that we want to model a.k.a the animal.

-   $\mathbf{z}$: latent variable a.k.a our imagination.

-   $p(\mathbf{x})$: probability distribution of the data, i.e. that
    animal kingdom.

-   $p(\mathbf{z})$: probability distribution of latent variable, i.e.
    our brain, the source of our imagination.

-   $p(\mathbf{x} | \mathbf{z})$: distribution of generating data given
    latent variable, e.g. turning imagination into real animal.
:::
:::
:::

# Objective

::: frame
Objective Our goal is to model the data, hence we want to find
$p(\mathbf{x})$ (the marginal likelihood, or evidence) in order to
generate data that looks like real data distribution.

::: block
Marginal Likelihood Using the law of probability, we could find its
relation with $\mathbf{z}$:
$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x | z})p_\theta(\mathbf{z})d\mathbf{z}$$
or,
$\textcolor{blue}{\log p(\mathbf{x}) = \log \int p(\mathbf{x | z})p(\mathbf{z})d\mathbf{z}}$
that is marginalize out $\mathbf{z}$ from the join probability
distribution $p(\mathbf{x, z})$.
:::

Instead of maximizing $p_\theta(\mathbf{x})$, we maximize
$\log p_\theta(\mathbf{x})$, since both are always increasing functions.
[We need to find $\theta^*$ that maximizes the
$\log p_\theta(\mathbf{x})$]{style="color: red"}.\

For example, if $p_\theta(\mathbf{x})$ = $10^{-50}$, the
$\log p_\theta(\mathbf{x})$ = -115.13, and this is much easier to work
with and also log transforms the multiplications into addition.
:::

::: frame
Kullback-Leibler divergence

::: alertblock
Intractable problem

-   We cannot find solution of $p(\mathbf{x | z})$ due to huge
    computational demand to sum over all the possible value of
    $\mathbf{z}$ in above equation, thus making this problem
    intractable.

-   Therefore, we need to approximate this with other probablity density
    function $q_\phi(\mathbf{x} | \mathbf{z})$ known as approximate
    posterior distribution
:::

We introduce a new term in the loss function called the Kullback-Leibler
(KL) divergence $D_{KL}$ and we call this method \"variation inference\"

::: theorem
KL divergence $D_{KL}$ measures how much information is lost if the
distribution $P(z|X)$ is used to represent $Q(z|X)$ and is expressed as:
$$D_{KL} [Q(z|X) \parallel P(z|X)] = \sum_z Q(z|X) \log \frac{Q(z|X)}{P(z|X)}$$
:::
:::

::: frame
Kullback-Leibler divergence Let's now expand the KL term:
$$\begin{aligned}
D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z} | \mathbf{x})) 
&= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z} | \mathbf{x})} d\mathbf{z} \\
&= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x}) p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} d\mathbf{z} \quad \text{; \textcolor{red}{Because}  } p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{z}, \mathbf{x})}{p(\mathbf{x})} \\
&= \int q_{\phi}(\mathbf{z} | \mathbf{x}) \left( \log p_{\theta}(\mathbf{x}) + \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} \right) d\mathbf{z} \\
&= \log p_{\theta}(\mathbf{x}) + \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} d\mathbf{z} \quad \text{; \textcolor{red}{Because} } \int q(\mathbf{z} | \mathbf{x}) d\mathbf{z} = 1 \\
&  \hspace{-2cm}= \log p_{\theta}(\mathbf{x}) + \int q_{\phi}(\mathbf{z} | \mathbf{x}) \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{x} | \mathbf{z}) p_{\theta}(\mathbf{z})} d\mathbf{z} \quad \text{; \textcolor{red}{Because} } p(\mathbf{z}, \mathbf{x}) = p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})\\
&= \log p_{\theta}(\mathbf{x}) + \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \log \frac{q_{\phi}(\mathbf{z} | \mathbf{x})}{p_{\theta}(\mathbf{z})} - \log p_{\theta}(\mathbf{x} | \mathbf{z}) \right]
\end{aligned}$$
:::

::: frame
Kullback-Leibler divergence $$\begin{aligned}
 D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z} | \mathbf{x})) &= \log p_{\theta}(\mathbf{x}) + D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z})) - \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})
\end{aligned}$$ Once we rearrange the left and right-hand sides of the
equation, we obtain: $$\begin{aligned}
\log p_{\theta}(\mathbf{x})  &= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z}) - D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z})) +  D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z} | \mathbf{x}))
\end{aligned}$$

KL term is $\ge0$, so:

$$\begin{aligned}
\log p_{\theta}(\mathbf{x})  &\ge \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z}) - D_{KL}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p_{\theta}(\mathbf{z})) 
\end{aligned}$$ Instead of maximizing LHS, we can maximize RHS.\

**Maximizing likelihood $\log p_{\theta}(\mathbf{x})$ is equivalent to
minimizing KL-Divergence and maximizing log-likelihood
($\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})$)**.\

In Variational Bayesian methods, this RHS
($\mathcal{L(\theta, \phi, \mathbf{x}, \mathbf{z})}$) is known as the
*variational lower bound*, or *evidence lower bound* **ELBO**.
:::

# Loss function

::: frame
First term in ELBO Now, deriving the final VAE loss function, let's
assume the likelihood $$\begin{aligned}
p(\mathbf{x} | \mathbf{z}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp \left(-\frac{1}{2\sigma^2}||\mathbf{x} - \hat{\mathbf{x}}||^2 \right)
    % p(\mathbf{x} | \mathbf{z}) &= \mathcal{N}(\hat{\mathbf{x}}, \sigma^2I) \\
    % &= \dfrac{1}{2\pi\sigma^2}\exp \left(-\dfrac{1}{2\sigma^2}||\mathbf{x} - \hat{\mathbf{x}}||^2 \right)
\end{aligned}$$

where

1.  $\hat{\mathbf{x}}$ is the reconstructed data,

2.  $\sigma^2$ is the variance of the Gaussian likelihood (often assumed
    to be 1 or learned)

Taking log:
$$\log  p(\mathbf{x} | \mathbf{z}) = - \dfrac{1}{2}\left[d \log (2\pi \sigma^2) + \dfrac{1}{\sigma^2} || \mathbf{x} - \hat{\mathbf{x}}||^2\right]$$
where $d$ is the dimensionality of $\mathbf{x}$.
:::

::: frame
First term in ELBO

The equation consists of two terms:

1.  **Constant Term:** $- \dfrac{d}{2} \log (2\pi \sigma^2)$
    (independent of $\mathbf{x}$, so doesn't effect optimization.)

2.  **Quadratic term**:
    $-\dfrac{1}{2\sigma^2} || \mathbf{x} - \hat{\mathbf{x}}||^2$, which
    is squared error.

So, the first part in ELBO is:
$$\mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} [\log p(\mathbf{x} | \mathbf{z})]$$
Substituting value of $\log p(\mathbf{x} | \mathbf{z})$:
$$\mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[- \dfrac{1}{2}\left[d \log (2\pi \sigma^2) + \dfrac{1}{\sigma^2} || \mathbf{x} - \hat{\mathbf{x}}||^2\right]\right]$$
Splitting the expectations:
$$- \dfrac{d}{2}  \log (2\pi \sigma^2) - \dfrac{1}{2\sigma^2} \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ || \mathbf{x} - \hat{\mathbf{x}}||^2 \right]$$
The first term is constant and does not contribute to gradient updates,
so it can be ignored. Thus, **maximizing ELBO** is equivalent to
**minimizing the expected squared error:**
:::

::: frame
KL Loss
$$\mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ || \mathbf{x} - \hat{\mathbf{x}}||^2 \right] = -\dfrac{1}{N}\sum_{i=1}^N (||\mathbf{x} - \hat{\mathbf{x}}||^2)$$
which is exactly the **Mean squared error**. [No Gaussian distribution,
use Monte-Carlo for integration]{style="color: red"}\
Now, let's move to the second term in the ELBO which is KL Divergence
term. If we assume

1.  **Approximate posterior:**
    $q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mu_z, \sigma^2_zI)$
    [use the reparameterization trick to make the latent variable
    differentiable]{style="color: red"}

2.  **Prior:** $p (\mathbf{z}) = \mathcal{N}(0, I)$

due to [Central limit theorem]{style="color: red"} and [closed form
solution]{style="color: red"} possible between gaussian distribution.\
The KL divergence is defined as:
$$D_{KL}(q(z|x) \parallel p(z)) = \mathbb{E}_{q(z|x)} \left[ \log \frac{q(z|x)}{p(z)} \right]$$
:::

::: frame
KL Loss

Expanding using the probability density functions of Gaussians:

$$D_{KL}(\mathcal{N}(\mu_z, \sigma_z^2 I) \parallel \mathcal{N}(0, I)) = \int q(z|x) \log \frac{q(z|x)}{p(z)} dz$$

**Step 1: Write the Gaussian PDFs**

The probability density function (PDF) of a multivariate Gaussian
$\mathcal{N}(\mu, \Sigma)$ is:

$$p(z) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (z - \mu)^T \Sigma^{-1} (z - \mu) \right)$$

For $q(z|x)$:

$$q(z|x) = \frac{1}{(2\pi)^{d/2} |\Sigma_q|^{1/2}} \exp \left( -\frac{1}{2} (z - \mu_z)^T \Sigma_q^{-1} (z - \mu_z) \right)$$
:::

::: frame
KL Loss

For $p(z)$ (where $\Sigma_p = I$ and $\mu_p = 0$):

$$p(z) = \frac{1}{(2\pi)^{d/2} |I|^{1/2}} \exp \left( -\frac{1}{2} z^T z \right)$$

**Step 2: Compute the Log Ratio** $\log \frac{q(z|x)}{p(z)}$

Taking the logarithm:

$$\log \frac{q(z|x)}{p(z)} = \log \frac{\exp \left( -\frac{1}{2} (z - \mu_z)^T \Sigma_q^{-1} (z - \mu_z) \right) / |\Sigma_q|^{1/2}}{\exp \left( -\frac{1}{2} z^T z \right)}$$

Breaking it down:

-   Exponential term difference:
    $$-\frac{1}{2} (z - \mu_z)^T \Sigma_q^{-1} (z - \mu_z) + \frac{1}{2} z^T z$$
:::

::: frame
KL Loss

-   Log determinant difference: $$-\frac{1}{2} \log |\Sigma_q|$$

Since $\Sigma_q = \text{diag}(\sigma_z^2)$, we have:

$$\log |\Sigma_q| = \sum_{i=1}^{d} \log \sigma_{z,i}^2$$

**Step 3: Compute the Expectation Over** $q(z|x)$

Now, taking expectation $\mathbb{E}_{q(z|x)}$ over $z$, we get:

$$D_{KL} = \mathbb{E}_{q(z|x)} \left[ -\frac{1}{2} (z - \mu_z)^T \Sigma_q^{-1} (z - \mu_z) + \frac{1}{2} z^T z - \frac{1}{2} \log |\Sigma_q| \right]$$

Using the expectation property
$\mathbb{E}_{q} [(z - \mu_z)(z - \mu_z)^T] = \Sigma_q$, the first term
simplifies to:

$$\mathbb{E}_{q(z|x)} \left[ (z - \mu_z)^T \Sigma_q^{-1} (z - \mu_z) \right] = \text{Tr}(I) = d$$
:::

::: frame
KL Loss

For the second term:

$$\mathbb{E}_{q(z|x)} [z^T z] = \mathbb{E}_{q(z|x)} \left[ (\mu_z + \sigma_z \epsilon)^T (\mu_z + \sigma_z \epsilon) \right]$$

Expanding,

$$= \mu_z^T \mu_z + 2\mathbb{E}[\epsilon^T \sigma_z \mu_z] + \mathbb{E}[\epsilon^T \sigma_z^2 \epsilon]$$

Since $\mathbb{E}[\epsilon] = 0$ and
$\mathbb{E}[\epsilon^T \epsilon] = d$, we get:

$$\mathbb{E}_{q(z|x)} [z^T z] = \mu_z^T \mu_z + \sum_{i=1}^{d} \sigma_{z,i}^2$$

Thus, the KL divergence simplifies to:

$$D_{KL} = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_{z,i}^2 + \mu_{z,i}^2 - 1 - \log \sigma_{z,i}^2 \right)$$
:::

::: frame
Total Loss So, the ELBO becomes: $$\begin{aligned}
    \mathcal{L(\theta, \phi, \mathbf{x}, \mathbf{z})} &=  E_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \left[ \ln{p_\theta(\mathbf{x} | \mathbf{z})}\right]- D_{KL}\left (q_\phi(\mathbf{z} | \mathbf{x}) || p_\theta(\mathbf{z})\right) \\
    &= - \dfrac{1}{N}\sum_{i=1}^N (||\mathbf{x} - \hat{\mathbf{x}}||^2) - \dfrac{1}{2}\sum_{j=1}^d(\mu_j^2 + \sigma_j^2-\log\sigma_j^2- 1)
\end{aligned}$$ The negation of the ELBO defines our loss function:

$$\begin{aligned}
L_{\text{VAE}}(\theta, \phi) &= -   \mathcal{L(\theta, \phi, \mathbf{x}, \mathbf{z})} =  \dfrac{1}{N}\sum_{i=1}^N (||\mathbf{x} - \hat{\mathbf{x}}||^2) + \dfrac{1}{2}\sum_{j=1}^d(\mu_j^2 + \sigma_j^2-\log\sigma_j^2- 1)
\end{aligned}$$ Therefore, by minimizing the loss, we are maximizing the
lower bound of the probability of generating real data samples.\
We just need to find the parameters
$\theta^*, \phi^* = \arg \min_{\theta, \phi} L_{\text{VAE}}$
:::

# Reparameterization trick

::: frame
Reparameterization trick

-   The expectation term in the loss function invokes generating samples
    from $E_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})}$

-   Sampling is a stochastic process and therefore we cannot
    backpropagate the gradient.

-   Using Reparameterization trick: it is possible to express the random
    variables $\mathbf{z}$ as a deterministic variable
    $\mathbf{z} \sim q_\phi( \mathbf{z|x^{(i)} }) = \mathcal{N}(\mathbf{z}; \mathbf{\mu^{(i)}, \sigma^{2(i)}I})$

-   $\mathbf{z} = \mathbf{\mu} + \mathbf{\sigma}  \mathbin{\mathpalette\pdot@\relax}\epsilon$,
    where $\mathbin{\mathpalette\pdot@\relax}$ refers to element-wise
    product.

-   VAEs are useful for generating new, similar data based on learned
    distributions
:::

::: frame
Reparameterization trick

![Illustration of how the reparameterization trick makes the sampling
process
trainable](images/reparameterization-trick.png){#vae-architecture
width="0.65\\linewidth"}
:::

# Generator

::: frame
Latent space After we trained our VAE model, we then could visualize the
latent variable space

::: columns
![[For animation, click
here.](https://giphy.com/gifs/vae-lqq0em9cuivVNWFwSX)](images/LatentSpace.png){width="0.99\\linewidth"}

::: block
Data generation

1.  First, sample a $\mathbf{z}^{(i)}$ from a prior distribution
    $p(\mathbf{z}) \sim \mathcal{N}(0, I)$.

2.  Then a value $\mathbf{x}^{(i)}$ is generated from a conditional
    distribution $p_\theta(\mathbf{x} | \mathbf{z} = \mathbf{z}^{(i)})$

3.  Continuously vary the parameters of $\mathbf{z}$ vector and analyze
    the reconstructed data.
:::
:::
:::

# Results

::: frame
**Thank you**

**The End**
:::
