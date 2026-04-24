this is what this project is about

basically the idea is to apply the k-armed-bandits problem, but instead of calling it like this, we are going to call it k-pizza-houses

suppose you are in a new city and you want to figure out where they make the best pizza. every time you order a pizza you assign a score to it (the score acts as reward)

each pizza house has a ground-truth probability distribution of scores that we are going to assume to be gaussian. 

$$r(k)=\mathcal N(\mu_k, \sigma_k)$$

Each different pizza house has a different $\mu_k$ and $\sigma_k$

we want to train a model that predicts the probability distribution for each pizza house.

For each pizza house $k$, let

$$
D_k = \{r_{k,1}, \dots, r_{k,n_k}\}
$$

be the set of observed scores collected so far.

We assume that scores from pizza house $k$ are generated from a Gaussian distribution with unknown parameters:

$$
r_{k,i} \mid \mu_k, \sigma_k^2 \sim \mathcal N(\mu_k, \sigma_k^2).
$$

Given the past observations $D_k$, the model does not directly estimate a single pair $(\mu_k, \sigma_k^2)$, but instead infers a posterior distribution over these latent parameters:

$$
p(\mu_k, \sigma_k^2 \mid D_k).
$$

This posterior captures our uncertainty about the true quality distribution of pizza house $k$.

From it, we can derive the posterior predictive distribution for the next score:

$$
p(r_{k,n_k+1} \mid D_k)=
\int p(r_{k,n_k+1} \mid \mu_k, \sigma_k^2)\, p(\mu_k, \sigma_k^2 \mid D_k)\, d\mu_k\, d\sigma_k^2.
$$

In other words, for each pizza house, the model uses past observations to infer a probability distribution over the house's Gaussian parameters, and therefore a predictive distribution over future scores.

# Objective

We want to maximise the reward, but also explore the various options to know where the best pizza is

The expected reward is quite easy to estimate, it's just the mean of the posterior predictive distribution:

$$
\mathbb E[r_{k,n_k+1} \mid D_k]=
\int r\, p(r \mid D_k)\, dr=
\mathbb E[\mu_k \mid D_k].
$$

The real challenge is balancing exploitation and exploration: we want to prefer pizza houses with high expected reward, while still sampling houses whose posterior remains uncertain.

# Uncertainty decomposition

At this point it is tempting to ask whether the log-likelihood of the next reward can be decomposed into:

1. an irreducible term coming from the intrinsic stochasticity of the reward;
2. a reducible term coming from uncertainty over the latent Gaussian parameters of the pizza house.

The right answer is: this decomposition does **not** hold in general for the log-likelihood of one realised reward sample.

Indeed, the posterior predictive density is

$$
p(r \mid D_k)=
\int p(r \mid \mu_k, \sigma_k^2)\, p(\mu_k, \sigma_k^2 \mid D_k)\, d\mu_k\, d\sigma_k^2,
$$

so the log predictive likelihood is

$$
\log p(r \mid D_k)=
\log \int p(r \mid \mu_k, \sigma_k^2)\, p(\mu_k, \sigma_k^2 \mid D_k)\, d\mu_k\, d\sigma_k^2.
$$

Because the logarithm is outside the integral, this quantity does not split cleanly into an additive "irreducible + reducible" form.

What **does** admit a clean decomposition is the predictive uncertainty, measured for example by the entropy of the posterior predictive distribution:

$$
H(r \mid D_k)=
\mathbb E_{(\mu_k,\sigma_k^2)\mid D_k}\big[H(r \mid \mu_k,\sigma_k^2)\big]
+
I\big(r;(\mu_k,\sigma_k^2)\mid D_k\big).
$$

This gives two conceptually different sources of uncertainty:

- The term

$$
\mathbb E_{(\mu_k,\sigma_k^2)\mid D_k}\big[H(r \mid \mu_k,\sigma_k^2)\big]
$$

is the **aleatoric** or **irreducible** uncertainty. It is the randomness in the pizza score that would remain even if we knew the true parameters of the pizza house.

- The term

$$
I\big(r;(\mu_k,\sigma_k^2)\mid D_k\big)
$$

is the **epistemic** or **reducible** uncertainty. It comes from not yet knowing the true values of $\mu_k$ and $\sigma_k^2$. As more scores are collected from house $k$, this term decreases.

For a Gaussian likelihood with known parameters,

$$
H(r \mid \mu_k,\sigma_k^2)=
\frac{1}{2}\log\big(2\pi e\,\sigma_k^2\big),
$$

so the irreducible part depends on the intrinsic variance of the pizza house.

A closely related decomposition holds for the posterior predictive variance:

$$
\mathrm{Var}(r \mid D_k)=
\mathbb E[\sigma_k^2 \mid D_k]
+
\mathrm{Var}(\mu_k \mid D_k).
$$

This is often the most intuitive way to think about the exploration problem:

- $\mathbb E[\sigma_k^2 \mid D_k]$ is the average within-house noise in pizza ratings;
- $\mathrm{Var}(\mu_k \mid D_k)$ is uncertainty about the house's true average quality.

As the model learns, the second term goes down, while the first one generally does not. This is exactly the distinction between uncertainty that can be reduced by gathering more data and uncertainty that is intrinsic to the reward-generating process.

# Probability of being the best pizza house

The quantity we ultimately care about is not the next noisy reward itself, but which pizza house has the largest **latent mean reward**.

For each house $k$, we can first marginalise out the uncertainty over the variance and obtain the posterior distribution of the mean:

$$
p(\mu_k \mid D_k)=
\int p(\mu_k, \sigma_k^2 \mid D_k)\, d\sigma_k^2.
$$

This posterior describes our belief about the true average quality of pizza house $k$.

Now let

$$
k^\star = \arg\max_{j \in \{1,\dots,K\}} \mu_j
$$

denote the index of the truly best pizza house, that is, the house with the largest latent mean reward.

The posterior probability that house $k$ is the best is then

$$
\mathbb P(k^\star = k \mid D)=
\mathbb P(\mu_k \ge \mu_j \ \forall j \neq k \mid D),
$$

where $D = (D_1,\dots,D_K)$ denotes all observations collected so far.

Equivalently, this can be written as

$$
\mathbb P(k^\star = k \mid D)=
\int \mathbf 1\!\{\mu_k = \max_{j} \mu_j\}
\; p(\mu_1,\dots,\mu_K \mid D)\;
d\mu_1 \cdots d\mu_K.
$$

If the posterior factorises across houses, this becomes

$$
p(\mu_1,\dots,\mu_K \mid D)=
\prod_{j=1}^{K} p(\mu_j \mid D_j),
$$

and therefore

$$
\mathbb P(k^\star = k \mid D)=
\int \mathbf 1\!\{\mu_k = \max_{j} \mu_j\}
\prod_{j=1}^{K} p(\mu_j \mid D_j)\,
d\mu_1 \cdots d\mu_K.
$$

This is a best-focused quantity: it does not ask whether house $k$ can generate a high reward on a single noisy draw, but whether its **true mean quality** is the highest among all houses.

In practice, this probability can be estimated by Monte Carlo:

1. for each posterior sample $s = 1,\dots,S$, draw

$$
\mu_j^{(s)} \sim p(\mu_j \mid D_j)
\qquad \text{for all } j \in \{1,\dots,K\};
$$

2. identify the winning house in that sample,

$$
k^{\star (s)} = \arg\max_j \mu_j^{(s)};
$$

3. estimate the probability of being best as

$$
\widehat{\mathbb P}(k^\star = k \mid D)=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf 1\!\{k^{\star (s)} = k\}.
$$

The current best recommendation is then naturally

$$
\hat{k}=
\arg\max_k \mathbb P(k^\star = k \mid D).
$$

This gives a clean objective for best-house identification: recommend the pizza house with the highest posterior probability of being the true best.

# 1D quadrature instead of Monte Carlo

The Monte Carlo estimator above is simple and general, but in this setting we can often avoid sampling altogether.

If the posterior over latent means factorises across houses, then the probability that house $k$ is the best can be reduced to a **one-dimensional integral**:

$$
\mathbb P(k^\star = k \mid D)=
\int p(\mu_k = x \mid D_k)\;
\prod_{j \neq k} \mathbb P(\mu_j \le x \mid D_j)\; dx.
$$

Writing

$$
f_k(x) = p(\mu_k = x \mid D_k),
\qquad
F_j(x) = \mathbb P(\mu_j \le x \mid D_j),
$$

this becomes

$$
\mathbb P(k^\star = k \mid D)=
\int f_k(x)\prod_{j\neq k} F_j(x)\,dx.
$$

This is called a 1D quadrature formulation because, instead of sampling many posterior worlds, we only need to numerically integrate over a single scalar variable $x$, interpreted as a candidate value for the latent mean of house $k$.

This formulation relies on the following assumptions:

1. the quantity used to rank houses is a single scalar latent variable, here the true mean reward $\mu_k$;
2. each house has a posterior distribution over that scalar, so we can evaluate its density $f_k$ and cumulative distribution function $F_k$;
3. the joint posterior over houses factorises across houses:

$$
p(\mu_1,\dots,\mu_K \mid D)=
\prod_{j=1}^{K} p(\mu_j \mid D_j);
$$

4. the posterior distributions are continuous, so ties occur with probability zero.

Under these assumptions, the interpretation is simple:

- choose a possible value $x$ for the latent mean of house $k$;
- weight it by how plausible that value is under the posterior of house $k$;
- multiply by the probability that every other house has latent mean below $x$;
- integrate over all possible values of $x$.

Importantly, this does **not** require the uncertainty to be isotropic or the variances to be equal across houses. Each house may have a different posterior spread.

For example, if $p(\mu_k \mid D_k)$ is Gaussian or Student-$t$, the 1D quadrature formula still applies. In general, this does not lead to a closed-form expression for $K > 2$, but it remains a deterministic one-dimensional numerical integration problem rather than a Monte Carlo estimation problem.

By contrast, some special families such as independent Gumbel posteriors with a **common** scale parameter do admit a closed-form expression for the probability of being best. However, once the scale differs across houses, that simplification no longer holds. For the pizza-house model, keeping the natural Gaussian or Student-$t$ posteriors and using 1D quadrature is typically the most faithful approach.
