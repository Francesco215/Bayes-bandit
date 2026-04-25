# Mathematical Details

This note derives the Bayesian model used by the Gaussian k-armed bandit, the posterior updates, the predictive distributions, the probability-of-being-best objective, and how the resulting policy relates to UCB and Thompson sampling.

## 1. Bandit Model

There are $K$ arms. At time $t$, the agent chooses one arm:

$$
A_t \in \{1,\dots,K\}
$$

and observes a scalar reward:

$$
R_t \in \mathbb R.
$$

Each arm has its own latent Gaussian reward distribution:

$$
R_t \mid A_t=k,\mu_k,\sigma_k^2
\sim
\mathcal N(\mu_k,\sigma_k^2).
$$

The latent mean $\mu_k$ is the true expected reward of arm $k$. The latent variance $\sigma_k^2$ is the intrinsic reward noise for that arm.

The environment used in the parameter study samples the true values independently:

$$
\mu_k \sim \mathcal N(0,\texttt{mean scale}^2),
$$

and

$$
\sigma_k \sim \mathrm{Uniform}(\texttt{min sigma},\texttt{max sigma}).
$$

The algorithms do not observe $\mu_k$ or $\sigma_k$. They only observe rewards from selected arms.

## 2. Sufficient Statistics

For a fixed arm $k$, suppose we have observed:

$$
D_k = \{r_{k,1},\dots,r_{k,n_k}\}.
$$

The implementation stores the following sufficient statistics:

$$
n_k = |D_k|,
$$

$$
s_k = \sum_{i=1}^{n_k} r_{k,i},
$$

$$
q_k = \sum_{i=1}^{n_k} r_{k,i}^2.
$$

The sample mean is:

$$
\bar r_k = \frac{s_k}{n_k}
$$

when $n_k>0$. The centered sum of squares is:

$$
S_k = \sum_{i=1}^{n_k}(r_{k,i}-\bar r_k)^2
= q_k - \frac{s_k^2}{n_k}.
$$

The code uses $n_k$, $s_k$, and $q_k$ directly, because they can be updated online in constant time.

## 3. Normal-Inverse-Gamma Prior

For each arm, the Bayesian model uses a Normal-Inverse-Gamma prior:

$$
\sigma_k^2 \sim \mathrm{InvGamma}(\alpha_0,\beta_0),
$$

$$
\mu_k \mid \sigma_k^2
\sim
\mathcal N\left(\mu_0,\frac{\sigma_k^2}{\kappa_0}\right).
$$

The prior hyperparameters are:

| Symbol | Config name | Meaning |
|---|---|---|
| $\mu_0$ | `prior.mu` or fixed `0.0` in study | Prior center for the mean. |
| $\kappa_0$ | `kappa` | Prior strength on the mean. |
| $\alpha_0$ | `alpha` | Prior shape for the variance. |
| $\beta_0$ | `beta` | Prior scale for the variance. |

The current config default is:

$$
\alpha_0 = 4,
\qquad
\beta_0 = 2.
$$

For an inverse-gamma random variable under this parameterization:

$$
\mathbb E[\sigma^2] = \frac{\beta_0}{\alpha_0-1} \qquad \text{for } \alpha_0>1.
$$

With $\alpha_0=4$ and $\beta_0=2$:

$$
\mathbb E[\sigma^2] = \frac{2}{3} \approx 0.667.
$$

The prior standard deviation scale is therefore roughly:

$$
\sqrt{2/3} \approx 0.816.
$$

That is in the same rough range as the default environment, where $\sigma_k$ is sampled between `0.5` and `1.5`.

## 4. Posterior Update

Because the Normal-Inverse-Gamma prior is conjugate to the Gaussian likelihood with unknown mean and variance, the posterior is also Normal-Inverse-Gamma:

$$
p(\mu_k,\sigma_k^2 \mid D_k) = \mathrm{NIG}(\mu_{k,n},\kappa_{k,n},\alpha_{k,n},\beta_{k,n}).
$$

The posterior parameters are:

$$
\kappa_{k,n} = \kappa_0 + n_k,
$$

$$
\mu_{k,n} = \frac{\kappa_0\mu_0+s_k}{\kappa_0+n_k},
$$

$$
\alpha_{k,n} = \alpha_0 + \frac{n_k}{2},
$$

and

$$
\beta_{k,n} = \beta_0 + \frac{1}{2}\left(q_k + \kappa_0\mu_0^2 - \kappa_{k,n}\mu_{k,n}^2\right).
$$

Equivalently, using the sample mean and centered sum of squares:

$$
\beta_{k,n} = \beta_0 + \frac{1}{2}S_k + \frac{1}{2}\frac{\kappa_0 n_k}{\kappa_0+n_k}(\bar r_k-\mu_0)^2.
$$

This decomposition is useful:

- $S_k$ measures within-arm reward noise;
- $(\bar r_k-\mu_0)^2$ measures how far the observed arm mean is from the prior mean;
- $\kappa_0$ controls how strongly the prior mean resists movement.

## 5. Posterior Over the Latent Mean

The posterior over the pair $(\mu_k,\sigma_k^2)$ is Normal-Inverse-Gamma. If we marginalize out $\sigma_k^2$, the posterior over the latent mean is Student-t:

$$
\mu_k \mid D_k \sim t_{\nu_{k,n}}\left(\mu_{k,n}, \tau_{k,n}\right),
$$

with degrees of freedom:

$$
\nu_{k,n} = 2\alpha_{k,n},
$$

location:

$$
\mu_{k,n},
$$

and scale:

$$
\tau_{k,n} = \sqrt{\frac{\beta_{k,n}}{\alpha_{k,n}\kappa_{k,n}}}.
$$

This is the distribution used to reason about which arm has the largest true mean.

The posterior variance of the latent mean is:

$$
\mathrm{Var}(\mu_k \mid D_k) = \frac{\beta_{k,n}}{(\alpha_{k,n}-1)\kappa_{k,n}} \qquad \text{for } \alpha_{k,n}>1.
$$

As $n_k$ grows, $\kappa_{k,n}$ grows, so uncertainty about $\mu_k$ shrinks.

## 6. Posterior Predictive Distribution

The posterior predictive distribution for the next reward from arm $k$ is:

$$
p(r_{k,n+1}\mid D_k) = \int p(r_{k,n+1}\mid \mu_k,\sigma_k^2)\, p(\mu_k,\sigma_k^2\mid D_k)\, d\mu_k\,d\sigma_k^2.
$$

This is also Student-t:

$$
r_{k,n+1}\mid D_k \sim t_{\nu_{k,n}}\left(\mu_{k,n}, \lambda_{k,n}\right),
$$

with:

$$
\lambda_{k,n} = \sqrt{\frac{\beta_{k,n}(\kappa_{k,n}+1)}{\alpha_{k,n}\kappa_{k,n}}}.
$$

The predictive scale is larger than the mean-posterior scale because it includes two sources of uncertainty:

1. uncertainty about the true mean;
2. irreducible reward noise around that mean.

## 7. Uncertainty Decomposition

The predictive variance decomposes as:

$$
\mathrm{Var}(R_{k,n+1}\mid D_k) = \mathbb E[\sigma_k^2\mid D_k] + \mathrm{Var}(\mu_k\mid D_k).
$$

For the Normal-Inverse-Gamma posterior:

$$
\mathbb E[\sigma_k^2\mid D_k] = \frac{\beta_{k,n}}{\alpha_{k,n}-1},
$$

and

$$
\mathrm{Var}(\mu_k\mid D_k) = \frac{1}{\kappa_{k,n}} \frac{\beta_{k,n}}{\alpha_{k,n}-1}.
$$

Therefore:

$$
\mathrm{Var}(R_{k,n+1}\mid D_k) = \frac{\beta_{k,n}}{\alpha_{k,n}-1} + \frac{\beta_{k,n}}{(\alpha_{k,n}-1)\kappa_{k,n}}.
$$

The first term is aleatoric uncertainty: reward noise that remains even if we know the arm perfectly.

The second term is epistemic uncertainty: uncertainty about the latent mean that can be reduced by sampling the arm.

This distinction is central to the Bayesian method. Exploration should be driven by epistemic uncertainty, not merely by arms that have noisy rewards.

## 8. Probability of Being the Best

The true best arm is:

$$
k^\star = \arg\max_{j\in\{1,\dots,K\}}\mu_j.
$$

The Bayesian quantity of interest is:

$$
\mathbb P(k^\star=k\mid D) = \mathbb P(\mu_k \ge \mu_j \;\forall j\ne k \mid D).
$$

Assuming the posterior factorizes across arms:

$$
p(\mu_1,\dots,\mu_K\mid D) = \prod_{j=1}^K p(\mu_j\mid D_j),
$$

we can write:

$$
\mathbb P(k^\star=k\mid D) = \int \mathbf 1\{x_k = \max_j x_j\} \prod_{j=1}^K p(\mu_j=x_j\mid D_j)\, dx_1\cdots dx_K.
$$

This is the most direct mathematical expression of "arm $k$ is the best."

## 9. Monte Carlo Estimator

One way to estimate $\mathbb P(k^\star=k\mid D)$ is posterior Monte Carlo:

1. Draw a latent mean for every arm:

$$
\mu_j^{(s)} \sim p(\mu_j\mid D_j).
$$

2. Identify the winner:

$$
k^{\star(s)} = \arg\max_j \mu_j^{(s)}.
$$

3. Estimate:

$$
\widehat{\mathbb P}(k^\star=k\mid D) = \frac{1}{S}\sum_{s=1}^{S}\mathbf 1\{k^{\star(s)}=k\}.
$$

This is simple and general. The downside is sampling noise.

## 10. One-Dimensional Quadrature Formula

The probability of being best can also be reduced to a one-dimensional integral for each arm.

Let:

$$
f_k(x) = p(\mu_k=x\mid D_k)
$$

be the posterior density of arm $k$'s latent mean, and let:

$$
F_j(x) = \mathbb P(\mu_j\le x\mid D_j)
$$

be the posterior CDF for arm $j$'s latent mean.

Then:

$$
\mathbb P(k^\star=k\mid D) = \int_{-\infty}^{\infty} f_k(x)\prod_{j\ne k}F_j(x)\,dx.
$$

Interpretation:

- pick a possible latent mean value $x$ for arm $k$;
- weight it by how plausible that value is under $k$'s posterior density;
- multiply by the probability that every other arm has latent mean below $x$;
- integrate across all possible $x$.

This is the formula used by the demo mode when computing final best-arm probabilities.

## 11. Policy Used in the Study

The study loop uses a fast sampling approximation. At each step:

1. compute the posterior Student-t distribution over every latent mean;
2. sample:

$$
\tilde \mu_k \sim p(\mu_k\mid D_k);
$$

3. select according to a softmax over sampled means:

$$
\pi(k\mid D) = \frac{\exp(\tilde\mu_k/T)}{\sum_j \exp(\tilde\mu_j/T)}.
$$

Here $T$ is the configured `temperature`.

As $T\to 0$, the softmax becomes nearly greedy:

$$
\pi(k\mid D) \approx \mathbf 1\{k=\arg\max_j \tilde\mu_j\}.
$$

This is essentially Thompson sampling over latent means. Since:

$$
\mathbb P\left(k=\arg\max_j \tilde\mu_j\right) = \mathbb P(k^\star=k\mid D),
$$

sampling the posterior winner is equivalent, in expectation, to sampling from the probability-of-being-best distribution.

## 12. Role of the Hyperparameters

### Kappa

$\kappa_0$ controls the strength of the prior mean:

$$
\mu_k\mid\sigma_k^2
\sim
\mathcal N
\left(
\mu_0,\frac{\sigma_k^2}{\kappa_0}
\right).
$$

Larger $\kappa_0$ means the prior mean is trusted more strongly. Smaller $\kappa_0$ means observations move the posterior mean faster.

### Alpha

$\alpha_0$ is the inverse-gamma shape parameter for the variance prior.

It affects:

- the prior expected variance;
- the Student-t degrees of freedom;
- how heavy-tailed the early posterior is.

The prior expected variance is:

$$
\mathbb E[\sigma^2] = \frac{\beta_0}{\alpha_0-1}.
$$

Holding $\beta_0$ fixed, larger $\alpha_0$ lowers the prior expected variance and increases the Student-t degrees of freedom:

$$
\nu_0 = 2\alpha_0.
$$

So larger $\alpha_0$ generally makes early posterior mean samples less heavy-tailed.

### Beta

$\beta_0$ is the inverse-gamma scale parameter.

Holding $\alpha_0$ fixed, larger $\beta_0$ increases the prior expected variance:

$$
\mathbb E[\sigma^2] = \frac{\beta_0}{\alpha_0-1}.
$$

In the beta sweep:

```text
alpha = 4
beta  = 2
```

so:

$$
\mathbb E[\sigma^2] = \frac{2}{3}.
$$

The best beta value being $2$ means that, under the current environment and horizon, this prior variance scale gave the best reward tradeoff among the powers-of-two candidates.

### Temperature

The temperature $T$ controls how deterministic the action selection is after drawing latent means.

Small $T$:

- close to Thompson sampling;
- often strong exploitation after each posterior draw;
- lower policy randomness.

Large $T$:

- more diffuse action probabilities;
- more random exploration;
- potentially worse cumulative reward if exploration is excessive.

## 13. Relationship to UCB

UCB chooses:

$$
A_t = \arg\max_k \left[\hat \mu_k + c\sqrt{\frac{\log t}{N_k(t)}}\right],
$$

where:

- $\hat \mu_k$ is the empirical mean reward for arm $k$;
- $N_k(t)$ is the number of times arm $k$ has been sampled;
- $c$ controls optimism.

This has a Bayesian flavor: it acts as if each uncertain arm receives an optimistic bonus.

But UCB and Bayesian P(best) optimize different practical quantities:

- UCB is an index policy designed for cumulative reward and regret control.
- Bayesian P(best) is a posterior probability over the identity of the best latent mean.

That difference explains the beta-sweep result:

```text
UCB average reward:              1.5490
Bayesian P(best) average reward: 1.5372
```

The Bayesian method is more directly aligned with best-arm identification, while UCB is extremely efficient at maximizing reward during the experiment.

## 14. Cumulative Reward vs. Best-Arm Identification

The current study reports average reward:

$$
\frac{1}{T}\sum_{t=1}^T R_t.
$$

This measures how much reward the agent earned while learning.

A different metric would be final recommendation accuracy:

$$
\mathbf 1\{\hat k_T = k^\star \},
$$

where:

$$
\hat k_T = \arg\max_k \mathbb P(k^\star=k\mid D_T).
$$

Another useful metric is simple regret:

$$
\mu_{k^\star} - \mu_{\hat k_T}.
$$

These metrics ask a different question:

- average reward: "Did we earn high rewards while learning?"
- recommendation accuracy: "Did we identify the true best arm?"
- simple regret: "How much expected reward did we lose in the final recommendation?"

The Bayesian probability-of-best method is theoretically most natural for the latter two.

## 15. Summary

The Bayesian method is rigorous because it maintains a posterior over the unknown Gaussian reward model and uses that posterior to reason about the identity of the best arm.

The key object is:

$$
\mathbb P(k^\star=k\mid D).
$$

This can be estimated by Monte Carlo, computed by one-dimensional quadrature, or approximated online through posterior sampling.

UCB can still win on cumulative reward because it is an efficient optimism-based index policy. That does not make the Bayesian objective wrong; it means the experiment metric matters. The current beta sweep shows that Bayesian P(best) is very competitive on cumulative reward, while remaining more directly tied to the posterior question of which arm is truly best.
