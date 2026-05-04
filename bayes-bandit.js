/* The Bayes Bandit — interactive figures
 * ---------------------------------------
 * Three D3-based widgets:
 *   #arms-svg : violin plot of true reward distributions (Figure 1)
 *   #post-svg : Student-t posteriors over each arm's latent mean (Figure 2)
 *   #mc-svg   : Monte Carlo estimator of P(k* = k | D)              (Figure 3)
 * Figures 2 & 3 share state — pulling arms in Fig 2 updates Fig 3.
 */

(function () {
  "use strict";

  /* ---------------- shared math helpers ---------------- */

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Gamma(shape) sampler (Marsaglia-Tsang) for shape >= 1; recurse otherwise.
  function gammaSample(shape) {
    if (shape < 1) {
      const u = Math.random();
      return gammaSample(shape + 1) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        x = randn();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  // Student-t sample with nu d.o.f., location loc, scale tau.
  function studentTSample(nu, loc, tau) {
    // t = z / sqrt(g/nu) where z ~ N(0,1), g ~ chi^2_nu = 2 * Gamma(nu/2)
    const z = randn();
    const g = 2 * gammaSample(nu / 2);
    return loc + tau * z / Math.sqrt(g / nu);
  }

  // Student-t pdf, density of t_nu(loc, tau)
  function studentTPdf(x, nu, loc, tau) {
    const z = (x - loc) / tau;
    // log gamma via Lanczos
    const num = Math.exp(lgamma((nu + 1) / 2));
    const den = Math.sqrt(nu * Math.PI) * Math.exp(lgamma(nu / 2)) * tau;
    return num / den * Math.pow(1 + z * z / nu, -(nu + 1) / 2);
  }

  function lgamma(z) {
    const g = 7;
    const c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    }
    z -= 1;
    let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  // Normal pdf
  function normPdf(x, mu, sigma) {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }

  /* ---------------- arm palette ---------------- */
  const ARM_COLORS = [
    "#7aa6ff", // light blue
    "#7e6ad7", // purple
    "#d63b76", // pink
    "#f07a26", // orange
    "#f1b840"  // yellow
  ];
  const K = 5;

  /* ---------------- world generation ---------------- */
  function makeWorld(opts) {
    const muScale = opts.muScale ?? 1.0;
    const sigMin = opts.sigMin ?? 0.5;
    const sigMax = opts.sigMax ?? 1.5;
    const arms = [];
    for (let k = 0; k < K; k++) {
      const mu = muScale * randn();
      const sigma = sigMin + (sigMax - sigMin) * Math.random();
      arms.push({ mu, sigma });
    }
    return arms;
  }

  /* ============================================================
     Figure 1 — sampled arm violin plot
     ============================================================ */
  (function armsFigure() {
    const svg = d3.select("#arms-svg");
    if (svg.empty()) return;
    const W = 1100, H = 480;
    const margin = { top: 28, right: 24, bottom: 48, left: 60 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleBand().domain(d3.range(K)).range([0, innerW]).padding(0.35);
    const yScale = d3.scaleLinear().domain([-4.5, 4.5]).range([innerH, 0]);

    // Axes
    root.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `arm ${d + 1}`))
      .call(g => g.selectAll("text").attr("font-size", 13));
    root.append("g")
      .call(d3.axisLeft(yScale).ticks(7))
      .call(g => g.selectAll("text").attr("font-size", 12));

    // Axis titles
    root.append("text")
      .attr("transform", `translate(${innerW/2},${innerH + 36})`)
      .attr("text-anchor", "middle")
      .attr("font-size", 13)
      .text("Arm");
    root.append("text")
      .attr("transform", `translate(-44,${innerH/2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .attr("font-size", 13)
      .text("Reward distribution");

    // Zero line
    root.append("line")
      .attr("x1", 0).attr("x2", innerW)
      .attr("y1", yScale(0)).attr("y2", yScale(0))
      .attr("stroke", "#999")
      .attr("stroke-dasharray", "4 4");

    const violinLayer = root.append("g");

    function drawViolins(arms) {
      const groups = violinLayer.selectAll("g.violin").data(arms);
      groups.exit().remove();
      const enter = groups.enter().append("g").attr("class", "violin");
      const merged = enter.merge(groups)
        .attr("transform", (_, i) => `translate(${xScale(i)},0)`);

      merged.each(function (arm, i) {
        const g = d3.select(this);
        g.selectAll("*").remove();
        const bw = xScale.bandwidth();
        const bx = bw / 2;

        // Sample density curve
        const ys = d3.range(80).map(j => -4.5 + (9 * j) / 79);
        const dens = ys.map(y => normPdf(y, arm.mu, arm.sigma));
        const maxDens = d3.max(dens);
        const halfWidth = bw / 2;

        const violinPath = d3.area()
          .x0(d => bx - (d.dens / maxDens) * halfWidth)
          .x1(d => bx + (d.dens / maxDens) * halfWidth)
          .y(d => yScale(d.y))
          .curve(d3.curveBasis);

        g.append("path")
          .datum(ys.map((y, j) => ({ y, dens: dens[j] })))
          .attr("d", violinPath)
          .attr("fill", ARM_COLORS[i])
          .attr("opacity", 0.85)
          .attr("stroke", ARM_COLORS[i])
          .attr("stroke-opacity", 0.7);

        // True mean bar
        g.append("line")
          .attr("x1", bx - halfWidth * 0.55)
          .attr("x2", bx + halfWidth * 0.55)
          .attr("y1", yScale(arm.mu))
          .attr("y2", yScale(arm.mu))
          .attr("stroke", "#222")
          .attr("stroke-width", 2);

        g.append("text")
          .attr("x", bx + halfWidth * 0.6 + 4)
          .attr("y", yScale(arm.mu) + 4)
          .attr("font-size", 12)
          .attr("font-style", "italic")
          .text(`μ${i + 1} = ${arm.mu.toFixed(2)}`);
      });
    }

    // Controls
    const muScaleEl = document.getElementById("arms-muscale");
    const sigMinEl = document.getElementById("arms-sigmin");
    const sigMaxEl = document.getElementById("arms-sigmax");
    const muScaleVal = document.getElementById("arms-muscale-val");
    const sigMinVal = document.getElementById("arms-sigmin-val");
    const sigMaxVal = document.getElementById("arms-sigmax-val");
    const statsEl = document.getElementById("arms-stats");

    function getOpts() {
      let sigMin = parseFloat(sigMinEl.value);
      let sigMax = parseFloat(sigMaxEl.value);
      if (sigMin > sigMax) { const t = sigMin; sigMin = sigMax; sigMax = t; }
      return {
        muScale: parseFloat(muScaleEl.value),
        sigMin, sigMax
      };
    }

    function syncLabels(opts) {
      muScaleVal.textContent = opts.muScale.toFixed(2);
      sigMinVal.textContent = opts.sigMin.toFixed(2);
      sigMaxVal.textContent = opts.sigMax.toFixed(2);
    }

    let arms = makeWorld(getOpts());
    function refresh(regen) {
      const opts = getOpts();
      syncLabels(opts);
      if (regen) arms = makeWorld(opts);
      drawViolins(arms);
      const best = arms.reduce((b, a, i) => (a.mu > arms[b].mu ? i : b), 0);
      statsEl.textContent = `Best arm in this world: arm ${best + 1} (μ = ${arms[best].mu.toFixed(2)})`;
    }

    document.getElementById("arms-resample").addEventListener("click", () => refresh(true));
    [muScaleEl, sigMinEl, sigMaxEl].forEach(el =>
      el.addEventListener("input", () => refresh(false))
    );
    refresh(false);
  })();

  /* ============================================================
     Figures 2 + 3 share posterior state
     ============================================================ */
  const sharedState = {
    arms: makeWorld({ muScale: 1.0, sigMin: 0.5, sigMax: 1.5 }),
    pulls: Array(K).fill(null).map(() => ({ n: 0, s: 0, q: 0 })),
    listeners: []
  };

  function priorParams() {
    return {
      mu0: parseFloat(document.getElementById("prior-mu").value),
      kappa0: parseFloat(document.getElementById("prior-kappa").value),
      alpha0: parseFloat(document.getElementById("prior-alpha").value),
      beta0: parseFloat(document.getElementById("prior-beta").value)
    };
  }

  function posteriorParams(k) {
    const { mu0, kappa0, alpha0, beta0 } = priorParams();
    const { n, s, q } = sharedState.pulls[k];
    const kappaN = kappa0 + n;
    const muN = (kappa0 * mu0 + s) / kappaN;
    const alphaN = alpha0 + n / 2;
    const betaN = beta0 + 0.5 * (q + kappa0 * mu0 * mu0 - kappaN * muN * muN);
    const nu = 2 * alphaN;
    const tau = Math.sqrt(Math.max(betaN, 1e-9) / (alphaN * kappaN));
    return { kappaN, muN, alphaN, betaN, nu, tau };
  }

  function pullArm(k) {
    const { mu, sigma } = sharedState.arms[k];
    const r = mu + sigma * randn();
    const p = sharedState.pulls[k];
    p.n += 1;
    p.s += r;
    p.q += r * r;
    notify();
  }

  function resetPulls() {
    sharedState.pulls = Array(K).fill(null).map(() => ({ n: 0, s: 0, q: 0 }));
    notify();
  }

  function newWorld() {
    sharedState.arms = makeWorld({ muScale: 1.0, sigMin: 0.5, sigMax: 1.5 });
    resetPulls();
  }

  function notify() {
    sharedState.listeners.forEach(fn => fn());
  }

  /* ============================================================
     Figure 2 — posteriors over latent means
     ============================================================ */
  (function posteriorFigure() {
    const svg = d3.select("#post-svg");
    if (svg.empty()) return;
    const W = 1100, H = 460;
    const margin = { top: 28, right: 24, bottom: 50, left: 60 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;
    const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([-4, 4]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1.6]).range([innerH, 0]);

    root.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(9))
      .call(g => g.selectAll("text").attr("font-size", 12));
    root.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .call(g => g.selectAll("text").attr("font-size", 12));

    root.append("text")
      .attr("transform", `translate(${innerW/2},${innerH + 38})`)
      .attr("text-anchor", "middle")
      .attr("font-size", 13)
      .text("Latent mean μₖ");
    root.append("text")
      .attr("transform", `translate(-44,${innerH/2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .attr("font-size", 13)
      .text("Posterior density");

    const trueLayer = root.append("g");
    const postLayer = root.append("g");

    const pullsBox = document.getElementById("arm-pulls");
    const statsEl = document.getElementById("post-stats");

    // Build the 5 arm-pull buttons
    for (let k = 0; k < K; k++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.arm = k;
      btn.innerHTML = `
        <span class="arm-dot" style="background:${ARM_COLORS[k]}"></span>
        <span>Pull arm ${k + 1}</span>
        <span class="arm-n" data-armn="${k}">n = 0</span>`;
      btn.addEventListener("click", () => pullArm(k));
      pullsBox.appendChild(btn);
    }

    function draw() {
      // True mean rules
      const trueSel = trueLayer.selectAll("line.truebar").data(sharedState.arms);
      trueSel.enter().append("line").attr("class", "truebar")
        .merge(trueSel)
        .attr("x1", d => xScale(d.mu))
        .attr("x2", d => xScale(d.mu))
        .attr("y1", 0)
        .attr("y2", innerH)
        .attr("stroke", (_, i) => ARM_COLORS[i])
        .attr("stroke-width", 2)
        .attr("opacity", 0.55);
      trueSel.exit().remove();

      const trueLab = trueLayer.selectAll("text.truelab").data(sharedState.arms);
      trueLab.enter().append("text").attr("class", "truelab")
        .merge(trueLab)
        .attr("x", d => xScale(d.mu) + 4)
        .attr("y", (_, i) => 14 + i * 14)
        .attr("font-size", 11)
        .attr("font-style", "italic")
        .attr("fill", (_, i) => ARM_COLORS[i])
        .text((d, i) => `μ${i + 1}*`);
      trueLab.exit().remove();

      // Posterior densities
      const xs = d3.range(220).map(i => -4 + (8 * i) / 219);
      const allPosts = sharedState.arms.map((_, k) => {
        const pp = posteriorParams(k);
        return xs.map(x => ({ x, y: studentTPdf(x, pp.nu, pp.muN, pp.tau) }));
      });
      // dynamic y-domain: cap at 2 so weak posteriors still show
      const maxY = Math.min(2.5, d3.max(allPosts.flat(), d => d.y) || 1.0);
      yScale.domain([0, Math.max(0.6, maxY * 1.05)]);
      root.select("g:nth-of-type(2)").call(d3.axisLeft(yScale).ticks(5))
        .selectAll("text").attr("font-size", 12);

      const line = d3.line().x(d => xScale(d.x)).y(d => yScale(d.y)).curve(d3.curveBasis);
      const area = d3.area().x(d => xScale(d.x)).y0(yScale(0)).y1(d => yScale(d.y)).curve(d3.curveBasis);

      const fillSel = postLayer.selectAll("path.postfill").data(allPosts);
      fillSel.enter().append("path").attr("class", "postfill")
        .merge(fillSel)
        .attr("d", area)
        .attr("fill", (_, i) => ARM_COLORS[i])
        .attr("opacity", 0.18);
      fillSel.exit().remove();

      const lineSel = postLayer.selectAll("path.postline").data(allPosts);
      lineSel.enter().append("path").attr("class", "postline")
        .merge(lineSel)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", (_, i) => ARM_COLORS[i])
        .attr("stroke-width", 2);
      lineSel.exit().remove();

      // Update n counts on buttons
      for (let k = 0; k < K; k++) {
        const el = document.querySelector(`[data-armn="${k}"]`);
        if (el) el.textContent = `n = ${sharedState.pulls[k].n}`;
      }

      const totalN = sharedState.pulls.reduce((acc, p) => acc + p.n, 0);
      const best = sharedState.arms.reduce((b, a, i) => (a.mu > sharedState.arms[b].mu ? i : b), 0);
      statsEl.textContent = `Total pulls: ${totalN} · True best: arm ${best + 1} (μ = ${sharedState.arms[best].mu.toFixed(2)})`;
    }

    // Listeners
    sharedState.listeners.push(draw);
    document.getElementById("post-pull10").addEventListener("click", () => {
      for (let i = 0; i < 10; i++) pullArm(Math.floor(Math.random() * K));
    });
    document.getElementById("post-reset").addEventListener("click", () => resetPulls());
    document.getElementById("post-newworld").addEventListener("click", () => newWorld());

    ["prior-mu", "prior-kappa", "prior-alpha", "prior-beta"].forEach(id => {
      const el = document.getElementById(id);
      const valEl = document.getElementById(`${id}-val`);
      el.addEventListener("input", () => {
        const v = parseFloat(el.value);
        valEl.textContent = (id === "prior-alpha") ? v.toFixed(1) : v.toFixed(2);
        draw();
        // also tickle MC fig
        notify();
      });
    });

    draw();
  })();

  /* ============================================================
     Figure 3 — Monte Carlo P(best)
     ============================================================ */
  (function mcFigure() {
    const svg = d3.select("#mc-svg");
    if (svg.empty()) return;
    const W = 1100, H = 380;
    const margin = { top: 28, right: 24, bottom: 50, left: 60 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;
    const root = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Two panels: left = strip of recent draws, right = bar chart of P(best).
    const stripWidth = innerW * 0.55;
    const barWidth = innerW * 0.40;
    const stripGap = innerW - stripWidth - barWidth;

    const stripG = root.append("g");
    const barG = root.append("g").attr("transform", `translate(${stripWidth + stripGap},0)`);

    // Strip: x = draw index, y = sampled value, dot color = winner
    const stripX = d3.scaleLinear().domain([0, 200]).range([0, stripWidth]);
    const stripY = d3.scaleLinear().domain([-4, 4]).range([innerH, 0]);

    stripG.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(stripX).ticks(5))
      .call(g => g.selectAll("text").attr("font-size", 11));
    stripG.append("g")
      .call(d3.axisLeft(stripY).ticks(5))
      .call(g => g.selectAll("text").attr("font-size", 11));
    stripG.append("text")
      .attr("transform", `translate(${stripWidth/2},${innerH + 36})`)
      .attr("text-anchor", "middle").attr("font-size", 12)
      .text("Most recent 200 draws (winning arm shown)");
    stripG.append("text")
      .attr("transform", `translate(-44,${innerH/2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("font-size", 12)
      .text("Winning μ̃");

    const dotsLayer = stripG.append("g");

    // Bars: P(best) per arm
    const barX = d3.scaleBand().domain(d3.range(K)).range([0, barWidth]).padding(0.25);
    const barY = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
    barG.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(barX).tickFormat(d => `arm ${d + 1}`))
      .call(g => g.selectAll("text").attr("font-size", 11));
    barG.append("g")
      .call(d3.axisLeft(barY).ticks(5).tickFormat(d3.format(".0%")))
      .call(g => g.selectAll("text").attr("font-size", 11));
    barG.append("text")
      .attr("transform", `translate(${barWidth/2},${innerH + 36})`)
      .attr("text-anchor", "middle").attr("font-size", 12)
      .text("P(k* = k | D)");

    const barsLayer = barG.append("g");
    const barLabels = barG.append("g");

    // State
    let draws = []; // recent draws (uncapped except for the strip render)
    let wins = Array(K).fill(0);
    let totalDraws = 0;
    let timer = null;

    const statsEl = document.getElementById("mc-stats");
    const pullsEl = document.getElementById("mc-pulls");

    function doDraw() {
      const samples = sharedState.arms.map((_, k) => {
        const pp = posteriorParams(k);
        return studentTSample(pp.nu, pp.muN, pp.tau);
      });
      let best = 0;
      for (let i = 1; i < K; i++) if (samples[i] > samples[best]) best = i;
      wins[best] += 1;
      totalDraws += 1;
      draws.push({ idx: totalDraws, winner: best, val: samples[best] });
      if (draws.length > 200) draws.shift();
    }

    function draw() {
      // strip
      const minIdx = Math.max(1, totalDraws - 199);
      stripX.domain([minIdx, Math.max(minIdx + 199, totalDraws)]);
      stripG.select("g").call(d3.axisBottom(stripX).ticks(5))
        .selectAll("text").attr("font-size", 11);

      const dots = dotsLayer.selectAll("circle").data(draws, d => d.idx);
      dots.exit().remove();
      dots.enter().append("circle")
        .attr("r", 3)
        .merge(dots)
        .attr("cx", d => stripX(d.idx))
        .attr("cy", d => stripY(Math.max(-4, Math.min(4, d.val))))
        .attr("fill", d => ARM_COLORS[d.winner])
        .attr("opacity", 0.75);

      // bars
      const probs = wins.map(w => totalDraws > 0 ? w / totalDraws : 0);
      const bars = barsLayer.selectAll("rect").data(probs);
      bars.enter().append("rect")
        .merge(bars)
        .attr("x", (_, i) => barX(i))
        .attr("y", d => barY(d))
        .attr("width", barX.bandwidth())
        .attr("height", d => innerH - barY(d))
        .attr("fill", (_, i) => ARM_COLORS[i])
        .attr("opacity", 0.85);
      bars.exit().remove();

      const labels = barLabels.selectAll("text").data(probs);
      labels.enter().append("text")
        .merge(labels)
        .attr("x", (_, i) => barX(i) + barX.bandwidth() / 2)
        .attr("y", d => barY(d) - 6)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("font-variant-numeric", "tabular-nums")
        .attr("fill", "#222")
        .text(d => totalDraws > 0 ? (d * 100).toFixed(0) + "%" : "—");
      labels.exit().remove();

      // text
      statsEl.textContent = totalDraws > 0
        ? `Total draws: ${totalDraws}`
        : "No draws yet — click below to start sampling.";

      const totalN = sharedState.pulls.reduce((acc, p) => acc + p.n, 0);
      const perArm = sharedState.pulls.map((p, i) => `arm${i+1}=${p.n}`).join(" · ");
      pullsEl.textContent = `Total: ${totalN} · ${perArm}`;
    }

    function resetDraws() {
      draws = [];
      wins = Array(K).fill(0);
      totalDraws = 0;
      draw();
    }

    function stopStream() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
        document.getElementById("mc-toggle").textContent = "Stream draws";
      }
    }

    document.getElementById("mc-toggle").addEventListener("click", function () {
      if (timer === null) {
        timer = setInterval(() => {
          for (let i = 0; i < 4; i++) doDraw();
          draw();
        }, 60);
        this.textContent = "Pause stream";
      } else {
        stopStream();
      }
    });
    document.getElementById("mc-step").addEventListener("click", () => {
      for (let i = 0; i < 50; i++) doDraw();
      draw();
    });
    document.getElementById("mc-reset").addEventListener("click", () => {
      stopStream();
      resetDraws();
    });

    // When pulls or priors change, the posterior changed — reset MC tally.
    sharedState.listeners.push(() => {
      resetDraws();
    });

    draw();
  })();
})();
