// Copy BibTeX to clipboard
function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button.querySelector('.copy-text');

  function showCopied() {
    button.classList.add('copied');
    copyText.textContent = 'Cop';
    setTimeout(function () {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  }

  navigator.clipboard.writeText(bibtexElement.textContent).then(showCopied).catch(function () {
    // Fallback for browsers without the clipboard API
    const textArea = document.createElement('textarea');
    textArea.value = bibtexElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    showCopied();
  });
}

// Scroll to top
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  const scrollButton = document.querySelector('.scroll-to-top');
  scrollButton.classList.toggle('visible', window.pageYOffset > 300);
});

// Result tabs (one figure per problem)
function setupResultTabs() {
  document.querySelectorAll('.result-tab').forEach(function (tab) {
    tab.addEventListener('click', function () {
      const target = tab.dataset.target;
      document.querySelectorAll('.result-tab').forEach(function (t) {
        const active = t === tab;
        t.classList.toggle('is-active', active);
        t.setAttribute('aria-selected', active);
      });
      document.querySelectorAll('.result-panel').forEach(function (panel) {
        panel.classList.toggle('is-active', panel.id === target);
      });
    });
  });
}

// Mixture explorer: p(xi_i) = exp(lambda * xi_i) / sum_j exp(lambda * xi_j), and the expected
// number of samples E[M] = B / E_p[c] that a budget B buys under that distribution.
// Fidelity levels and per-sample costs (solver iterations) of the linear elasticity solver error
// dataset; the budget is B = B~ * c_HF.
const MIX_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const MIX_COSTS = [25000, 26505, 28188, 30095, 32297, 34902, 38089, 42199, 47991, 57893, 262144];
const MIX_BUDGETS = [100, 200, 400, 800, 1600, 3200, 6400];
const LAMBDA_MIN = -20, LAMBDA_MAX = 20;
const COLOR_OURS = '#2d6a4f', COLOR_BASELINE = '#cf4924';
const INK_MUTED = '#64748b', INK = '#1e293b', GRID = '#e2e8f0';

function mixturePmf(lambda) {
  const w = MIX_LEVELS.map(function (xi) { return Math.exp(lambda * xi); });
  const z = w.reduce(function (a, b) { return a + b; }, 0);
  return w.map(function (v) { return v / z; });
}

function expectedSamples(lambda, btilde) {
  const p = mixturePmf(lambda);
  const meanCost = p.reduce(function (acc, pi, i) { return acc + pi * MIX_COSTS[i]; }, 0);
  return btilde * MIX_COSTS[MIX_COSTS.length - 1] / meanCost;
}

function formatCount(n) {
  return Math.round(n).toLocaleString('en-US');
}

function svgPlot(svg, W, H) {
  const NS = 'http://www.w3.org/2000/svg';
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  return function el(name, attrs, text, parent) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (text !== undefined && text !== null) node.textContent = text;
    (parent || svg).appendChild(node);
    return node;
  };
}

function setupMixtureExplorer() {
  const pmfSvg = document.getElementById('pmf-plot');
  const countSvg = document.getElementById('count-plot');
  const lambdaSlider = document.getElementById('lambda-slider');
  const budgetSlider = document.getElementById('budget-slider');
  if (!pmfSvg || !countSvg || !lambdaSlider || !budgetSlider) return;

  const W = 420, H = 280, left = 62, right = 14, top = 14, bottom = 48;
  const plotW = W - left - right, plotH = H - top - bottom;

  // Left: probability of each fidelity level
  const el1 = svgPlot(pmfSvg, W, H);
  const slot = plotW / MIX_LEVELS.length, barW = slot * 0.62;
  [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
    const y = top + plotH * (1 - v);
    el1('line', { x1: left, x2: W - right, y1: y, y2: y, stroke: GRID, 'stroke-width': 1 });
    el1('text', { x: left - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 12, fill: INK_MUTED }, v.toFixed(2));
  });
  el1('text', { x: 14, y: top + plotH / 2, 'text-anchor': 'middle', 'font-size': 13, fill: INK,
                transform: 'rotate(-90 14 ' + (top + plotH / 2) + ')' }, 'probability');
  el1('text', { x: left + plotW / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 13, fill: INK },
      'normalized fidelity');
  const bars = MIX_LEVELS.map(function (xi, i) {
    const x = left + i * slot + (slot - barW) / 2;
    if (i % 2 === 0) {
      el1('text', { x: x + barW / 2, y: top + plotH + 18, 'text-anchor': 'middle', 'font-size': 12, fill: INK_MUTED },
          xi.toFixed(1));
    }
    const bar = el1('rect', { x: x, width: barW, rx: 3, fill: COLOR_OURS });
    el1('title', {}, '', bar);
    return bar;
  });

  // Right: expected number of samples over lambda, one curve per budget (log y axis)
  const el2 = svgPlot(countSvg, W, H);
  const yMin = 50, yMax = 1e5;
  const xOf = function (lam) { return left + plotW * (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN); };
  const yOf = function (n) {
    return top + plotH * (1 - (Math.log10(n) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)));
  };
  [[100, '100'], [1000, '1k'], [10000, '10k'], [100000, '100k']].forEach(function (t) {
    const y = yOf(t[0]);
    el2('line', { x1: left, x2: W - right, y1: y, y2: y, stroke: GRID, 'stroke-width': 1 });
    el2('text', { x: left - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 12, fill: INK_MUTED }, t[1]);
  });
  [-20, -10, 0, 10, 20].forEach(function (lam) {
    el2('text', { x: xOf(lam), y: top + plotH + 18, 'text-anchor': 'middle', 'font-size': 12, fill: INK_MUTED },
        String(lam));
  });
  el2('text', { x: 14, y: top + plotH / 2, 'text-anchor': 'middle', 'font-size': 13, fill: INK,
                transform: 'rotate(-90 14 ' + (top + plotH / 2) + ')' }, 'number of samples');
  el2('text', { x: left + plotW / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 13, fill: INK }, 'λ');

  const lambdas = [];
  for (let lam = LAMBDA_MIN; lam <= LAMBDA_MAX + 1e-9; lam += 0.25) lambdas.push(lam);
  const curvePath = function (btilde) {
    return lambdas.map(function (lam, i) {
      return (i ? 'L' : 'M') + xOf(lam).toFixed(1) + ' ' + yOf(expectedSamples(lam, btilde)).toFixed(1);
    }).join(' ');
  };
  const curves = MIX_BUDGETS.map(function (b) {
    return el2('path', { d: curvePath(b), fill: 'none', stroke: '#cbd5e1', 'stroke-width': 1.5 });
  });
  const hfLine = el2('line', { x1: left, x2: W - right, stroke: COLOR_BASELINE, 'stroke-width': 2,
                               'stroke-dasharray': '6 4' });
  const hfLabel = el2('text', { x: left + 6, 'text-anchor': 'start', 'font-size': 12, fill: COLOR_BASELINE },
                      'high fidelity only');
  const marker = el2('circle', { r: 6, fill: COLOR_OURS, stroke: '#ffffff', 'stroke-width': 2 });

  const lambdaOut = document.getElementById('lambda-value');
  const meanOut = document.getElementById('mean-fidelity');
  const budgetOut = document.getElementById('budget-value');
  const countOut = document.getElementById('sample-count');
  const hfOut = document.getElementById('hf-count');

  function update() {
    const lambda = parseFloat(lambdaSlider.value);
    const bIndex = parseInt(budgetSlider.value, 10);
    const btilde = MIX_BUDGETS[bIndex];
    const p = mixturePmf(lambda);
    const n = expectedSamples(lambda, btilde);

    p.forEach(function (pi, i) {
      const h = Math.max(pi * plotH, 1);
      bars[i].setAttribute('y', top + plotH - h);
      bars[i].setAttribute('height', h);
      bars[i].firstChild.textContent = 'fidelity ' + MIX_LEVELS[i].toFixed(1) + ': p = ' + pi.toFixed(3) +
        ', ' + formatCount(pi * n) + ' samples';
    });

    curves.forEach(function (c, i) {
      const active = i === bIndex;
      c.setAttribute('stroke', active ? COLOR_OURS : '#cbd5e1');
      c.setAttribute('stroke-width', active ? 3 : 1.5);
    });
    countSvg.appendChild(curves[bIndex]);  // draw the selected curve on top
    hfLine.setAttribute('y1', yOf(btilde));
    hfLine.setAttribute('y2', yOf(btilde));
    hfLabel.setAttribute('y', yOf(btilde) - 7);
    countSvg.appendChild(hfLine);
    countSvg.appendChild(hfLabel);
    marker.setAttribute('cx', xOf(lambda));
    marker.setAttribute('cy', yOf(n));
    countSvg.appendChild(marker);

    const mean = p.reduce(function (acc, pi, i) { return acc + pi * MIX_LEVELS[i]; }, 0);
    lambdaOut.textContent = (lambda > 0 ? '+' : '') + lambda.toFixed(1);
    meanOut.textContent = mean.toFixed(2);
    budgetOut.textContent = formatCount(btilde);
    countOut.textContent = formatCount(n);
    hfOut.textContent = formatCount(btilde);
  }

  lambdaSlider.addEventListener('input', update);
  budgetSlider.addEventListener('input', update);
  update();
}

document.addEventListener('DOMContentLoaded', function () {
  setupResultTabs();
  setupMixtureExplorer();

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  }
});
