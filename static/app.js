/* ══════════════════════════════════════════════════════════════════════════
   SafeDPMSolver — Interactive Visualiser
   app.js  (vanilla JS + D3 v7)
   ══════════════════════════════════════════════════════════════════════════ */

const API = '';   // same origin

/* ── State ─────────────────────────────────────────────────────────────── */
let cachedData      = null;   // full /api/run response
let currentStep     = 0;
let currentSafeTraj = null;   // [[x,y],...] for current step (safe path)
let draggedIdx      = -1;     // index of waypoint being dragged

let obstaclePos  = { x: 0.0, y: 0.0 };
let params = { c: 1.0, k1: 1.0, k2: 1.0, r: 0.3, gamma_delta: 0.05 };

/* ── Canvas geometry ────────────────────────────────────────────────────── */
const WORLD_MIN  = -1.5;
const WORLD_MAX  =  1.5;
let   SZ         = 500;   // updated on layout

const xS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([0, SZ]);
const yS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([SZ, 0]);

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const svg        = d3.select('#canvas-svg');
const overlay    = document.getElementById('loading-overlay');
const runBtn     = document.getElementById('run-btn');
const scrubber   = document.getElementById('step-scrubber');
const stepLabel  = document.getElementById('step-label');
const hxtValue   = document.getElementById('hxt-value');
const tbody      = document.getElementById('inspector-tbody');

/* ════════════════════════════════════════════════════════════════════════
   INITIALISATION
   ════════════════════════════════════════════════════════════════════════ */

window.addEventListener('DOMContentLoaded', () => {
  fetchModels();
  buildCanvas();
  bindParamControls();
  bindRunButton();
  bindScrubber();
  bindKeyboardCursor();
  window.addEventListener('resize', onResize);
});

function onResize() {
  buildCanvas();
  if (currentSafeTraj) renderCurrentStep();
}

/* ── Fetch model list ───────────────────────────────────────────────────── */
async function fetchModels() {
  try {
    const res   = await fetch(`${API}/api/models`);
    const data  = await res.json();
    const sel   = document.getElementById('model-select');
    sel.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    });
    if (data.models && data.models.length === 0) {
      sel.innerHTML = '<option value="">No checkpoints found</option>';
    }
  } catch (e) {
    console.warn('Could not load models:', e);
  }
}

/* ════════════════════════════════════════════════════════════════════════
   CANVAS SETUP
   ════════════════════════════════════════════════════════════════════════ */

function buildCanvas() {
  const container = document.getElementById('canvas-container');
  SZ = Math.min(container.clientWidth, container.clientHeight) - 4;
  SZ = Math.max(SZ, 300);

  svg.attr('width', SZ).attr('height', SZ);
  xS.range([40, SZ - 20]);
  yS.range([SZ - 30, 20]);

  svg.selectAll('*').remove();

  // Arrow-head markers
  const defs = svg.append('defs');
  ['safe', 'danger'].forEach(cls => {
    const col = cls === 'safe' ? 'var(--safe-green, #3ddc84)' : 'var(--danger-red, #ff5370)';
    defs.append('marker')
      .attr('id', `arrow-${cls}`)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3)
      .attr('orient', 'auto')
      .append('path')
        .attr('d', 'M0,0 L0,6 L6,3 z')
        .attr('fill', col);
  });

  drawGrid();
  drawAxes();

  // Layer order (back → front)
  svg.append('g').attr('id', 'layer-prior');
  svg.append('g').attr('id', 'layer-plain');
  svg.append('g').attr('id', 'layer-safe-path');
  svg.append('g').attr('id', 'layer-obstacle');
  svg.append('g').attr('id', 'layer-grads');
  svg.append('g').attr('id', 'layer-waypoints');
  svg.append('g').attr('id', 'layer-markers');

  drawObstacle();
  drawMarkers();
}

function drawGrid() {
  const g = svg.append('g').attr('id', 'grid');
  const step = 0.5;
  for (let v = WORLD_MIN; v <= WORLD_MAX + 0.01; v += step) {
    const px = xS(v), py = yS(v);
    g.append('line').attr('class','grid-line')
      .attr('x1', px).attr('y1', yS(WORLD_MIN))
      .attr('x2', px).attr('y2', yS(WORLD_MAX));
    g.append('line').attr('class','grid-line')
      .attr('x1', xS(WORLD_MIN)).attr('y1', py)
      .attr('x2', xS(WORLD_MAX)).attr('y2', py);
  }
}

function drawAxes() {
  const g = svg.append('g').attr('id', 'axes');
  const ox = xS(0), oy = yS(0);
  g.append('line').attr('class','axis-line')
    .attr('x1', xS(WORLD_MIN)).attr('y1', oy)
    .attr('x2', xS(WORLD_MAX)).attr('y2', oy);
  g.append('line').attr('class','axis-line')
    .attr('x1', ox).attr('y1', yS(WORLD_MIN))
    .attr('x2', ox).attr('y2', yS(WORLD_MAX));
  [-1, -0.5, 0.5, 1].forEach(v => {
    g.append('text').attr('class','axis-label')
      .attr('x', xS(v)).attr('y', oy + 14)
      .attr('text-anchor','middle').text(v);
    g.append('text').attr('class','axis-label')
      .attr('x', ox - 6).attr('y', yS(v) + 4)
      .attr('text-anchor','end').text(v);
  });
}

function drawObstacle() {
  const g = d3.select('#layer-obstacle');
  g.selectAll('*').remove();

  const cx = xS(obstaclePos.x), cy = yS(obstaclePos.y);
  const rPx     = xS(params.r) - xS(0);
  const marginPx = xS(params.r + params.gamma_delta) - xS(0);

  // Filled area
  g.append('circle').attr('class','obstacle-fill obstacle-cursor')
    .attr('cx', cx).attr('cy', cy).attr('r', rPx);
  // Hard ring
  g.append('circle').attr('class','obstacle-ring obstacle-cursor')
    .attr('cx', cx).attr('cy', cy).attr('r', rPx);
  // Safety margin ring
  g.append('circle').attr('class','obstacle-margin obstacle-cursor')
    .attr('cx', cx).attr('cy', cy).attr('r', marginPx);

  // Drag to reposition obstacle
  const drag = d3.drag()
    .on('drag', function(event) {
      obstaclePos.x = xS.invert(event.x);
      obstaclePos.y = yS.invert(event.y);
      drawObstacle();
      if (currentSafeTraj) debouncedMathUpdate();
    });
  g.selectAll('circle').call(drag);
}

function drawMarkers() {
  const g = d3.select('#layer-markers');
  g.selectAll('*').remove();
  if (!cachedData) return;

  const [sx, sy] = cachedData.start;
  const [gx, gy] = cachedData.goal;

  // Start: lime square
  g.append('rect').attr('class','start-marker')
    .attr('x', xS(sx) - 6).attr('y', yS(sy) - 6)
    .attr('width', 12).attr('height', 12)
    .attr('rx', 2);
  // Goal: gold star (unicode path)
  g.append('text')
    .attr('x', xS(gx)).attr('y', yS(gy) + 6)
    .attr('text-anchor', 'middle').attr('font-size', 18)
    .attr('fill', '#ffc107').text('★');
}

/* ════════════════════════════════════════════════════════════════════════
   RENDERING
   ════════════════════════════════════════════════════════════════════════ */

function renderCurrentStep() {
  if (!cachedData) return;
  const step = currentStep;

  const priorTraj = cachedData.prior;            // [T,2]
  const plainTraj = cachedData.plain_history[step]; // [T,2]
  const safeTraj  = cachedData.safe_history[step];  // [T,2]
  const cbf       = cachedData.cbf_step_data[step];

  currentSafeTraj = safeTraj;

  renderPrior(priorTraj);
  renderPlain(plainTraj);
  renderSafePath(safeTraj);
  renderGradientArrows(safeTraj, cbf);
  renderWaypoints(safeTraj, cbf);
  drawObstacle();
  drawMarkers();
  updateInspector(cbf);
}

const lineGen = d3.line().x(d => xS(d[0])).y(d => yS(d[1]));

function renderPrior(traj) {
  const g = d3.select('#layer-prior');
  g.selectAll('*').remove();
  g.append('path').attr('class','prior-path').attr('d', lineGen(traj));
}

function renderPlain(traj) {
  const g = d3.select('#layer-plain');
  g.selectAll('*').remove();
  g.append('path').attr('class','plain-path').attr('d', lineGen(traj));
}

function renderSafePath(traj) {
  const g = d3.select('#layer-safe-path');
  g.selectAll('*').remove();
  g.append('path').attr('class','safe-path').attr('d', lineGen(traj));
}

function renderGradientArrows(traj, cbf) {
  const g = d3.select('#layer-grads');
  g.selectAll('*').remove();
  if (!cbf || !cbf.grad_x) return;

  const gx = cbf.grad_x, gy = cbf.grad_y;
  const mags = gx.map((v, i) => Math.sqrt(v * v + gy[i] * gy[i]));
  const maxMag = Math.max(...mags, 1e-10);
  const MAX_PX = 30;

  traj.forEach(([wx, wy], i) => {
    const mag = mags[i];
    if (mag < 1e-8) return;
    const len = (mag / maxMag) * MAX_PX;
    const nx = gx[i] / mag;
    const ny = gy[i] / mag;   // world-space

    const x1 = xS(wx),  y1 = yS(wy);
    const x2 = x1 + nx * len;
    const y2 = y1 - ny * len;  // SVG y inverted

    const isSafe = cbf.d_raw[i] >= 0;
    const cls    = isSafe ? 'grad-arrow-safe' : 'grad-arrow-danger';
    const marker = isSafe ? 'url(#arrow-safe)' : 'url(#arrow-danger)';

    g.append('line').attr('class', cls)
      .attr('x1', x1).attr('y1', y1)
      .attr('x2', x2).attr('y2', y2)
      .attr('marker-end', marker);
  });
}

function renderWaypoints(traj, cbf) {
  const g = d3.select('#layer-waypoints');
  g.selectAll('*').remove();
  if (!traj) return;

  const circles = g.selectAll('circle').data(traj).join('circle')
    .attr('cx', d => xS(d[0]))
    .attr('cy', d => yS(d[1]))
    .attr('r', 4)
    .attr('class', (d, i) => {
      const safe = !cbf || cbf.d_raw[i] >= 0;
      return safe ? 'waypoint-safe' : 'waypoint-danger';
    });

  // Drag behaviour
  const drag = d3.drag()
    .on('start', function(event, d) {
      draggedIdx = traj.indexOf(d);
    })
    .on('drag', function(event) {
      if (draggedIdx < 0) return;
      const wx = xS.invert(event.x);
      const wy = yS.invert(event.y);

      if (event.sourceEvent && event.sourceEvent.ctrlKey) {
        currentSafeTraj = gaussianDeform(currentSafeTraj, draggedIdx, wx, wy);
      } else {
        currentSafeTraj = currentSafeTraj.map((pt, i) =>
          i === draggedIdx ? [wx, wy] : pt
        );
      }

      renderSafePath(currentSafeTraj);
      renderWaypoints(currentSafeTraj, null);
      debouncedMathUpdate();
    })
    .on('end', () => { draggedIdx = -1; });

  circles.call(drag);
}

/* ── Gaussian spline deformation (Ctrl+drag) ───────────────────────────── */
function gaussianDeform(traj, anchorIdx, newX, newY) {
  const sigma = 6;
  const [ax, ay] = traj[anchorIdx];
  const dx = newX - ax, dy = newY - ay;
  return traj.map((pt, i) => {
    const w = Math.exp(-((i - anchorIdx) ** 2) / (2 * sigma * sigma));
    return [pt[0] + w * dx, pt[1] + w * dy];
  });
}

/* ════════════════════════════════════════════════════════════════════════
   STEP INSPECTOR
   ════════════════════════════════════════════════════════════════════════ */

function updateInspector(cbf) {
  if (!cbf) {
    tbody.innerHTML = '';
    hxtValue.textContent = '—';
    hxtValue.className = 'hxt-neutral';
    return;
  }

  const hxt = cbf.h_Xt;
  hxtValue.textContent = hxt.toFixed(5);
  hxtValue.className   = hxt < 0 ? 'hxt-danger' : 'hxt-safe';

  const T = cbf.d_raw.length;
  const rows = [];
  for (let i = 0; i < T; i++) {
    const dr = cbf.d_raw[i];
    const dt = cbf.d_tilde[i];
    const hw = cbf.h_wi[i];
    const si = cbf.sigma_i[i];
    const gx = cbf.grad_x[i];
    const gy = cbf.grad_y[i];
    const gn = Math.sqrt(gx * gx + gy * gy);
    const safe = dr >= 0;

    rows.push(`<tr>
      <td>${i + 1}</td>
      <td class="${safe ? 'safe-cell' : 'danger-cell'}">${dr.toFixed(4)}</td>
      <td>${dt.toFixed(4)}</td>
      <td>${hw.toFixed(4)}</td>
      <td>${si.toFixed(5)}</td>
      <td>${gx.toFixed(4)}</td>
      <td>${gy.toFixed(4)}</td>
      <td class="bold-cell">${gn.toFixed(4)}</td>
    </tr>`);
  }
  tbody.innerHTML = rows.join('');
}

/* ════════════════════════════════════════════════════════════════════════
   EVENT BINDINGS
   ════════════════════════════════════════════════════════════════════════ */

function bindRunButton() {
  runBtn.addEventListener('click', runOptimisation);
}

function bindScrubber() {
  scrubber.addEventListener('input', () => {
    currentStep = parseInt(scrubber.value, 10);
    stepLabel.textContent = currentStep;
    if (cachedData) renderCurrentStep();
  });
}

function bindKeyboardCursor() {
  document.addEventListener('keydown', e => { if (e.key === 'Control') document.body.classList.add('ctrl-drag'); });
  document.addEventListener('keyup',   e => { if (e.key === 'Control') document.body.classList.remove('ctrl-drag'); });
}

function bindParamControls() {
  const bind = (id, key, displayId, toFixed) => {
    const el = document.getElementById(id);
    const valEl = displayId ? document.getElementById(displayId) : null;
    el.addEventListener('input', () => {
      params[key] = parseFloat(el.value);
      if (valEl) valEl.textContent = params[key].toFixed(toFixed);
      drawObstacle();
      if (currentSafeTraj) debouncedMathUpdate();
    });
  };
  bind('c-slider',  'c',           'c-val',  2);
  bind('k1-slider', 'k1',          'k1-val', 2);
  bind('k2-input',  'k2',          null,     2);
  bind('r-slider',  'r',           'r-val',  2);
  bind('gd-slider', 'gamma_delta', 'gd-val', 2);

  document.getElementById('k2-input').addEventListener('change', () => {
    params.k2 = parseFloat(document.getElementById('k2-input').value);
    if (currentSafeTraj) debouncedMathUpdate();
  });
}

/* ════════════════════════════════════════════════════════════════════════
   API CALLS
   ════════════════════════════════════════════════════════════════════════ */

async function runOptimisation() {
  const modelName = document.getElementById('model-select').value;
  if (!modelName) { alert('Select a model first.'); return; }

  const nSteps = parseInt(document.getElementById('n-steps-input').value, 10);

  setLoading(true);

  const body = {
    model_name:  modelName,
    n_steps:     nSteps,
    c:           params.c,
    k1:          params.k1,
    k2:          params.k2,
    r:           params.r,
    gamma_delta: params.gamma_delta,
    obs_x:       obstaclePos.x,
    obs_y:       obstaclePos.y,
    start_x:     -0.8,
    start_y:     -0.8,
    goal_x:       0.8,
    goal_y:       0.8,
  };

  try {
    const res  = await fetch(`${API}/api/run`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    cachedData  = await res.json();
    currentStep = 0;

    scrubber.max      = cachedData.n_steps;
    scrubber.value    = 0;
    scrubber.disabled = false;
    stepLabel.textContent = '0';

    renderCurrentStep();
  } catch (e) {
    alert(`Run failed: ${e.message}`);
    console.error(e);
  } finally {
    setLoading(false);
  }
}

async function mathUpdate() {
  if (!currentSafeTraj) return;
  const body = {
    traj:        currentSafeTraj,
    c:           params.c,
    k1:          params.k1,
    k2:          params.k2,
    r:           params.r,
    gamma_delta: params.gamma_delta,
    obs_x:       obstaclePos.x,
    obs_y:       obstaclePos.y,
  };

  try {
    const res = await fetch(`${API}/api/math`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!res.ok) return;
    const cbf = await res.json();

    // Patch cbf into cached step so scrubbing shows updated values
    if (cachedData) {
      cachedData.cbf_step_data[currentStep] = cbf;
      cachedData.safe_history[currentStep]  = currentSafeTraj;
    }

    renderGradientArrows(currentSafeTraj, cbf);
    renderWaypoints(currentSafeTraj, cbf);
    drawObstacle();
    updateInspector(cbf);
  } catch (e) {
    console.warn('Math update failed:', e);
  }
}

/* ── Debounce math updates to avoid flooding the API ───────────────────── */
let _mathTimer = null;
function debouncedMathUpdate() {
  clearTimeout(_mathTimer);
  _mathTimer = setTimeout(mathUpdate, 80);
}

/* ── Loading helpers ────────────────────────────────────────────────────── */
function setLoading(on) {
  overlay.classList.toggle('active', on);
  runBtn.disabled    = on;
  scrubber.disabled  = on;
}
