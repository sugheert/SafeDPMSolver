/* ══════════════════════════════════════════════════════════════════════
   SafeDPMSolver — Three-Panel Visualiser
   Plain DPM (left)  |  DPM-CBF Before Control (middle, interactive)
   DPM-CBF After Control (right, passive)  |  Inspector
   ══════════════════════════════════════════════════════════════════════ */

const API = '';

/* ── Global state ─────────────────────────────────────────────────── */
let cachedData = null;
let currentStep = 0;
let currentBeforeTraj = null;  // editable before-control traj; reset on step change
let currentSafeTraj = null;   // after-control traj; updated by recompute
let draggedIdx = -1;
let isCtrlHeld = false;

/* ── Per-SVG zoom transforms (persist across buildCanvases rebuilds) ── */
const zoomStates = {};   // keyed by SVG id → d3.ZoomTransform

const OBS_OFFSETS = [[0,0],[0.4,0],[-0.4,0],[0,0.4],[0,-0.4],[0.4,0.4],[-0.4,0.4]];
let obstacles = [{ x: 0.0, y: 0.0, a: 0.3, b: 0.3 }];
let params = { c: 1.0, k1: 1.0, k2: 1.0, gamma_delta: 0.05, alpha0: 1.0 };

function updateObsUI() {
  const container = document.getElementById('obs-list');
  if (!container) return;
  container.innerHTML = '';
  obstacles.forEach((obs, idx) => {
    const div = document.createElement('div');
    div.style.cssText = 'border:1px solid #444;padding:4px;margin-bottom:4px;font-size:12px;border-radius:3px';
    div.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <b>Obs ${idx + 1}</b>
        <button class="step-btn" style="padding:2px 5px;font-size:10px" onclick="removeObs(${idx})">✕</button>
      </div>
      <div style="display:flex;gap:4px">
        <label>a: <input type="number" step="0.05" value="${obs.a.toFixed(2)}" onchange="changeObs(${idx}, 'a', this.value)" style="width:45px"></label>
        <label>b: <input type="number" step="0.05" value="${obs.b.toFixed(2)}" onchange="changeObs(${idx}, 'b', this.value)" style="width:45px"></label>
      </div>
    `;
    container.appendChild(div);
  });

  // "+" button always at the bottom, after all obstacle cards
  const addDiv = document.createElement('div');
  addDiv.innerHTML = `<button class="step-btn" style="width:100%;font-size:13px;padding:3px 0" onclick="addObs()">+ Add Obstacle</button>`;
  container.appendChild(addDiv);
}

function addObs() {
  const off = OBS_OFFSETS[obstacles.length % OBS_OFFSETS.length];
  obstacles.push({ x: off[0], y: off[1], a: 0.25, b: 0.25 });
  updateObsUI();
  drawObstacleOnAll();
  if (currentBeforeTraj) debouncedRecomputeUpdate();
}

function removeObs(idx) {
  obstacles.splice(idx, 1);
  updateObsUI();
  drawObstacleOnAll();
  if (currentBeforeTraj) debouncedRecomputeUpdate();
}

function changeObs(idx, key, val) {
  obstacles[idx][key] = Math.max(0.01, parseFloat(val) || 0.1);
  drawObstacleOnAll();
  if (currentBeforeTraj) debouncedRecomputeUpdate();
}


/* ── World bounds ─────────────────────────────────────────────────── */
const WORLD_MIN = -1.5;
const WORLD_MAX = 1.5;

/* ── Maze walls (populated by fetchMaze) ─────────────────────────── */
let walls = [];
let viewMin = WORLD_MIN, viewMax = WORLD_MAX;

/* ── Three SVG selections ─────────────────────────────────────────── */
const svgP = d3.select('#plain-svg');    // plain (left)
const svgB = d3.select('#before-svg');   // before-control (middle, interactive)
const svgA = d3.select('#after-svg');    // after-control  (right, passive)

/* ── Scale functions  — shared size, updated in buildCanvases() ───── */
let SZ = 400;
const xS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([44, SZ - 16]);
const yS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([SZ - 32, 16]);

/* ── DOM refs ─────────────────────────────────────────────────────── */
const overlay = document.getElementById('loading-overlay');
const runBtn = document.getElementById('run-btn');
const scrubber = document.getElementById('step-scrubber');
const stepLabel = document.getElementById('step-label');
const stepMax = document.getElementById('step-max');
const hxtValue = document.getElementById('hxt-value');
const minHwiValue = document.getElementById('min-hwi-value');
const sigmaDeltaValue = document.getElementById('sigma-delta-value');
const sigmaDotValue = document.getElementById('sigma-dot-value');
const stepDeltaValue = document.getElementById('step-delta-value');
const omegaValue = document.getElementById('omega-value');
const tbody = document.getElementById('inspector-tbody');
const btnPrev = document.getElementById('step-prev');
const btnNext = document.getElementById('step-next');

/* ════════════════════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════════════════════ */

const DEFAULTS_KEY = 'safedpm_defaults';

window.addEventListener('DOMContentLoaded', () => {
  loadDefaults();        // restore saved params before binding controls
  fetchModels();
  fetchEnvs();
  buildCanvases();
  updateObsUI();
  window.changeObs = changeObs; // expose globally for inline onclick handlers
  window.removeObs = removeObs;
  window.addObs = addObs;
  bindParamControls();
  bindRunButton();
  bindBatchButton();
  bindScrubber();
  bindKeyboard();
  bindStepButtons();
  bindInspectorResize();
  bindParamResize();
  bindPanelDrag();
  bindSaveDefaults();
  window.addEventListener('resize', () => { buildCanvases(); if (cachedData) renderCurrentStep(); });
});

/* ════════════════════════════════════════════════════════════════════
   DEFAULTS  (localStorage persistence)
   ════════════════════════════════════════════════════════════════════ */

function saveDefaults() {
  const snapshot = {
    params: { ...params },
    obstacles: JSON.parse(JSON.stringify(obstacles)),
    n_steps: document.getElementById('n-steps-input').value,
    model_name: document.getElementById('model-select').value,
  };
  localStorage.setItem(DEFAULTS_KEY, JSON.stringify(snapshot));

  // Brief visual confirmation
  const btn = document.getElementById('save-defaults-btn');
  btn.classList.add('saved');
  btn.textContent = '✓ Saved';
  setTimeout(() => { btn.classList.remove('saved'); btn.textContent = '★ Save as Default'; }, 1500);
}

function loadDefaults() {
  const raw = localStorage.getItem(DEFAULTS_KEY);
  if (!raw) return;
  try {
    const snap = JSON.parse(raw);

    // Restore params object
    if (snap.params) Object.assign(params, snap.params);

    // Restore obstacles
    if (snap.obstacles && Array.isArray(snap.obstacles)) {
      obstacles = snap.obstacles;
    }

    // Sync all slider/number DOM elements
    const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    setEl('c-slider', params.c);
    setEl('c-val', params.c.toFixed(2));
    setEl('k1-input', params.k1);
    setEl('k2-input', params.k2);
    setEl('gd-slider', params.gamma_delta);
    setEl('gd-val', params.gamma_delta.toFixed(2));
    setEl('alpha0-input', params.alpha0);
    if (snap.n_steps) setEl('n-steps-input', snap.n_steps);

    // Model selection — must wait until options are populated
    if (snap.model_name) {
      const trySelect = () => {
        const sel = document.getElementById('model-select');
        const opt = [...sel.options].find(o => o.value === snap.model_name);
        if (opt) { sel.value = snap.model_name; }
        else { setTimeout(trySelect, 100); }   // retry until options load
      };
      trySelect();
    }
  } catch (e) { console.warn('Failed to load defaults:', e); }
}

function bindSaveDefaults() {
  document.getElementById('save-defaults-btn')
    .addEventListener('click', saveDefaults);
}

/* ── Environment / maze list ──────────────────────────────────────── */
async function fetchEnvs() {
  try {
    const res = await fetch(`${API}/api/envs`);
    const data = await res.json();
    const sel = document.getElementById('env-select');
    (data.envs || []).forEach(e => {
      const opt = document.createElement('option');
      opt.value = e; opt.textContent = e;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => fetchMaze(sel.value));
  } catch (e) { console.warn('Envs fetch failed:', e); }
}

async function fetchMaze(envId) {
  if (!envId) {
    walls = [];
    xS.domain([WORLD_MIN, WORLD_MAX]);
    yS.domain([WORLD_MIN, WORLD_MAX]);
    buildCanvases();
    if (cachedData) renderCurrentStep();
    return;
  }
  try {
    const res = await fetch(`${API}/api/maze?env_id=${encodeURIComponent(envId)}`);
    const data = await res.json();
    walls = data.walls;
    viewMin = data.view_min; viewMax = data.view_max;
    xS.domain([viewMin, viewMax]);
    yS.domain([viewMin, viewMax]);
    buildCanvases();
    if (cachedData) renderCurrentStep();
  } catch (e) { console.warn('Maze fetch failed:', e); }
}

function renderWallsOn(svgSel) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-walls`);
  g.selectAll('*').remove();
  walls.forEach(w => {
    g.append('rect')
      .attr('class', 'wall-rect')
      .attr('x', xS(w.cx - w.hw))
      .attr('y', yS(w.cy + w.hh))
      .attr('width',  Math.abs(xS(w.cx + w.hw) - xS(w.cx - w.hw)))
      .attr('height', Math.abs(yS(w.cy - w.hh) - yS(w.cy + w.hh)));
  });
}

function renderWallsOnAll() {
  renderWallsOn(svgP);
  renderWallsOn(svgB);
  renderWallsOn(svgA);
}

/* ── Model list ───────────────────────────────────────────────────── */
async function fetchModels() {
  try {
    const res = await fetch(`${API}/api/models`);
    const data = await res.json();
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    });
    if (!data.models || !data.models.length)
      sel.innerHTML = '<option value="">No checkpoints found</option>';
  } catch (e) { console.warn('Models fetch failed:', e); }
}

/* ════════════════════════════════════════════════════════════════════
   CANVAS CONSTRUCTION  (called on load and window resize)
   ════════════════════════════════════════════════════════════════════ */

function buildCanvases() {
  // Compute square canvas size from the smallest container
  const cP = document.getElementById('plain-container');
  const cBe = document.getElementById('before-container');
  const cAf = document.getElementById('after-container');
  const w = Math.min(cP.clientWidth, cBe.clientWidth, cAf.clientWidth) - 4;
  const h = Math.min(cP.clientHeight, cBe.clientHeight, cAf.clientHeight) - 4;
  SZ = Math.max(Math.min(w, h), 200);

  xS.range([44, SZ - 16]);
  yS.range([SZ - 32, 16]);

  initSvg(svgP);
  initSvg(svgB);
  initSvg(svgA);

  applyZoom(svgP);
  applyZoom(svgB);
  applyZoom(svgA);

  renderWallsOnAll();
  drawObstacleOnAll();
  drawMarkersOn(svgP);
  drawMarkersOn(svgB);
  drawMarkersOn(svgA);

  // Attach canvas-wide drag for waypoints (before-canvas only)
  attachBeforeDrag();
}

/* ── Initialise one SVG: size, defs, grid, axes, empty layers ─────── */
function initSvg(svgSel) {
  svgSel.attr('width', SZ).attr('height', SZ);
  svgSel.selectAll('*').remove();

  // Arrow marker defs sit outside the zoom layer (IDs remain stable)
  const defs = svgSel.append('defs');
  [['safe', '#3ddc84'], ['danger', '#ff5370']].forEach(([cls, col]) => {
    defs.append('marker')
      .attr('id', `arrow-${cls}-${svgSel.attr('id')}`)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', col);
  });

  // Everything zoomable lives inside this group
  const zl = svgSel.append('g').attr('class', 'zoom-layer');

  drawGrid(zl);
  drawAxes(zl);

  // Layer order: walls first (background), then trajectories, obstacle, waypoints on top.
  ['walls', 'prior', 'main-path', 'grads', 'obstacle', 'waypoints', 'markers'].forEach(name =>
    zl.append('g').attr('id', `${svgSel.attr('id')}-${name}`)
  );
}

function applyZoom(svgSel) {
  const id = svgSel.attr('id');
  const behavior = d3.zoom()
    .scaleExtent([0.5, 20])
    .filter(e => e.type === 'wheel')          // scroll-to-zoom only; drag stays for waypoints
    .on('zoom', event => {
      svgSel.select('.zoom-layer').attr('transform', event.transform);
      zoomStates[id] = event.transform;
    });
  svgSel.call(behavior);
  svgSel.on('dblclick.zoom', null);           // disable d3 default dblclick zoom
  svgSel.on('dblclick', () => {               // double-click = reset zoom
    behavior.transform(svgSel, d3.zoomIdentity);
    zoomStates[id] = d3.zoomIdentity;
  });
  // Restore zoom state that survived a buildCanvases() rebuild
  if (zoomStates[id]) behavior.transform(svgSel, zoomStates[id]);
}

/* Convert SVG-element pixel coords → world coords, accounting for zoom */
function svgToWorld(px, py, svgSel) {
  const zt = zoomStates[svgSel.attr('id')] || d3.zoomIdentity;
  const [ox, oy] = zt.invert([px, py]);
  return [xS.invert(ox), yS.invert(oy)];
}

function drawGrid(containerSel) {
  const g = containerSel.append('g');
  for (let v = WORLD_MIN; v <= WORLD_MAX + 0.001; v += 0.5) {
    g.append('line').attr('class', 'grid-line')
      .attr('x1', xS(v)).attr('y1', yS(WORLD_MIN))
      .attr('x2', xS(v)).attr('y2', yS(WORLD_MAX));
    g.append('line').attr('class', 'grid-line')
      .attr('x1', xS(WORLD_MIN)).attr('y1', yS(v))
      .attr('x2', xS(WORLD_MAX)).attr('y2', yS(v));
  }
}

function drawAxes(containerSel) {
  const g = containerSel.append('g');
  const ox = xS(0), oy = yS(0);
  g.append('line').attr('class', 'axis-line')
    .attr('x1', xS(WORLD_MIN)).attr('y1', oy)
    .attr('x2', xS(WORLD_MAX)).attr('y2', oy);
  g.append('line').attr('class', 'axis-line')
    .attr('x1', ox).attr('y1', yS(WORLD_MIN))
    .attr('x2', ox).attr('y2', yS(WORLD_MAX));
  [-1, -0.5, 0.5, 1].forEach(v => {
    g.append('text').attr('class', 'axis-label')
      .attr('x', xS(v)).attr('y', oy + 13).attr('text-anchor', 'middle').text(v);
    g.append('text').attr('class', 'axis-label')
      .attr('x', ox - 5).attr('y', yS(v) + 4).attr('text-anchor', 'end').text(v);
  });
}

/* ── Degree-4 superellipse path ────────────────────────────────────── */
// Parametric: x = cx + aPx * sign(cos t) * |cos t|^(2/n)
//             y = cy + bPx * sign(sin t) * |sin t|^(2/n)
// For n=4, exponent = 0.5  →  squircle-like rounded shape.
function superellipsePath(cx, cy, aPx, bPx, n = 4, nPts = 120) {
  const exp = 2 / n;
  let d = '';
  for (let i = 0; i <= nPts; i++) {
    const t = (2 * Math.PI * i) / nPts;
    const ct = Math.cos(t), st = Math.sin(t);
    const x = cx + aPx * Math.sign(ct) * Math.pow(Math.abs(ct), exp);
    const y = cy + bPx * Math.sign(st) * Math.pow(Math.abs(st), exp);
    d += (i === 0 ? 'M' : 'L') + `${x.toFixed(2)},${y.toFixed(2)} `;
  }
  return d + 'Z';
}

/* ── Obstacle ─────────────────────────────────────────────────────── */

function drawObstacleOn(svgSel, draggable) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-obstacle`);
  g.selectAll('*').remove();

  obstacles.forEach((obs, idx) => {
    const cx = xS(obs.x), cy = yS(obs.y);
    const aPx = Math.abs(xS(obs.a) - xS(0));
    const bPx = Math.abs(yS(obs.b) - yS(0));
    const maPx = Math.abs(xS(obs.a + params.gamma_delta) - xS(0));
    const mbPx = Math.abs(yS(obs.b + params.gamma_delta) - yS(0));

    g.append('path').attr('class', `obs-fill${draggable ? ' obs-drag' : ''}`)
      .attr('data-idx', idx).attr('d', superellipsePath(cx, cy, aPx, bPx));
    g.append('path').attr('class', `obs-ring${draggable ? ' obs-drag' : ''}`)
      .attr('data-idx', idx).attr('d', superellipsePath(cx, cy, aPx, bPx));
    g.append('path').attr('class', 'obs-margin')
      .attr('data-idx', idx).attr('d', superellipsePath(cx, cy, maPx, mbPx));

    if (draggable) {
      const drag = d3.drag()
        .container(svgSel.node())
        .on('drag', function (event) {
          event.sourceEvent.stopPropagation();
          const [wx, wy] = svgToWorld(event.x, event.y, svgSel);
          obstacles[idx].x = wx;
          obstacles[idx].y = wy;
          drawObstacleOnAll();
          if (currentBeforeTraj) debouncedRecomputeUpdate();
        });
      g.selectAll(`.obs-fill[data-idx="${idx}"], .obs-ring[data-idx="${idx}"]`).call(drag);
    }
  });
}

function drawObstacleOnAll() {
  drawObstacleOn(svgP, false);   // plain: non-draggable, just visual
  drawObstacleOn(svgB, true);    // before: draggable
  drawObstacleOn(svgA, false);   // after: non-draggable
}

/* ── Start / Goal markers ─────────────────────────────────────────── */
function drawMarkersOn(svgSel) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-markers`);
  g.selectAll('*').remove();
  if (!cachedData) return;
  const [sx, sy] = cachedData.start;
  const [gx, gy] = cachedData.goal;
  g.append('rect').attr('class', 'start-mk')
    .attr('x', xS(sx) - 6).attr('y', yS(sy) - 6).attr('width', 12).attr('height', 12).attr('rx', 2);
  g.append('text')
    .attr('x', xS(gx)).attr('y', yS(gy) + 6).attr('text-anchor', 'middle')
    .attr('font-size', 18).attr('fill', '#ffc107').text('★');
}

/* ════════════════════════════════════════════════════════════════════
   CANVAS-WIDE DRAG ON BEFORE SVG
   ════════════════════════════════════════════════════════════════════ */

function attachBeforeDrag() {
  const drag = d3.drag()
    .container(svgB.node())
    .filter(function (event) {
      if (!currentBeforeTraj) return false;
      // Ignore if click originated on the draggable obstacle
      let el = event.sourceEvent && event.sourceEvent.target;
      while (el) { if (el.id === 'before-svg-obstacle') return false; el = el.parentElement; }
      return true;
    })
    .on('start', function (event) {
      const [wx, wy] = svgToWorld(event.x, event.y, svgB);
      draggedIdx = findNearest(wx, wy, currentBeforeTraj);
    })
    .on('drag', function (event) {
      if (draggedIdx < 0) return;
      const [wx, wy] = svgToWorld(event.x, event.y, svgB);

      if (isCtrlHeld) {
        currentBeforeTraj = gaussianDeform(currentBeforeTraj, draggedIdx, wx, wy);
      } else {
        currentBeforeTraj = currentBeforeTraj.map((pt, i) => i === draggedIdx ? [wx, wy] : pt);
      }

      renderBeforePath(currentBeforeTraj);
      renderWaypoints(svgB, currentBeforeTraj, null);
      debouncedRecomputeUpdate();
    })
    .on('end', () => { draggedIdx = -1; });

  svgB.call(drag);
}

function findNearest(wx, wy, traj) {
  let minD = Infinity, idx = 0;
  traj.forEach(([x, y], i) => {
    const d = (x - wx) ** 2 + (y - wy) ** 2;
    if (d < minD) { minD = d; idx = i; }
  });
  return idx;
}

function gaussianDeform(traj, anchor, newX, newY) {
  const sigma = 6;
  const [ax, ay] = traj[anchor];
  const dx = newX - ax, dy = newY - ay;
  return traj.map((pt, i) => {
    const w = Math.exp(-((i - anchor) ** 2) / (2 * sigma * sigma));
    return [pt[0] + w * dx, pt[1] + w * dy];
  });
}

/* ════════════════════════════════════════════════════════════════════
   RENDERING
   ════════════════════════════════════════════════════════════════════ */

function renderCurrentStep() {
  if (!cachedData) return;

  const prior = cachedData.prior;
  const plainTraj = cachedData.plain_history[currentStep];
  const beforeTraj = cachedData.before_history[currentStep];
  const afterTraj = cachedData.safe_history[currentStep];
  const cbf = cachedData.cbf_step_data[currentStep];

  // Deep copy so edits don't corrupt the cache; also resets any dragged state
  currentBeforeTraj = beforeTraj.map(pt => [pt[0], pt[1]]);
  currentSafeTraj = afterTraj.map(pt => [pt[0], pt[1]]);

  // ── Plain canvas ──────────────────────────────────────────────────
  renderWallsOn(svgP);
  renderPathOn(svgP, 'prior', prior, 'prior-path');
  renderPathOn(svgP, 'main-path', plainTraj, 'plain-path');
  drawObstacleOn(svgP, false);
  drawMarkersOn(svgP);

  // ── Before-control canvas (interactive) ───────────────────────────
  renderWallsOn(svgB);
  renderPathOn(svgB, 'prior', prior, 'prior-path');
  renderBeforePath(currentBeforeTraj);
  renderGradArrows(svgB, currentBeforeTraj, cbf);
  renderWaypoints(svgB, currentBeforeTraj, cbf);
  drawObstacleOn(svgB, true);
  drawMarkersOn(svgB);

  // ── After-control canvas (passive) ────────────────────────────────
  renderWallsOn(svgA);
  renderPathOn(svgA, 'prior', prior, 'prior-path');
  renderAfterPath(currentSafeTraj);
  renderWaypoints(svgA, currentSafeTraj, cbf);
  drawObstacleOn(svgA, false);
  drawMarkersOn(svgA);

  updateInspector(cbf);
}

const lineGen = d3.line().x(d => xS(d[0])).y(d => yS(d[1]));

function renderPathOn(svgSel, layerName, traj, cssClass) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-${layerName}`);
  g.selectAll('*').remove();
  if (!traj) return;
  g.append('path').attr('class', cssClass).attr('d', lineGen(traj));
}

function renderBeforePath(traj) {
  renderPathOn(svgB, 'main-path', traj, 'before-path');
}

function renderAfterPath(traj) {
  renderPathOn(svgA, 'main-path', traj, 'after-path');
}

function renderGradArrows(svgSel, traj, cbf) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-grads`);
  g.selectAll('*').remove();
  if (!cbf || !cbf.grad_x) return;

  const gx = cbf.grad_x, gy = cbf.grad_y;
  const mags = gx.map((v, i) => Math.sqrt(v * v + gy[i] * gy[i]));
  const maxMag = Math.max(...mags, 1e-10);
  const MAX_PX = 28;
  const svgId = svgSel.attr('id');

  traj.forEach(([wx, wy], i) => {
    const mag = mags[i];
    if (mag < 1e-8) return;
    const len = (mag / maxMag) * MAX_PX;
    const nx = gx[i] / mag, ny = gy[i] / mag;
    const x1 = xS(wx), y1 = yS(wy);
    const x2 = x1 + nx * len, y2 = y1 - ny * len;
    const safe = cbf.d_raw[i] >= 0;
    g.append('line')
      .attr('class', safe ? 'arrow-safe' : 'arrow-danger')
      .attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
      .attr('marker-end', `url(#arrow-${safe ? 'safe' : 'danger'}-${svgId})`);
  });
}

const tooltip = document.getElementById('wp-tooltip');

function renderWaypoints(svgSel, traj, cbf) {
  const id = svgSel.attr('id');
  const g = svgSel.select(`#${id}-waypoints`);
  g.selectAll('*').remove();
  if (!traj) return;

  g.selectAll('circle')
    .data(traj.map((pt, i) => ({ pt, i })))
    .join('circle')
    .attr('cx', d => xS(d.pt[0]))
    .attr('cy', d => yS(d.pt[1]))
    .attr('r', 5)
    .attr('class', d => (!cbf || cbf.d_raw[d.i] >= 0) ? 'wp-safe' : 'wp-danger')
    .on('mouseenter', function (event, d) {
      const i = d.i;
      const [wx, wy] = d.pt;
      const safe = !cbf || cbf.d_raw[i] >= 0;
      const hw = cbf ? cbf.h_wi[i] : null;
      const gx = cbf ? cbf.grad_x[i] : null;
      const gy = cbf ? cbf.grad_y[i] : null;
      const gn = (gx !== null) ? Math.sqrt(gx * gx + gy * gy) : null;
      const cls = safe ? 'tt-safe' : 'tt-danger';
      const fmt = v => v !== null ? v.toFixed(5) : '—';

      tooltip.innerHTML = `
        <div class="tt-title">Waypoint ${i + 1}</div>
        <div><span class="tt-dim">x =</span> ${wx.toFixed(5)}</div>
        <div><span class="tt-dim">y =</span> ${wy.toFixed(5)}</div>
        <div><span class="tt-dim">h(w<sub>${i + 1}</sub>) =</span> <span class="${cls}">${fmt(hw)}</span></div>
        <div><span class="tt-dim">∂h/∂x =</span> ${fmt(gx)}</div>
        <div><span class="tt-dim">∂h/∂y =</span> ${fmt(gy)}</div>
        <div><span class="tt-dim">|∇h|  =</span> ${fmt(gn)}</div>`;

      tooltip.style.display = 'block';
      positionTooltip(event);
    })
    .on('mousemove', positionTooltip)
    .on('mouseleave', () => { tooltip.style.display = 'none'; });
}

function positionTooltip(event) {
  const pad = 14;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  let lx = event.clientX + pad;
  let ly = event.clientY + pad;
  if (lx + tw > window.innerWidth) lx = event.clientX - tw - pad;
  if (ly + th > window.innerHeight) ly = event.clientY - th - pad;
  tooltip.style.left = lx + 'px';
  tooltip.style.top = ly + 'px';
}

/* ════════════════════════════════════════════════════════════════════
   STEP INSPECTOR
   ════════════════════════════════════════════════════════════════════ */

function updateInspector(cbf) {
  if (!cbf) {
    tbody.innerHTML = '';
    hxtValue.textContent = '—'; hxtValue.className = 'hxt-neutral';
    minHwiValue.textContent = '—'; minHwiValue.className = 'hxt-neutral';
    sigmaDeltaValue.textContent = '—'; sigmaDeltaValue.className = 'hxt-neutral';
    sigmaDotValue.textContent = '—'; sigmaDotValue.className = 'hxt-neutral';
    stepDeltaValue.textContent = '—'; stepDeltaValue.className = 'hxt-neutral';
    omegaValue.textContent = '—'; omegaValue.className = 'hxt-neutral';
    return;
  }

  // h(Xt)
  const hxt = cbf.h_Xt;
  hxtValue.textContent = hxt.toFixed(5);
  hxtValue.className = hxt < 0 ? 'hxt-danger' : 'hxt-safe';

  // min h(w_i)
  const minHwi = Math.min(...cbf.h_wi);
  minHwiValue.textContent = minHwi.toFixed(5);
  minHwiValue.className = minHwi < 0 ? 'hxt-danger' : 'hxt-safe';

  // sigma_delta, sigma_dot, step_delta
  const sd = cbf.sigma_delta ?? null;
  const sdot = cbf.sigma_dot ?? null;
  const sdt = cbf.step_delta ?? null;
  sigmaDeltaValue.textContent = sd !== null ? sd.toFixed(5) : '—';
  sigmaDotValue.textContent = sdot !== null ? sdot.toFixed(5) : '—';
  stepDeltaValue.textContent = sdt !== null ? sdt.toString() : '—';
  sigmaDeltaValue.className = 'hxt-neutral';
  sigmaDotValue.className = 'hxt-neutral';
  stepDeltaValue.className = 'hxt-neutral';

  // omega (scalar, may be null for step-0)
  const omega = cbf.omega ?? null;
  if (omega === null) {
    omegaValue.textContent = '—';
    omegaValue.className = 'hxt-neutral';
  } else {
    omegaValue.textContent = omega.toFixed(5);
    omegaValue.className = omega < 0 ? 'omega-neg' : (omega === 0 ? 'omega-zero' : 'omega-pos');
  }

  const hasCtrls = cbf.ctrl_x && cbf.ctrl_raw_x;

  const T = cbf.d_raw.length;
  const sdVal = cbf.sigma_delta ?? null;
  const rows = cbf.d_raw.map((dr, i) => {
    const dt = cbf.d_tilde[i], hw = cbf.h_wi[i], si = cbf.sigma_i[i];
    const gx = cbf.grad_x[i], gy = cbf.grad_y[i];
    const gn = Math.sqrt(gx * gx + gy * gy);
    const cx = hasCtrls ? cbf.ctrl_x[i] : null;
    const cy = hasCtrls ? cbf.ctrl_y[i] : null;
    const cx0 = hasCtrls ? cbf.ctrl_raw_x[i] : null;
    const cy0 = hasCtrls ? cbf.ctrl_raw_y[i] : null;
    const fmt = v => v === null ? '<td>—</td>' : `<td class="ctrl-cell">${v.toFixed(4)}</td>`;
    const fmt0 = v => v === null ? '<td>—</td>' : `<td class="ctrl0-cell">${v.toFixed(4)}</td>`;
    const sdCell = i === 0
      ? `<td rowspan="${T}" style="text-align:center;color:var(--text-dim)">${sdVal !== null ? sdVal.toFixed(5) : '—'}</td>`
      : '';
    return `<tr>
      <td>${i + 1}</td>
      <td class="${dr >= 0 ? 'safe-cell' : 'danger-cell'}">${dr.toFixed(4)}</td>
      <td>${dt.toFixed(4)}</td>
      <td>${hw.toFixed(4)}</td>
      <td>${si.toFixed(5)}</td>
      <td>${gx.toFixed(4)}</td>
      <td>${gy.toFixed(4)}</td>
      <td class="bold-cell">${gn.toFixed(4)}</td>
      ${sdCell}${fmt(cx)}${fmt(cy)}${fmt0(cx0)}${fmt0(cy0)}
    </tr>`;
  });
  tbody.innerHTML = rows.join('');
}

/* ════════════════════════════════════════════════════════════════════
   STEP CONTROL  (scrubber, arrows, keyboard)
   ════════════════════════════════════════════════════════════════════ */

function setStep(step) {
  if (!cachedData) return;
  currentStep = Math.max(0, Math.min(cachedData.n_steps, step));
  scrubber.value = currentStep;
  stepLabel.textContent = currentStep;
  renderCurrentStep();
}

function bindScrubber() {
  scrubber.addEventListener('input', () => setStep(parseInt(scrubber.value, 10)));
}

function bindStepButtons() {
  btnPrev.addEventListener('click', () => setStep(currentStep - 1));
  btnNext.addEventListener('click', () => setStep(currentStep + 1));
}

function bindKeyboard() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Control') { isCtrlHeld = true; document.body.classList.add('ctrl-drag'); }
    if (!cachedData) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); setStep(currentStep - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); setStep(currentStep + 1); }
  });
  document.addEventListener('keyup', e => {
    if (e.key === 'Control') { isCtrlHeld = false; document.body.classList.remove('ctrl-drag'); }
  });
}

/* ════════════════════════════════════════════════════════════════════
   PARAMETER CONTROLS
   ════════════════════════════════════════════════════════════════════ */

function bindParamControls() {
  const bind = (id, key, valId, dec) => {
    const el = document.getElementById(id);
    const vel = valId ? document.getElementById(valId) : null;
    el.addEventListener('input', () => {
      params[key] = parseFloat(el.value);
      if (vel) vel.textContent = params[key].toFixed(dec);
      drawObstacleOnAll();
      if (currentBeforeTraj) debouncedRecomputeUpdate();
    });
  };
  bind('c-slider', 'c', 'c-val', 2);
  document.getElementById('k1-input').addEventListener('change', () => {
    params.k1 = Math.max(0.01, parseFloat(document.getElementById('k1-input').value) || 1.0);
    if (currentBeforeTraj) debouncedRecomputeUpdate();
  });
  bind('gd-slider', 'gamma_delta', 'gd-val', 2);
  document.getElementById('k2-input').addEventListener('change', () => {
    params.k2 = Math.max(0.01, parseFloat(document.getElementById('k2-input').value) || 1.0);
    if (currentBeforeTraj) debouncedRecomputeUpdate();
  });
  document.getElementById('alpha0-input').addEventListener('change', () => {
    params.alpha0 = Math.max(0.0, parseFloat(document.getElementById('alpha0-input').value) || 1.0);
    if (currentBeforeTraj) debouncedRecomputeUpdate();
  });
}

/* ════════════════════════════════════════════════════════════════════
   RUN BUTTON
   ════════════════════════════════════════════════════════════════════ */

function bindRunButton() { runBtn.addEventListener('click', runOptimisation); }

/* ════════════════════════════════════════════════════════════════════
   BATCH RUN  — 100 random-start/goal samples in one GPU batch
   ════════════════════════════════════════════════════════════════════ */

function bindBatchButton() {
  document.getElementById('batch-btn').addEventListener('click', runBatch);
}

async function runBatch() {
  const modelName = document.getElementById('model-select').value;
  if (!modelName) { alert('Select a model first.'); return; }
  const nSteps = parseInt(document.getElementById('n-steps-input').value, 10) || 20;
  params.k2 = Math.max(0.01, parseFloat(document.getElementById('k2-input').value) || 1.0);
  params.alpha0 = Math.max(0.0, parseFloat(document.getElementById('alpha0-input').value) || 1.0);
  params.use_softplus = document.getElementById('softplus-check').checked;

  const btn = document.getElementById('batch-btn');
  btn.disabled = true;
  btn.textContent = 'Running\u2026';
  try {
    const res = await fetch(`${API}/api/batch_run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName,
        n_steps: nSteps,
        n_samples: 100,
        c: params.c,
        k1: params.k1,
        k2: params.k2,
        gamma_delta: params.gamma_delta,
        alpha0: params.alpha0,
        use_softplus: params.use_softplus,
        obstacles: obstacles,
      }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }
    const data = await res.json();
    sessionStorage.setItem('batchData', JSON.stringify(data));
    window.open('/static/batch.html', '_blank');
  } catch (e) {
    alert(`Batch run failed: ${e.message}`);
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = '\u25BA\u25BA\u00A0Batch\u00A0100';
  }
}

async function runOptimisation() {
  const modelName = document.getElementById('model-select').value;
  if (!modelName) { alert('Select a model first.'); return; }
  const nSteps = parseInt(document.getElementById('n-steps-input').value, 10) || 20;
  params.k2 = Math.max(0.01, parseFloat(document.getElementById('k2-input').value) || 1.0);
  params.alpha0 = Math.max(0.0, parseFloat(document.getElementById('alpha0-input').value) || 1.0);
  params.use_softplus = document.getElementById('softplus-check').checked;

  setLoading(true);
  try {
    const res = await fetch(`${API}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName, n_steps: nSteps,
        c: params.c, k1: params.k1, k2: params.k2,
        gamma_delta: params.gamma_delta,
        alpha0: params.alpha0, use_softplus: params.use_softplus,
        obstacles: obstacles,
        start_x: -0.8, start_y: -0.8,
        goal_x: 0.8, goal_y: 0.8,
      }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }

    cachedData = await res.json();

    scrubber.max = cachedData.n_steps;
    scrubber.disabled = false;
    stepMax.textContent = cachedData.n_steps;
    btnPrev.disabled = false;
    btnNext.disabled = false;

    // Show sampling times
    document.getElementById('plain-time').textContent = `${cachedData.plain_time}s`;
    document.getElementById('safe-time').textContent = `${cachedData.safe_time}s`;

    // Show the final step immediately so divergence is visible
    setStep(cachedData.n_steps);

    drawMarkersOn(svgP);
    drawMarkersOn(svgB);
    drawMarkersOn(svgA);
  } catch (e) {
    alert(`Run failed: ${e.message}`);
    console.error(e);
  } finally {
    setLoading(false);
  }
}

/* ════════════════════════════════════════════════════════════════════
   RECOMPUTE UPDATE  (live re-evaluation on before-canvas drag / param change)
   Calls /api/recompute_ctrl with the current before-traj + cached eps_pred
   → updates the inspector table AND the after-canvas path
   ════════════════════════════════════════════════════════════════════ */

async function recomputeUpdate() {
  if (!currentBeforeTraj || !cachedData) return;

  const cbfCached = cachedData.cbf_step_data[currentStep];
  if (!cbfCached) return;

  // Step 0: no ODE step taken yet, no control — nothing to recompute
  if (currentStep === 0) return;

  try {
    const res = await fetch(`${API}/api/recompute_ctrl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        before_traj: currentBeforeTraj,
        eps_pred_x: cbfCached.eps_pred_x,
        eps_pred_y: cbfCached.eps_pred_y,
        sigma_delta: cbfCached.sigma_delta,
        sigma_dot: cbfCached.sigma_dot,
        noise_idx: cbfCached.noise_idx,
        step_delta: cbfCached.step_delta ?? 0,
        n_steps: cachedData.n_steps,
        c: params.c,
        k1: params.k1,
        k2: params.k2,
        gamma_delta: params.gamma_delta,
        alpha0: params.alpha0,
        use_softplus: params.use_softplus,
        obstacles: obstacles,
      }),
    });
    if (!res.ok) return;
    const result = await res.json();

    // Update after-control trajectory
    currentSafeTraj = result.after_traj;
    renderAfterPath(currentSafeTraj);
    renderWaypoints(svgA, currentSafeTraj, result);
    drawObstacleOn(svgA, false);
    drawMarkersOn(svgA);

    // Refresh before-canvas overlays with new metrics
    renderGradArrows(svgB, currentBeforeTraj, result);
    renderWaypoints(svgB, currentBeforeTraj, result);
    drawObstacleOnAll();

    updateInspector(result);
  } catch (e) { console.warn('Recompute error:', e); }
}

let _recomputeTimer = null;
function debouncedRecomputeUpdate() {
  clearTimeout(_recomputeTimer);
  _recomputeTimer = setTimeout(recomputeUpdate, 80);
}

/* ── Loading ──────────────────────────────────────────────────────── */
function setLoading(on) {
  overlay.classList.toggle('active', on);
  runBtn.disabled = on;
  scrubber.disabled = on;
  btnPrev.disabled = on;
  btnNext.disabled = on;
}

/* ════════════════════════════════════════════════════════════════════
   PANEL RESIZE  — generic drag-to-resize for any handle / target pair
   sign: +1 → drag right expands target (left panel)
         -1 → drag left  expands target (right panel / inspector)
   ════════════════════════════════════════════════════════════════════ */

function bindResizeHandle(handleId, targetId, sign, minPx, maxFrac) {
  const handle = document.getElementById(handleId);
  const target = document.getElementById(targetId);
  if (!handle || !target) return;

  let startX = 0, startW = 0;

  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    startX = e.clientX;
    startW = target.getBoundingClientRect().width;
    handle.classList.add('dragging');

    const onMove = ev => {
      const delta = (ev.clientX - startX) * sign;
      const newW = Math.max(minPx, Math.min(window.innerWidth * maxFrac, startW + delta));
      target.style.flex = `0 0 ${newW}px`;
    };

    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      // Rebuild canvases so SVGs re-fit the new panel sizes
      buildCanvases();
      if (cachedData) renderCurrentStep();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

function bindInspectorResize() {
  bindResizeHandle('plain-resize-handle', 'panel-plain', +1, 120, 0.35);
  bindResizeHandle('before-after-resize-handle', 'panel-before', +1, 120, 0.35);
  bindResizeHandle('inspector-resize-handle', 'inspector', -1, 220, 0.60);
}

/* ── Param bar vertical resize ────────────────────────────────────────── */
function bindParamResize() {
  const handle = document.getElementById('param-resize-handle');
  const target = document.getElementById('param-controls');
  if (!handle || !target) return;

  let startY = 0, startH = 0;

  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    startY = e.clientY;
    startH = target.getBoundingClientRect().height;
    handle.classList.add('dragging');

    const onMove = ev => {
      const delta = startY - ev.clientY;   // drag up → taller bar
      const newH = Math.max(36, Math.min(window.innerHeight * 0.7, startH + delta));
      target.style.height = `${newH}px`;
    };

    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      buildCanvases();
      if (cachedData) renderCurrentStep();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

/* ════════════════════════════════════════════════════════════════════
   PANEL DRAG-TO-REORDER
   Drag a panel title bar to swap that panel with another.
   Handles stay in their positional slots (between panels) via DOM swap.
   ════════════════════════════════════════════════════════════════════ */

function bindPanelDrag() {
  const PANEL_IDS = ['panel-plain', 'panel-before', 'panel-after'];
  let dragSrcId = null;

  PANEL_IDS.forEach(id => {
    const panel = document.getElementById(id);
    const title = panel.querySelector('.canvas-title');

    title.setAttribute('draggable', 'true');

    title.addEventListener('dragstart', e => {
      dragSrcId = id;
      e.dataTransfer.effectAllowed = 'move';
      setTimeout(() => panel.classList.add('panel-dragging'), 0);
    });

    panel.addEventListener('dragend', () => {
      panel.classList.remove('panel-dragging');
      PANEL_IDS.forEach(pid =>
        document.getElementById(pid).classList.remove('drop-left', 'drop-right'));
      dragSrcId = null;
    });

    panel.addEventListener('dragover', e => {
      if (!dragSrcId || dragSrcId === id) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      const rect = panel.getBoundingClientRect();
      const isLeft = e.clientX < rect.left + rect.width / 2;
      panel.classList.toggle('drop-left',  isLeft);
      panel.classList.toggle('drop-right', !isLeft);
    });

    panel.addEventListener('dragleave', e => {
      if (!panel.contains(e.relatedTarget)) {
        panel.classList.remove('drop-left', 'drop-right');
      }
    });

    panel.addEventListener('drop', e => {
      if (!dragSrcId || dragSrcId === id) return;
      e.preventDefault();
      panel.classList.remove('drop-left', 'drop-right');

      const main  = document.getElementById('main');
      const srcEl = document.getElementById(dragSrcId);
      const tgtEl = panel;

      // Swap the two panel nodes; resize handles stay between positions.
      const placeholder = document.createElement('div');
      main.insertBefore(placeholder, srcEl);
      main.insertBefore(srcEl, tgtEl);
      main.insertBefore(tgtEl, placeholder);
      main.removeChild(placeholder);

      buildCanvases();
      if (cachedData) renderCurrentStep();
    });
  });
}
