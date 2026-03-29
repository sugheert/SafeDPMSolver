/* ══════════════════════════════════════════════════════════════════════
   SafeDPMSolver — UMaze Three-Panel Visualiser
   Plain DPM (left)  |  DPM-CBF Before Control (middle, interactive)
   DPM-CBF After Control (right, passive)  |  Inspector
   ══════════════════════════════════════════════════════════════════════ */

const API = '';

/* ── Global state ─────────────────────────────────────────────────── */
let cachedData        = null;
let currentStep       = 0;
let currentBeforeTraj = null;  // editable before-control traj; reset on step change
let currentSafeTraj   = null;  // after-control traj; updated by recompute
let draggedIdx        = -1;
let isCtrlHeld        = false;
let mazeData          = null;  // populated by fetchMaze() on page load

/* ── Per-SVG zoom transforms (persist across buildCanvases rebuilds) ── */
const zoomStates = {};   // keyed by SVG id → d3.ZoomTransform

let obstaclePos = { x: 0.0, y: 0.0 };
let params = { c: 1.0, k1: 1.0, k2: 1.0, r: 0.3, gamma_delta: 0.05, alpha0: 1.0 };

/* ── World bounds — extended to show U-Maze outer walls ──────────── */
const WORLD_MIN = -1.9;
const WORLD_MAX =  1.9;

/* ── Three SVG selections ─────────────────────────────────────────── */
const svgP = d3.select('#plain-svg');    // plain (left)
const svgB = d3.select('#before-svg');   // before-control (middle, interactive)
const svgA = d3.select('#after-svg');    // after-control  (right, passive)

/* ── Scale functions  — shared size, updated in buildCanvases() ───── */
let SZ = 400;
const xS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([44, SZ - 16]);
const yS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([SZ - 32, 16]);

/* ── DOM refs ─────────────────────────────────────────────────────── */
const overlay         = document.getElementById('loading-overlay');
const runBtn          = document.getElementById('run-btn');
const scrubber        = document.getElementById('step-scrubber');
const stepLabel       = document.getElementById('step-label');
const stepMax         = document.getElementById('step-max');
const hxtValue        = document.getElementById('hxt-value');
const minHwiValue     = document.getElementById('min-hwi-value');
const sigmaDeltaValue = document.getElementById('sigma-delta-value');
const sigmaDotValue   = document.getElementById('sigma-dot-value');
const omegaValue      = document.getElementById('omega-value');
const tbody           = document.getElementById('inspector-tbody');
const btnPrev         = document.getElementById('step-prev');
const btnNext         = document.getElementById('step-next');

/* ════════════════════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════════════════════ */

const DEFAULTS_KEY = 'safedpm_umaze_defaults';

window.addEventListener('DOMContentLoaded', async () => {
  await fetchMaze();     // load maze geometry before building canvases
  loadDefaults();        // restore saved params before binding controls
  fetchModels();
  buildCanvases();
  bindParamControls();
  bindRunButton();
  bindScrubber();
  bindKeyboard();
  bindStepButtons();
  bindInspectorResize();
  bindSaveDefaults();
  window.addEventListener('resize', () => { buildCanvases(); if (cachedData) renderCurrentStep(); });
});

/* ════════════════════════════════════════════════════════════════════
   MAZE  — fetch wall geometry from /api/maze
   ════════════════════════════════════════════════════════════════════ */

async function fetchMaze() {
  try {
    const res = await fetch(`${API}/api/maze`);
    if (res.ok) mazeData = await res.json();
  } catch (e) { console.warn('Maze fetch failed:', e); }
}

function drawMazeWalls(svgSel) {
  if (!mazeData) return;
  const id = svgSel.attr('id');
  const g  = svgSel.select(`#${id}-maze`);
  g.selectAll('*').remove();
  mazeData.walls.forEach(w => {
    const px = xS(w.cx), py = yS(w.cy);
    const pw = Math.abs(xS(w.cx + w.hw) - xS(w.cx - w.hw));
    const ph = Math.abs(yS(w.cy + w.hh) - yS(w.cy - w.hh));
    g.append('rect')
      .attr('class', 'maze-wall')
      .attr('x', px - pw / 2)
      .attr('y', py - ph / 2)
      .attr('width', pw)
      .attr('height', ph);
  });
}

/* ════════════════════════════════════════════════════════════════════
   DEFAULTS  (localStorage persistence)
   ════════════════════════════════════════════════════════════════════ */

function saveDefaults() {
  const snapshot = {
    params:     { ...params },
    obstaclePos: { ...obstaclePos },
    n_steps:    document.getElementById('n-steps-input').value,
    model_name: document.getElementById('model-select').value,
  };
  localStorage.setItem(DEFAULTS_KEY, JSON.stringify(snapshot));

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
    if (snap.params) Object.assign(params, snap.params);
    if (snap.obstaclePos) Object.assign(obstaclePos, snap.obstaclePos);

    const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    setEl('c-slider',     params.c);
    setEl('c-val',        params.c.toFixed(2));
    setEl('k1-slider',    params.k1);
    setEl('k1-val',       params.k1.toFixed(2));
    setEl('k2-input',     params.k2);
    setEl('r-slider',     params.r);
    setEl('r-val',        params.r.toFixed(2));
    setEl('gd-slider',    params.gamma_delta);
    setEl('gd-val',       params.gamma_delta.toFixed(2));
    setEl('alpha0-input', params.alpha0);
    if (snap.n_steps) setEl('n-steps-input', snap.n_steps);

    if (snap.model_name) {
      const trySelect = () => {
        const sel = document.getElementById('model-select');
        const opt = [...sel.options].find(o => o.value === snap.model_name);
        if (opt) { sel.value = snap.model_name; }
        else     { setTimeout(trySelect, 100); }
      };
      trySelect();
    }
  } catch (e) { console.warn('Failed to load defaults:', e); }
}

function bindSaveDefaults() {
  document.getElementById('save-defaults-btn').addEventListener('click', saveDefaults);
}

/* ── Model list ───────────────────────────────────────────────────── */
async function fetchModels() {
  try {
    const res  = await fetch(`${API}/api/models`);
    const data = await res.json();
    const sel  = document.getElementById('model-select');
    sel.innerHTML = '';
    (data.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === 've_unet_umaze_diffuser.pt') opt.selected = true;
      sel.appendChild(opt);
    });
    if (!data.models || !data.models.length)
      sel.innerHTML = '<option value="">No checkpoints found</option>';
  } catch (e) { console.warn('Models fetch failed:', e); }
}

/* ════════════════════════════════════════════════════════════════════
   CANVAS CONSTRUCTION
   ════════════════════════════════════════════════════════════════════ */

function buildCanvases() {
  const cP  = document.getElementById('plain-container');
  const cBe = document.getElementById('before-container');
  const cAf = document.getElementById('after-container');
  const w   = Math.min(cP.clientWidth,  cBe.clientWidth,  cAf.clientWidth)  - 4;
  const h   = Math.min(cP.clientHeight, cBe.clientHeight, cAf.clientHeight) - 4;
  SZ        = Math.max(Math.min(w, h), 200);

  xS.range([44, SZ - 16]);
  yS.range([SZ - 32, 16]);

  initSvg(svgP);
  initSvg(svgB);
  initSvg(svgA);

  applyZoom(svgP);
  applyZoom(svgB);
  applyZoom(svgA);

  drawObstacleOnAll();
  drawMarkersOn(svgP);
  drawMarkersOn(svgB);
  drawMarkersOn(svgA);

  attachBeforeDrag();
}

/* ── Initialise one SVG ───────────────────────────────────────────── */
function initSvg(svgSel) {
  svgSel.attr('width', SZ).attr('height', SZ);
  svgSel.selectAll('*').remove();

  const defs = svgSel.append('defs');
  [['safe', '#3ddc84'], ['danger', '#ff5370']].forEach(([cls, col]) => {
    defs.append('marker')
      .attr('id',         `arrow-${cls}-${svgSel.attr('id')}`)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', col);
  });

  const zl = svgSel.append('g').attr('class', 'zoom-layer');

  drawGrid(zl);
  drawAxes(zl);

  // maze layer comes first so waypoints sit on top
  ['maze', 'prior', 'main-path', 'grads', 'obstacle', 'waypoints', 'markers'].forEach(name =>
    zl.append('g').attr('id', `${svgSel.attr('id')}-${name}`)
  );

  drawMazeWalls(svgSel);
}

function applyZoom(svgSel) {
  const id = svgSel.attr('id');
  const behavior = d3.zoom()
    .scaleExtent([0.5, 20])
    .filter(e => e.type === 'wheel')
    .on('zoom', event => {
      svgSel.select('.zoom-layer').attr('transform', event.transform);
      zoomStates[id] = event.transform;
    });
  svgSel.call(behavior);
  svgSel.on('dblclick.zoom', null);
  svgSel.on('dblclick', () => {
    behavior.transform(svgSel, d3.zoomIdentity);
    zoomStates[id] = d3.zoomIdentity;
  });
  if (zoomStates[id]) behavior.transform(svgSel, zoomStates[id]);
}

function svgToWorld(px, py, svgSel) {
  const zt = zoomStates[svgSel.attr('id')] || d3.zoomIdentity;
  const [ox, oy] = zt.invert([px, py]);
  return [xS.invert(ox), yS.invert(oy)];
}

function drawGrid(containerSel) {
  const g = containerSel.append('g');
  for (let v = WORLD_MIN; v <= WORLD_MAX + 0.001; v += 0.5) {
    g.append('line').attr('class','grid-line')
      .attr('x1', xS(v)).attr('y1', yS(WORLD_MIN))
      .attr('x2', xS(v)).attr('y2', yS(WORLD_MAX));
    g.append('line').attr('class','grid-line')
      .attr('x1', xS(WORLD_MIN)).attr('y1', yS(v))
      .attr('x2', xS(WORLD_MAX)).attr('y2', yS(v));
  }
}

function drawAxes(containerSel) {
  const g  = containerSel.append('g');
  const ox = xS(0), oy = yS(0);
  g.append('line').attr('class','axis-line')
    .attr('x1', xS(WORLD_MIN)).attr('y1', oy)
    .attr('x2', xS(WORLD_MAX)).attr('y2', oy);
  g.append('line').attr('class','axis-line')
    .attr('x1', ox).attr('y1', yS(WORLD_MIN))
    .attr('x2', ox).attr('y2', yS(WORLD_MAX));
  [-1, -0.5, 0.5, 1].forEach(v => {
    g.append('text').attr('class','axis-label')
      .attr('x', xS(v)).attr('y', oy + 13).attr('text-anchor','middle').text(v);
    g.append('text').attr('class','axis-label')
      .attr('x', ox - 5).attr('y', yS(v) + 4).attr('text-anchor','end').text(v);
  });
}

/* ── Obstacle ─────────────────────────────────────────────────────── */

function drawObstacleOn(svgSel, draggable) {
  const id  = svgSel.attr('id');
  const g   = svgSel.select(`#${id}-obstacle`);
  g.selectAll('*').remove();

  const cx  = xS(obstaclePos.x), cy = yS(obstaclePos.y);
  const rPx = xS(params.r)                      - xS(0);
  const mPx = xS(params.r + params.gamma_delta) - xS(0);

  g.append('circle').attr('class', `obs-fill${draggable ? ' obs-drag' : ''}`)
    .attr('cx', cx).attr('cy', cy).attr('r', rPx);
  g.append('circle').attr('class', `obs-ring${draggable ? ' obs-drag' : ''}`)
    .attr('cx', cx).attr('cy', cy).attr('r', rPx);
  g.append('circle').attr('class', 'obs-margin')
    .attr('cx', cx).attr('cy', cy).attr('r', mPx);

  if (draggable) {
    const drag = d3.drag()
      .on('drag', function(event) {
        event.sourceEvent.stopPropagation();
        obstaclePos.x = xS.invert(event.x);
        obstaclePos.y = yS.invert(event.y);
        drawObstacleOnAll();
        if (currentBeforeTraj) debouncedRecomputeUpdate();
      });
    g.selectAll('.obs-fill, .obs-ring').call(drag);
  }
}

function drawObstacleOnAll() {
  drawObstacleOn(svgP, false);
  drawObstacleOn(svgB, true);    // draggable
  drawObstacleOn(svgA, false);
}

/* ── Start / Goal markers ─────────────────────────────────────────── */
function drawMarkersOn(svgSel) {
  const id = svgSel.attr('id');
  const g  = svgSel.select(`#${id}-markers`);
  g.selectAll('*').remove();
  if (!cachedData) return;
  const [sx, sy] = cachedData.start;
  const [gx, gy] = cachedData.goal;
  g.append('rect').attr('class','start-mk')
    .attr('x', xS(sx)-6).attr('y', yS(sy)-6).attr('width',12).attr('height',12).attr('rx',2);
  g.append('text')
    .attr('x', xS(gx)).attr('y', yS(gy)+6).attr('text-anchor','middle')
    .attr('font-size',18).attr('fill','#ffc107').text('★');
}

/* ════════════════════════════════════════════════════════════════════
   CANVAS-WIDE DRAG ON BEFORE SVG
   ════════════════════════════════════════════════════════════════════ */

function attachBeforeDrag() {
  const drag = d3.drag()
    .container(svgB.node())
    .filter(function(event) {
      if (!currentBeforeTraj) return false;
      let el = event.sourceEvent && event.sourceEvent.target;
      while (el) { if (el.id === 'before-svg-obstacle') return false; el = el.parentElement; }
      return true;
    })
    .on('start', function(event) {
      const [wx, wy] = svgToWorld(event.x, event.y, svgB);
      draggedIdx = findNearest(wx, wy, currentBeforeTraj);
    })
    .on('drag', function(event) {
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

  const prior      = cachedData.prior;
  const plainTraj  = cachedData.plain_history[currentStep];
  const beforeTraj = cachedData.before_history[currentStep];
  const afterTraj  = cachedData.safe_history[currentStep];
  const cbf        = cachedData.cbf_step_data[currentStep];

  // Reset from cache (discards any prior drags)
  currentBeforeTraj = beforeTraj.map(pt => [pt[0], pt[1]]);
  currentSafeTraj   = afterTraj.map(pt => [pt[0], pt[1]]);

  // ── Plain canvas ──────────────────────────────────────────────────
  renderPathOn(svgP, 'prior',     prior,     'prior-path');
  renderPathOn(svgP, 'main-path', plainTraj, 'plain-path');
  drawMazeWalls(svgP);
  drawObstacleOn(svgP, false);
  drawMarkersOn(svgP);

  // ── Before-control canvas (interactive) ───────────────────────────
  renderPathOn(svgB, 'prior',     prior,             'prior-path');
  renderBeforePath(currentBeforeTraj);
  renderGradArrows(svgB, currentBeforeTraj, cbf);
  renderWaypoints(svgB, currentBeforeTraj, cbf);
  drawMazeWalls(svgB);
  drawObstacleOn(svgB, true);
  drawMarkersOn(svgB);

  // ── After-control canvas (passive) ────────────────────────────────
  renderPathOn(svgA, 'prior',     prior,           'prior-path');
  renderAfterPath(currentSafeTraj);
  renderWaypoints(svgA, currentSafeTraj, cbf);
  drawMazeWalls(svgA);
  drawObstacleOn(svgA, false);
  drawMarkersOn(svgA);

  updateInspector(cbf);
}

const lineGen = d3.line().x(d => xS(d[0])).y(d => yS(d[1]));

function renderPathOn(svgSel, layerName, traj, cssClass) {
  const id = svgSel.attr('id');
  const g  = svgSel.select(`#${id}-${layerName}`);
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
  const g  = svgSel.select(`#${id}-grads`);
  g.selectAll('*').remove();
  if (!cbf || !cbf.grad_x) return;

  const gx = cbf.grad_x, gy = cbf.grad_y;
  const mags   = gx.map((v, i) => Math.sqrt(v * v + gy[i] * gy[i]));
  const maxMag = Math.max(...mags, 1e-10);
  const MAX_PX = 28;
  const svgId  = svgSel.attr('id');

  traj.forEach(([wx, wy], i) => {
    const mag = mags[i];
    if (mag < 1e-8) return;
    const len  = (mag / maxMag) * MAX_PX;
    const nx   = gx[i] / mag, ny = gy[i] / mag;
    const x1   = xS(wx), y1 = yS(wy);
    const x2   = x1 + nx * len, y2 = y1 - ny * len;
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
  const g  = svgSel.select(`#${id}-waypoints`);
  g.selectAll('*').remove();
  if (!traj) return;

  g.selectAll('circle')
    .data(traj.map((pt, i) => ({ pt, i })))
    .join('circle')
    .attr('cx', d => xS(d.pt[0]))
    .attr('cy', d => yS(d.pt[1]))
    .attr('r', 5)
    .attr('class', d => (!cbf || cbf.d_raw[d.i] >= 0) ? 'wp-safe' : 'wp-danger')
    .on('mouseenter', function(event, d) {
      const i   = d.i;
      const [wx, wy] = d.pt;
      const safe = !cbf || cbf.d_raw[i] >= 0;
      const hw   = cbf ? cbf.h_wi[i]   : null;
      const gx   = cbf ? cbf.grad_x[i] : null;
      const gy   = cbf ? cbf.grad_y[i] : null;
      const gn   = (gx !== null) ? Math.sqrt(gx*gx + gy*gy) : null;
      const cls  = safe ? 'tt-safe' : 'tt-danger';
      const fmt  = v => v !== null ? v.toFixed(5) : '—';

      tooltip.innerHTML = `
        <div class="tt-title">Waypoint ${i + 1}</div>
        <div><span class="tt-dim">x =</span> ${wx.toFixed(5)}</div>
        <div><span class="tt-dim">y =</span> ${wy.toFixed(5)}</div>
        <div><span class="tt-dim">h(w<sub>${i+1}</sub>) =</span> <span class="${cls}">${fmt(hw)}</span></div>
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
  const tw  = tooltip.offsetWidth, th = tooltip.offsetHeight;
  let lx = event.clientX + pad;
  let ly = event.clientY + pad;
  if (lx + tw > window.innerWidth)  lx = event.clientX - tw - pad;
  if (ly + th > window.innerHeight) ly = event.clientY - th - pad;
  tooltip.style.left = lx + 'px';
  tooltip.style.top  = ly + 'px';
}

/* ════════════════════════════════════════════════════════════════════
   STEP INSPECTOR
   ════════════════════════════════════════════════════════════════════ */

function updateInspector(cbf) {
  if (!cbf) {
    tbody.innerHTML = '';
    hxtValue.textContent        = '—'; hxtValue.className        = 'hxt-neutral';
    minHwiValue.textContent     = '—'; minHwiValue.className     = 'hxt-neutral';
    sigmaDeltaValue.textContent = '—'; sigmaDeltaValue.className = 'hxt-neutral';
    sigmaDotValue.textContent   = '—'; sigmaDotValue.className   = 'hxt-neutral';
    omegaValue.textContent      = '—'; omegaValue.className      = 'hxt-neutral';
    return;
  }

  // h(Xt)
  const hxt = cbf.h_Xt;
  hxtValue.textContent = hxt.toFixed(5);
  hxtValue.className   = hxt < 0 ? 'hxt-danger' : 'hxt-safe';

  // min h(w_i)
  const minHwi = Math.min(...cbf.h_wi);
  minHwiValue.textContent = minHwi.toFixed(5);
  minHwiValue.className   = minHwi < 0 ? 'hxt-danger' : 'hxt-safe';

  // sigma_delta and sigma_dot
  const sd   = cbf.sigma_delta ?? null;
  const sdot = cbf.sigma_dot   ?? null;
  sigmaDeltaValue.textContent = sd   !== null ? sd.toFixed(5)   : '—';
  sigmaDotValue.textContent   = sdot !== null ? sdot.toFixed(5) : '—';
  sigmaDeltaValue.className   = 'hxt-neutral';
  sigmaDotValue.className     = 'hxt-neutral';

  // omega
  const omega = cbf.omega ?? null;
  if (omega === null) {
    omegaValue.textContent = '—';
    omegaValue.className   = 'hxt-neutral';
  } else {
    omegaValue.textContent = omega.toFixed(5);
    omegaValue.className   = omega < 0 ? 'omega-neg' : (omega === 0 ? 'omega-zero' : 'omega-pos');
  }

  const hasCtrls = cbf.ctrl_x && cbf.ctrl_raw_x;

  const T = cbf.d_raw.length;
  const sdVal = cbf.sigma_delta ?? null;
  const rows = cbf.d_raw.map((dr, i) => {
    const dt  = cbf.d_tilde[i], hw = cbf.h_wi[i], si = cbf.sigma_i[i];
    const gx  = cbf.grad_x[i],  gy = cbf.grad_y[i];
    const gn  = Math.sqrt(gx * gx + gy * gy);
    const cx  = hasCtrls ? cbf.ctrl_x[i]     : null;
    const cy  = hasCtrls ? cbf.ctrl_y[i]     : null;
    const cx0 = hasCtrls ? cbf.ctrl_raw_x[i] : null;
    const cy0 = hasCtrls ? cbf.ctrl_raw_y[i] : null;
    const fmt  = v => v === null ? '<td>—</td>' : `<td class="ctrl-cell">${v.toFixed(4)}</td>`;
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
   STEP CONTROL
   ════════════════════════════════════════════════════════════════════ */

function setStep(step) {
  if (!cachedData) return;
  currentStep           = Math.max(0, Math.min(cachedData.n_steps, step));
  scrubber.value        = currentStep;
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
    if (e.key === 'ArrowLeft')  { e.preventDefault(); setStep(currentStep - 1); }
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
    const el  = document.getElementById(id);
    const vel = valId ? document.getElementById(valId) : null;
    el.addEventListener('input', () => {
      params[key] = parseFloat(el.value);
      if (vel) vel.textContent = params[key].toFixed(dec);
      drawObstacleOnAll();
      if (currentBeforeTraj) debouncedRecomputeUpdate();
    });
  };
  bind('c-slider',  'c',           'c-val',  2);
  bind('k1-slider', 'k1',          'k1-val', 2);
  bind('r-slider',  'r',           'r-val',  2);
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

async function runOptimisation() {
  const modelName = document.getElementById('model-select').value;
  if (!modelName) { alert('Select a model first.'); return; }
  const nSteps  = parseInt(document.getElementById('n-steps-input').value,  10) || 20;
  params.k2          = Math.max(0.01, parseFloat(document.getElementById('k2-input').value)    || 1.0);
  params.alpha0      = Math.max(0.0,  parseFloat(document.getElementById('alpha0-input').value) || 1.0);
  params.use_softplus = document.getElementById('softplus-check').checked;

  setLoading(true);
  try {
    const res = await fetch(`${API}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName, n_steps: nSteps,
        c: params.c, k1: params.k1, k2: params.k2,
        r: params.r, gamma_delta: params.gamma_delta,
        alpha0: params.alpha0, use_softplus: params.use_softplus,
        obs_x: obstaclePos.x, obs_y: obstaclePos.y,
        start_x: -0.8, start_y: -0.8,
        goal_x:   0.8, goal_y:   0.8,
      }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }

    cachedData = await res.json();

    scrubber.max      = cachedData.n_steps;
    scrubber.disabled = false;
    stepMax.textContent = cachedData.n_steps;
    btnPrev.disabled  = false;
    btnNext.disabled  = false;

    document.getElementById('plain-time').textContent = `${cachedData.plain_time}s`;
    document.getElementById('safe-time').textContent  = `${cachedData.safe_time}s`;

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
   RECOMPUTE UPDATE
   ════════════════════════════════════════════════════════════════════ */

async function recomputeUpdate() {
  if (!currentBeforeTraj || !cachedData) return;

  const cbfCached = cachedData.cbf_step_data[currentStep];
  if (!cbfCached) return;

  if (currentStep === 0) return;

  try {
    const res = await fetch(`${API}/api/recompute_ctrl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        before_traj:  currentBeforeTraj,
        eps_pred_x:   cbfCached.eps_pred_x,
        eps_pred_y:   cbfCached.eps_pred_y,
        sigma_delta:  cbfCached.sigma_delta,
        sigma_dot:    cbfCached.sigma_dot,
        noise_idx:    cbfCached.noise_idx,
        n_steps:      cachedData.n_steps,
        c:            params.c,
        k1:           params.k1,
        k2:           params.k2,
        r:            params.r,
        gamma_delta:  params.gamma_delta,
        alpha0:       params.alpha0,
        use_softplus: params.use_softplus,
        obs_x:        obstaclePos.x,
        obs_y:        obstaclePos.y,
      }),
    });
    if (!res.ok) return;
    const result = await res.json();

    currentSafeTraj = result.after_traj;
    renderAfterPath(currentSafeTraj);
    renderWaypoints(svgA, currentSafeTraj, result);
    drawMazeWalls(svgA);
    drawObstacleOn(svgA, false);
    drawMarkersOn(svgA);

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
  runBtn.disabled   = on;
  scrubber.disabled = on;
  btnPrev.disabled  = on;
  btnNext.disabled  = on;
}

/* ════════════════════════════════════════════════════════════════════
   PANEL RESIZE
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
      const newW  = Math.max(minPx, Math.min(window.innerWidth * maxFrac, startW + delta));
      target.style.flex = `0 0 ${newW}px`;
    };

    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
      buildCanvases();
      if (cachedData) renderCurrentStep();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });
}

function bindInspectorResize() {
  bindResizeHandle('plain-resize-handle',        'panel-plain',  +1, 120, 0.35);
  bindResizeHandle('before-after-resize-handle', 'panel-before', +1, 120, 0.35);
  bindResizeHandle('inspector-resize-handle',    'inspector',    -1, 220, 0.60);
}
