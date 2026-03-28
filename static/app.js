/* ══════════════════════════════════════════════════════════════════════
   SafeDPMSolver — Side-by-Side Visualiser
   Plain DPM (left)  |  Safe DPM / CBF (right, interactive)  |  Inspector
   ══════════════════════════════════════════════════════════════════════ */

const API = '';

/* ── Global state ─────────────────────────────────────────────────── */
let cachedData      = null;
let currentStep     = 0;
let currentSafeTraj = null;   // deep copy of safe traj for current step (editable)
let draggedIdx      = -1;
let isCtrlHeld      = false;

/* ── Per-SVG zoom transforms (persist across buildCanvases rebuilds) ── */
const zoomStates = {};   // keyed by SVG id → d3.ZoomTransform

let obstaclePos = { x: 0.0, y: 0.0 };
let params = { c: 1.0, k1: 1.0, k2: 1.0, r: 0.3, gamma_delta: 0.05, alpha0: 1.0 };

/* ── World bounds ─────────────────────────────────────────────────── */
const WORLD_MIN = -1.5;
const WORLD_MAX =  1.5;

/* ── Two SVG selections ───────────────────────────────────────────── */
const svgP = d3.select('#plain-svg');   // plain (left)
const svgS = d3.select('#safe-svg');    // safe  (right, interactive)

/* ── Scale functions  — shared size, updated in buildCanvases() ───── */
let SZ = 400;
const xS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([44, SZ - 16]);
const yS = d3.scaleLinear().domain([WORLD_MIN, WORLD_MAX]).range([SZ - 32, 16]);

/* ── DOM refs ─────────────────────────────────────────────────────── */
const overlay   = document.getElementById('loading-overlay');
const runBtn    = document.getElementById('run-btn');
const scrubber  = document.getElementById('step-scrubber');
const stepLabel = document.getElementById('step-label');
const stepMax   = document.getElementById('step-max');
const hxtValue   = document.getElementById('hxt-value');
const omegaValue = document.getElementById('omega-value');
const tbody      = document.getElementById('inspector-tbody');
const btnPrev   = document.getElementById('step-prev');
const btnNext   = document.getElementById('step-next');

/* ════════════════════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════════════════════ */

const DEFAULTS_KEY = 'safedpm_defaults';

window.addEventListener('DOMContentLoaded', () => {
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

    // Restore obstacle position
    if (snap.obstaclePos) Object.assign(obstaclePos, snap.obstaclePos);

    // Sync all slider/number DOM elements
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

    // Model selection — must wait until options are populated
    if (snap.model_name) {
      const trySelect = () => {
        const sel = document.getElementById('model-select');
        const opt = [...sel.options].find(o => o.value === snap.model_name);
        if (opt) { sel.value = snap.model_name; }
        else     { setTimeout(trySelect, 100); }   // retry until options load
      };
      trySelect();
    }
  } catch (e) { console.warn('Failed to load defaults:', e); }
}

function bindSaveDefaults() {
  document.getElementById('save-defaults-btn')
    .addEventListener('click', saveDefaults);
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
  // Compute square canvas size from whichever container is smaller
  const cP = document.getElementById('plain-container');
  const cS = document.getElementById('safe-container');
  const w  = Math.min(cP.clientWidth,  cS.clientWidth)  - 4;
  const h  = Math.min(cP.clientHeight, cS.clientHeight) - 4;
  SZ       = Math.max(Math.min(w, h), 200);

  xS.range([44, SZ - 16]);
  yS.range([SZ - 32, 16]);

  initSvg(svgP);
  initSvg(svgS);

  applyZoom(svgP);
  applyZoom(svgS);

  drawObstacleOnBoth();
  drawMarkersOn(svgP);
  drawMarkersOn(svgS);

  // Attach canvas-wide drag for waypoints (safe canvas only)
  attachSafeDrag();
}

/* ── Initialise one SVG: size, defs, grid, axes, empty layers ─────── */
function initSvg(svgSel) {
  svgSel.attr('width', SZ).attr('height', SZ);
  svgSel.selectAll('*').remove();

  // Arrow marker defs sit outside the zoom layer (IDs remain stable)
  const defs = svgSel.append('defs');
  [['safe', '#3ddc84'], ['danger', '#ff5370']].forEach(([cls, col]) => {
    defs.append('marker')
      .attr('id',         `arrow-${cls}-${svgSel.attr('id')}`)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('refX', 5).attr('refY', 3)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L0,6 L6,3 z').attr('fill', col);
  });

  // Everything zoomable lives inside this group
  const zl = svgSel.append('g').attr('class', 'zoom-layer');

  drawGrid(zl);
  drawAxes(zl);

  // Layer order: obstacle BEFORE waypoints so waypoints are on top and
  // receive pointer events even when inside the obstacle.
  ['prior','main-path','grads','obstacle','waypoints','markers'].forEach(name =>
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
        drawObstacleOnBoth();
        if (currentSafeTraj) debouncedMathUpdate();
      });
    g.selectAll('.obs-fill, .obs-ring').call(drag);
  }
}

function drawObstacleOnBoth() {
  drawObstacleOn(svgP, false);  // plain: non-draggable, just visual
  drawObstacleOn(svgS, true);   // safe: draggable
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
   CANVAS-WIDE DRAG ON SAFE SVG
   ════════════════════════════════════════════════════════════════════ */

function attachSafeDrag() {
  const drag = d3.drag()
    // Use the SVG element itself as the coordinate container so event.x/y
    // are in SVG pixel space (matching xS/yS), not the surrounding div.
    .container(svgS.node())
    .filter(function(event) {
      if (!currentSafeTraj) return false;
      // Ignore if click originated on the draggable obstacle
      let el = event.sourceEvent && event.sourceEvent.target;
      while (el) { if (el.id === 'safe-svg-obstacle') return false; el = el.parentElement; }
      return true;
    })
    .on('start', function(event) {
      const [wx, wy] = svgToWorld(event.x, event.y, svgS);
      draggedIdx = findNearest(wx, wy, currentSafeTraj);
    })
    .on('drag', function(event) {
      if (draggedIdx < 0) return;
      const [wx, wy] = svgToWorld(event.x, event.y, svgS);

      if (isCtrlHeld) {
        currentSafeTraj = gaussianDeform(currentSafeTraj, draggedIdx, wx, wy);
      } else {
        currentSafeTraj = currentSafeTraj.map((pt, i) => i === draggedIdx ? [wx, wy] : pt);
      }

      renderSafePath(currentSafeTraj);
      renderWaypoints(currentSafeTraj, null);
      debouncedMathUpdate();
    })
    .on('end', () => { draggedIdx = -1; });

  svgS.call(drag);
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

  const prior     = cachedData.prior;
  const plainTraj = cachedData.plain_history[currentStep];
  const safeTraj  = cachedData.safe_history[currentStep];
  const cbf       = cachedData.cbf_step_data[currentStep];

  // Deep copy so edits don't corrupt the cache
  currentSafeTraj = safeTraj.map(pt => [pt[0], pt[1]]);

  // ── Plain canvas ──────────────────────────────────────────────────
  renderPathOn(svgP, 'prior',     prior,     'prior-path');
  renderPathOn(svgP, 'main-path', plainTraj, 'plain-path');
  drawObstacleOn(svgP, false);
  drawMarkersOn(svgP);

  // ── Safe canvas ───────────────────────────────────────────────────
  renderPathOn(svgS, 'prior',     prior,          'prior-path');
  renderSafePath(currentSafeTraj);
  renderGradArrows(currentSafeTraj, cbf);
  renderWaypoints(currentSafeTraj, cbf);
  drawObstacleOn(svgS, true);
  drawMarkersOn(svgS);

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

function renderSafePath(traj) {
  renderPathOn(svgS, 'main-path', traj, 'safe-path');
}

function renderGradArrows(traj, cbf) {
  const id = svgS.attr('id');
  const g  = svgS.select(`#${id}-grads`);
  g.selectAll('*').remove();
  if (!cbf || !cbf.grad_x) return;

  const gx = cbf.grad_x, gy = cbf.grad_y;
  const mags   = gx.map((v, i) => Math.sqrt(v * v + gy[i] * gy[i]));
  const maxMag = Math.max(...mags, 1e-10);
  const MAX_PX = 28;
  const svgId  = svgS.attr('id');

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

function renderWaypoints(traj, cbf) {
  const id = svgS.attr('id');
  const g  = svgS.select(`#${id}-waypoints`);
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
    hxtValue.textContent  = '—'; hxtValue.className  = 'hxt-neutral';
    omegaValue.textContent = '—'; omegaValue.className = 'hxt-neutral';
    return;
  }

  // h(Xt)
  const hxt = cbf.h_Xt;
  hxtValue.textContent = hxt.toFixed(5);
  hxtValue.className   = hxt < 0 ? 'hxt-danger' : 'hxt-safe';

  // omega (scalar, may be null for /api/math or step-0)
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
   STEP CONTROL  (scrubber, arrows, keyboard)
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
      drawObstacleOnBoth();
      if (currentSafeTraj) debouncedMathUpdate();
    });
  };
  bind('c-slider',  'c',           'c-val',  2);
  bind('k1-slider', 'k1',          'k1-val', 2);
  bind('r-slider',  'r',           'r-val',  2);
  bind('gd-slider', 'gamma_delta', 'gd-val', 2);
  document.getElementById('k2-input').addEventListener('change', () => {
    params.k2 = Math.max(0.01, parseFloat(document.getElementById('k2-input').value) || 1.0);
    if (currentSafeTraj) debouncedMathUpdate();
  });
  document.getElementById('alpha0-input').addEventListener('change', () => {
    params.alpha0 = Math.max(0.0, parseFloat(document.getElementById('alpha0-input').value) || 1.0);
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

    // Show sampling times
    document.getElementById('plain-time').textContent = `${cachedData.plain_time}s`;
    document.getElementById('safe-time').textContent  = `${cachedData.safe_time}s`;

    // Show the final step immediately so divergence is visible
    setStep(cachedData.n_steps);

    drawMarkersOn(svgP);
    drawMarkersOn(svgS);
  } catch (e) {
    alert(`Run failed: ${e.message}`);
    console.error(e);
  } finally {
    setLoading(false);
  }
}

/* ════════════════════════════════════════════════════════════════════
   MATH UPDATE  (live re-evaluation on drag / param change)
   ════════════════════════════════════════════════════════════════════ */

async function mathUpdate() {
  if (!currentSafeTraj) return;
  try {
    const res = await fetch(`${API}/api/math`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        traj: currentSafeTraj,
        c: params.c, k1: params.k1, k2: params.k2,
        r: params.r, gamma_delta: params.gamma_delta,
        use_softplus: params.use_softplus,
        obs_x: obstaclePos.x, obs_y: obstaclePos.y,
      }),
    });
    if (!res.ok) return;
    const fresh = await res.json();

    // /api/math has no score-network access so it cannot recompute omega or
    // ctrl (those need eps_pred).  Preserve the values from the original run.
    const prev = cachedData?.cbf_step_data[currentStep];
    const cbf = {
      ...fresh,
      omega:       prev?.omega       ?? null,
      ctrl_x:      prev?.ctrl_x      ?? null,
      ctrl_y:      prev?.ctrl_y      ?? null,
      ctrl_raw_x:  prev?.ctrl_raw_x  ?? null,
      ctrl_raw_y:  prev?.ctrl_raw_y  ?? null,
      sigma_delta: prev?.sigma_delta ?? null,
    };

    if (cachedData) {
      cachedData.cbf_step_data[currentStep] = cbf;
      cachedData.safe_history[currentStep]  = currentSafeTraj.map(p => [p[0], p[1]]);
    }

    renderGradArrows(currentSafeTraj, cbf);
    renderWaypoints(currentSafeTraj, cbf);
    drawObstacleOnBoth();
    updateInspector(cbf);
  } catch (e) { console.warn('Math update error:', e); }
}

let _mathTimer = null;
function debouncedMathUpdate() {
  clearTimeout(_mathTimer);
  _mathTimer = setTimeout(mathUpdate, 80);
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
      const newW  = Math.max(minPx, Math.min(window.innerWidth * maxFrac, startW + delta));
      target.style.flex = `0 0 ${newW}px`;
    };

    const onUp = () => {
      handle.classList.remove('dragging');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
      // Rebuild canvases so SVGs re-fit the new panel sizes
      buildCanvases();
      if (cachedData) renderCurrentStep();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });
}

function bindInspectorResize() {
  // Plain panel: drag right → expand (+1), bounded 120px – 55% of window
  bindResizeHandle('plain-resize-handle',    'panel-plain', +1, 120, 0.55);
  // Inspector:   drag left  → expand (-1), bounded 220px – 60% of window
  bindResizeHandle('inspector-resize-handle','inspector',   -1, 220, 0.60);
}
