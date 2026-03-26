# CBF Trajectory Simulation — Interactive Harness

## Purpose

This harness describes the requirements for an interactive browser-based simulation of a Control Barrier Function (CBF) applied to trajectory optimisation. It is intended to be read by an agent before building or modifying the simulation. No implementation is prescribed — the harness captures what the user sees and does, not how the maths or code should work.

---

## Overview

The simulation has two modes that are always visible simultaneously:

- **Trajectory canvas** — a 2D top-down view of a robot path and an obstacle.
- **Step inspector** — per-waypoint numerical readout of all computed values and gradients.

A sampling workflow sits on top: the user picks a model, sets a step count, runs the optimisation, and scrubs through the resulting steps. The canvas and inspector update for whichever step is selected.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  TOOLBAR  [Model dropdown]  [Steps input]  [Run button]  [Scrubber]  │
├───────────────────────────────┬──────────────────────────────────────┤
│                               │                                      │
│      TRAJECTORY CANVAS        │        STEP INSPECTOR TABLE          │
│        (square, ~55%)         │          (scrollable, ~45%)          │
│                               │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│  PARAMETER CONTROLS  [c]  [k₁ — disabled]  [k₂]  [r]               │
└──────────────────────────────────────────────────────────────────────┘
```

All four sections are visible at once. No tabs, no collapsing panels.

---

## Trajectory Canvas

### Static elements
- Grid with labelled axes (world coordinates).
- One circular obstacle — draggable anywhere on the canvas.
- Obstacle radius ring visualising `r`.

### Trajectory rendering
- **Initial trajectory** (before any sampling) — rendered as a light grey dashed polyline. Persists throughout the entire session as a reference.
- **Current trajectory** (at the selected step) — rendered as a solid coloured polyline on top.
- 64 waypoints rendered as small filled dots:
  - Red dot — waypoint violates the safety constraint (inside or on the obstacle).
  - Green dot — waypoint is safe.
- Gradient arrows at each waypoint:
  - Direction: the gradient vector for that waypoint.
  - Length: normalised relative to the largest gradient magnitude across all 64 waypoints at the current step.
  - Same red/green colouring as the dot.
- A blue highlight ring appears on the waypoint currently being Ctrl-dragged.

### Interaction
- **Plain drag** — moves the nearest waypoint individually. Snaps to the closest of the 64 points.
- **Ctrl + drag** — Gaussian spline deformation. Dragging one point bends a smooth neighbourhood of waypoints around it. Influence falls off with distance along the trajectory index.
- **Drag obstacle** — repositions the obstacle circle. All values recompute live.
- Cursor changes to a crosshair when Ctrl is held; returns to grab cursor on release.

### Overlay during sampling
- While the optimisation is running, the canvas shows a subtle loading indicator (e.g. a spinner or pulsing overlay). Interaction is disabled.
- Once complete, the canvas shows the trajectory at the selected step with the initial trajectory still ghosted underneath.

---

## Step Inspector

A scrollable table with a sticky header. One row per waypoint (64 rows total).

### Header row (sticky)
`i` | `dᵢ` | `d̃ᵢ` | `h(wᵢ)` | `σᵢ` | `∂x` | `∂y` | `|∇|`

### Per-row content
| Column | Description |
|--------|-------------|
| `i` | Waypoint index, 1–64 |
| `dᵢ` | Raw signed distance value for this waypoint. Red text if negative (violated), green if non-negative (safe). |
| `d̃ᵢ` | Transformed distance. Safe waypoints saturate near a fixed upper bound; violated ones grow negative. |
| `h(wᵢ)` | Waypoint-level CBF value (equals `d̃ᵢ` when there is one obstacle). |
| `σᵢ` | Softmax weight of this waypoint in the trajectory-level aggregation. All 64 values sum to 1. |
| `∂x` | x-component of the gradient contribution from this waypoint. |
| `∂y` | y-component of the gradient contribution from this waypoint. |
| `|∇|` | Euclidean norm of `(∂x, ∂y)`. Bold. |

- Violated rows have a faint red background tint.
- All numerical values displayed to 4 decimal places.
- Table scrolls independently of the rest of the UI.

### Summary scalar
Above the table: a large prominent readout of `h(X_t)`, the scalar trajectory-level CBF value.
- Red if negative (unsafe trajectory).
- Green if non-negative (safe trajectory).
- Displayed to 5 decimal places.

---

## Toolbar (Sampling Controls)

### Model dropdown
- A `<select>` listing the available sampling/optimisation models.
- Minimum required options: at least two distinct models the user can compare.
- Selecting a different model does not automatically re-run; the user must press Run.

### Steps input
- A numeric text input accepting any positive integer.
- Default: 20.
- No hard upper limit enforced in the UI (the backend may have one).

### Run button
- Labelled **Run**.
- Triggers the optimisation/sampling loop for the selected model and step count.
- Disabled while a run is in progress.
- On completion: populates the scrubber and jumps to the final step.

### Step scrubber
- A horizontal slider (or equivalent) ranging from `0` (initial trajectory, before any updates) to `N` (the final step of the completed run).
- Step `0` always shows the original initial trajectory and the initial values in the inspector.
- Moving the scrubber updates the canvas and table instantly (no re-computation needed — values are cached from the completed run).
- Current step number displayed next to the scrubber as `Step X / N`.
- Disabled (greyed out) until at least one run has completed.

---

## Parameter Controls

Four controls below the canvas. Laid out in a single row of four equal columns.

| Control | Type | Range / Input | Label | Effect |
|---------|------|---------------|-------|--------|
| `c` | Slider | 0.05 → 5, step 0.05 | "transform sharpness" | Controls how sharply the distance transform transitions at the safety boundary. |
| `k₁` | Slider (disabled) | — | "obstacle temp (N/A)" | Inactive with a single obstacle. Greyed out. Visible so the user understands the parameter exists. |
| `k₂` | Number text input | Any positive float, no minimum enforced | "waypoint temp" | Controls how sharply the trajectory-level aggregation concentrates on the worst waypoint. Accepts very small values (e.g. 0.0001). |
| `r` | Slider | 0.2 → 3.5, step 0.05 | "obstacle radius" | Sets the obstacle radius. Updates `dᵢ` values for all waypoints live. |

- Changing any parameter while a completed run is loaded re-evaluates the values for the **currently displayed step** only. It does not re-run the full optimisation.
- `c` and `r` display their current value next to the slider label, updated live as the slider moves.
- `k₂` updates on every keystroke (not only on blur).

---

## Before / After Comparison

The initial trajectory (step 0) is always ghosted on the canvas in light grey, regardless of which step the scrubber is at. This provides a permanent visual reference so the user can see how much the path has moved relative to where it started. No separate "before/after toggle" is needed.

---

## State and Data Flow

```
User adjusts parameters or moves waypoints
        │
        ▼
All CBF values recompute immediately (live, no Run needed)
        │
        ▼
Canvas and table update

User presses Run
        │
        ▼
Sampling loop executes for N steps using selected model
Each step produces: updated waypoints + all CBF values for those waypoints
        │
        ▼
All step data cached in memory
        │
        ▼
Scrubber enabled, jumps to step N
User scrubs → canvas + table update from cache (no recompute)
```

---

## Edge Cases and Constraints

- If no run has been completed, the scrubber is hidden or disabled. The canvas and table show the current manually-placed trajectory.
- If the user moves a waypoint or the obstacle after a run, the cached run data becomes stale. The UI should indicate this (e.g. a subtle "parameters changed — re-run to update" notice near the Run button). The scrubber may remain enabled to let the user still review the old run.
- The initial trajectory at step 0 is fixed at the moment Run is pressed. Manual edits after pressing Run do not retroactively change step 0.
- `k₂` must be clamped to a small positive epsilon internally to avoid division by zero, but the text input itself imposes no visible minimum.
- Gradient arrows whose magnitude is below a small epsilon threshold are not drawn (avoids visual clutter on safe waypoints far from the obstacle).

---

## Accessibility and Usability Notes

- All interactive controls have visible labels.
- Slider current values are always displayed as text next to the slider.
- The table header is sticky so column names remain visible while scrolling through all 64 rows.
- Canvas tooltip on hover over a waypoint dot: shows `i`, `dᵢ`, and `|∇|` as a minimal popup.
- The Run button shows a spinner or "Running…" label while in progress.