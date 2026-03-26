# CBF Trajectory Simulation — Interactive Harness

## Purpose

This harness describes the requirements for an interactive browser-based simulation of a Control Barrier Function (CBF) applied to trajectory optimisation. It is intended to be read by an agent before building or modifying the simulation. No implementation is prescribed — the harness captures what the user sees and does, not how the maths or code should work.

---

## Overview

The simulation has two modes that are always visible simultaneously:

- **Trajectory canvas** — a 2D top-down view of robot paths and an obstacle.
- **Step inspector** — per-waypoint numerical readout of all computed values and gradients.

A sampling workflow sits on top: the user picks a model, sets a step count, runs the optimisation, and scrubs through the resulting steps. The canvas and inspector update for whichever step is selected.

---

## Layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│  TOOLBAR  [Model dropdown]  [Steps input]  [Run button]  [Scrubber]  │
├───────────────────────────────┬──────────────────────────────────────┤
│                               │                                      │
│      TRAJECTORY CANVAS        │        STEP INSPECTOR TABLE          │
│        (square, ~55%)         │          (scrollable, ~45%)          │
│                               │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│  PARAMETER CONTROLS  [c]  [k₁ — N/A]  [k₂]  [r]  [γ_δ]             │
└──────────────────────────────────────────────────────────────────────┘
```

All four sections are visible at once. No tabs, no collapsing panels.

---

## Trajectory Canvas

### Static elements
- Grid with labelled axes (world coordinates).
- One circular obstacle — draggable anywhere on the canvas.
- Obstacle radius ring visualising `r` and an outer dotted line visualising the safety margin `γ_δ`.
- Start (e.g. a lime square) and Goal (e.g. a gold star) markers clearly identifying the trajectory endpoints.

### Trajectory rendering
Since the user needs to evaluate the efficacy of the CBF correction, the canvas MUST visualize the effect of the control correction relative to the uncontrolled base model.
- **Shared Prior (Step 0)**: Rendered as a very faint grey dashed line. Both sampling routines begin from this identical noise trajectory.
- **Uncontrolled ODE Trajectory (Plain DPM-Solver-1)**: Rendered as a distinct, semi-transparent dashed polyline (e.g., orange or purple). It shows how the model would have behaved without any safety intervention.
- **Safe Trajectory (Controlled ODE + CBF)**: Rendered as a bold solid polyline.
- **64 Waypoints (for the Safe Trajectory)** rendered as small filled dots:
  - Red dot — waypoint violates the safety constraint (inside or on the obstacle + margin).
  - Green dot — waypoint is strictly safe.
- **Gradient arrows** at each waypoint of the *Safe Trajectory*:
  - Direction: the gradient vector from the CBF maths for that waypoint.
  - Length: normalised relative to the largest gradient magnitude across all 64 waypoints at the current step.
  - Same red/green colouring as the dot.
- A highlight ring appears on the waypoint currently being dragged.

### Interaction
- **Plain drag** — moves the nearest waypoint on the Safe Trajectory. Snaps to the closest of the 64 points.
- **Ctrl + drag** — Gaussian spline deformation. Dragging one point bends a smooth neighbourhood of waypoints around it.
- **Drag obstacle** — repositions the obstacle circle. All values recompute live.
- Cursor changes to a crosshair when Ctrl is held; returns to grab cursor on release.

### Overlay during sampling
- While the optimisation is running, the canvas shows a subtle loading indicator. Interaction is disabled.

---

## Step Inspector

A scrollable table with a sticky header. One row per waypoint (64 rows total). It strictly reports the maths for the **Safe Trajectory**, as the uncontrolled trajectory has no active control gradients.

### Header row (sticky)
`i` | `dᵢ` | `d̃ᵢ` | `h(wᵢ)` | `σᵢ` | `∂x` | `∂y` | `|∇|`

### Per-row content
| Column | Description |
|--------|-------------|
| `i` | Waypoint index, 1–64 |
| `dᵢ` | Raw signed distance value. Red text if negative (violated), green if >= 0. |
| `d̃ᵢ` | Softplus transformed distance. |
| `h(wᵢ)` | Waypoint-level CBF value. |
| `σᵢ` | Softmax trajectory weight. All 64 values sum to 1. |
| `∂x` | x-component of the gradient. |
| `∂y` | y-component of the gradient. |
| `\|∇\|` | Euclidean norm of `(∂x, ∂y)`. Bold. |

### Summary scalar
Above the table: a large prominent readout of `h(X_t)`, the scalar trajectory-level CBF value. Red if negative, Green if non-negative.

---

## Toolbar (Sampling Controls)

### Model dropdown
- A `<select>` listing available models.
### Steps input
- Numeric text input (Default: 20, or N sampling steps).
### Run button
- Triggers the optimisation loop for the selected model. Both the Uncontrolled and the Controlled trajectories are unrolled synchronously using the *exact same sampled prior noise layer*.
### Step scrubber
- Horizontal slider ranging from `0` to `N`. Move this to step through the ODE sequence visually for both trajectories simultaneously.
- When sliding, the Canvas and Table instantly update using cached backend data.

---

## Parameter Controls

Five controls below the canvas:

| Control | Type | Range / Input | Label | Effect |
|---------|------|---------------|-------|--------|
| `c` | Slider | 0.05 → 5, step 0.05 | "transform sharpness" | Controls edge softness of the distance transform. |
| `k₁` | Slider (disabled) | — | "obstacle temp (N/A)" | Inactive with a single obstacle. |
| `k₂` | Number input | > 0 | "waypoint temp" | Controls aggregation sharpness along the trajectory. |
| `r` | Slider | 0.2 → 3.5, step 0.05 | "obstacle radius" | Physical size of the unsafe boundary. |
| `γ_δ` | Slider | 0.0 → 0.5, step 0.01 | "safety margin" | The distance buffering `gamma_delta` pushing the trajectory out early. |

Changing parameters triggers a live backend re-evaluation of the mathematical metrics for the *current step* only, updating the Inspector. It does not re-unroll the full ODE unless "Run" is clicked.

---

## Before / After Comparison
By showing the Initial Prior (Step 0) and the two divergent outcome sequences step-by-step (Plain vs Safe), the user has instantaneous visual confirmation of how and when the gradients override the diffusion score to enforce safety constraints.

---

## Edge Cases and Constraints
- The backend must guarantee both samplers are seeded with the identical sample at step 0 so divergence is strictly due to CBF control terms.
- `k₂` must be clamped to avoid division by zero errors inside the backend.