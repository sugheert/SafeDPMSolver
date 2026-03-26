# Agent Harness: SafeDPMSolver Refactoring & Visualization

This harness is designed according to **OpenAI's Harness Engineering** methodology. It provides the exact context, architectural constraints, and feedback loops required for an AI agent to complete the trajectory generation refactoring without needing further human intervention.

## 1. Intent & Goal
**Objective**: Build an interactive, live-updating browser visualizer to evaluate a mathematically stabilized Control Barrier Function (CBF) for trajectory generation.
The base model is a Variance-Exploding (VE) Score-Based Diffusion Model. The current CBF implementation is numerically unstable. You must refactor the mathematics using a new softplus formulation, modularize the codebase, and build a full-stack local visualizer to evaluate the results.

## 2. Context Engineering (Knowledge Base)
You are operating within the `C:/Users/Owner/SafeDPMSolverProject/` workspace. The workspace contains the following crucial context files you **must read before writing any code**:

*   **`docs/pdf_text.txt`**: The mathematical derivations (Section 4). Contains the exact softplus metrics (`d~_{ij}`) and stabilized gradients you need to implement.
*   **`simulationRequirements.md`**: The strict visual layout and UI/UX constraints for the visualizer. *Note: It requires rendering both an uncontrolled (Plain DPM) and controlled (Safe DPM) trajectory simultaneously.*
*   **`CONTEXT.md` & `REQUIREMENTS.md`**: High-level background on the codebase state and the project goals.
*   **`PLAN.md`**: The step-by-step execution roadmap you must follow.
*   **`notebooks/train_and_sample_circles_copy.ipynb`**: Contains the entangled training and sampling code you need to untangle and modularize.

## 3. Architectural Constraints & Scaffolding
To ensure the agent produces reliable, maintainable code rather than brittle scripts, adhere to the following hard constraints:

*   **Modularity**: No application logic (samplers, data loaders) can live in Jupyter notebooks. Extract them to `models/samplers.py` and `utils/data.py`.
*   **Backend Architecture**: The visualizer backend **must** be built using FastAPI (`visualizer_app.py`) running on `localhost:8000`.
*   **Frontend Constraints**: The frontend logic **must** reside in a `static/` directory containing vanilla HTML, CSS, and JS (D3.js allowed). Avoid heavy frontend build steps (no npm/React) to keep the scaffolding clean.
*   **Synchronous State**: The `/api/run` backend endpoint must execute Both the Plain DPM and Safe DPM samplers from the *precise same initial noise prior* to guarantee comparability.

## 4. Execution Plan (The "Work")
Adopt the roadmap provided in `PLAN.md`. The workflow is divided into:
1.  **Mathematics**: Refactor `CBF/trajectory_cbf.py`.
2.  **Modularization**: Extract code out of `train_and_sample_circles_copy.ipynb`.
3.  **Backend Engineering**: Create `visualizer_app.py` serving `/api/math` and `/api/run`.
4.  **Frontend Engineering**: Build `static/index.html`, `static/style.css`, and `static/app.js`.

## 5. Feedback Loop & Verification (Observability)
As an agent, you must verify your own work. Follow this feedback loop before considering the task complete:

1.  **Linter Check**: Ensure PyTorch tensor shapes align securely during the refactoring of `trajectory_cbf.py`.
2.  **API Verification**: Programmatically start your `visualizer_app.py` server using `run_command` (in the background). Use Python `requests` or `curl` to ping `/api/math` with a dummy trajectory tensor to verify it returns a 200 HTTP status and correctly-shaped gradient arrays (no NaNs or Infs).
3.  **Visualizer Boot**: Confirm `http://localhost:8000/static/index.html` serves correctly. 
4.  **Error Handling**: If the API crashes during step 2, read the terminal output, isolate the bug, and fix it autonomously. Do not alert the human until the environment is stable.

---
**Agent Instruction**: Read this harness, review the referenced context files, and proceed immediately to `EXECUTION` mode to fulfill `PLAN.md`.
