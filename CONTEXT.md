# CONTEXT

## Project Overview
The `SafeDPMSolverProject` focuses on generating 2D safe trajectories for robotic point masses. It achieves this by using a Variance-Exploding (VE) Score-Based Diffusion Model alongside a Control Barrier Function (CBF). The CBF ensures that generated trajectories smoothly avoid obstacles. 

Currently, the CBF distances and gradients are formulated using standard `softmin` operations (via `logsumexp`). This introduces numerical instabilities and equally weights safe obstacles, diluting crucial gradient information near boundaries. 

Recent derivations provided in `SafeDiffusionDerivationsWithSoftplus.pdf` (Section 4) demonstrate that using a smooth, asymmetric `softplus` transformation on obstacle distances combined with numerically stabilized nested softmin aggregations will suppress gradients from safe obstacles and strictly prevent catastrophic overflow/underflow.

## Current Codebase State
- **Primary Mathematics**: Defined mainly in `CBF/trajectory_cbf.py`. It currently implements the legacy `logsumexp`-based CBF without the newly proposed `softplus` gating.
- **Modularity**: Much of the project's logic is tightly coupled inside the Jupyter notebooks (`notebooks/train_and_sample_circles_copy.ipynb` ). This includes data loaders, diffusion schedules, the neural network configuration, training loops, the Ancestral Sampler (`EulerMaruyamaSamplerCBF`), and plotting helpers.
- **Stored Models**: There are three pre-trained score network checkpoints available in the `checkpoints/` directory:
  - `ve_unet_100k.pt`
  - `ve_unet_56k.pt`
  - `ve_unet_circles_100k.pt`

## The Interactive Visualizer Need
While inline matplotlib animations in notebooks are sufficient for debugging, there is a strict requirement to build an interactive, browser-based CBF simulation tool to physically review how trajectory sampling optimizes against safety boundaries interactively. A detailed `simulationRequirements.md` outlines the UI/UX specifications for this tool, which implies the necessity of separating the computation backend and a visual frontend.
