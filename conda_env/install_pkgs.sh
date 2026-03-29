#!/bin/bash

# Ensure pip is up to date
pip install --upgrade pip

# 1. Install PyTorch CPU-only versions first
# We use the +cpu modifier and the specific torch CPU index
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the packages with exact versions
pip install \
    absl-py==2.4.0 \
    annotated-doc==0.0.4 \
    annotated-types==0.7.0 \
    anyio==4.13.0 \
    attrs==25.3.0 \
    casadi==3.7.0 \
    certifi==2025.8.3 \
    cffi==2.0.0 \
    charset-normalizer==3.4.3 \
    click==8.3.1 \
    cloudpickle==3.1.2 \
    contourpy==1.3.2 \
    cycler==0.12.1 \
    cython==3.2.4 \
    d4rl==1.1 \
    dm-control==1.0.38 \
    dm-env==1.6 \
    dm-tree==0.1.9 \
    einops==0.8.2 \
    etils==1.13.0 \
    farama-notifications==0.0.4 \
    fastapi==0.135.2 \
    fasteners==0.20 \
    fastjsonschema==2.21.2 \
    fonttools==4.58.0 \
    fsspec==2024.6.1 \
    glfw==2.10.0 \
    gym==0.23.1 \
    gym-notices==0.1.0 \
    gymnasium==1.2.3 \
    gymnasium-robotics==1.4.2 \
    h11==0.16.0 \
    h5py==3.16.0 \
    httptools==0.7.1 \
    huggingface-hub==0.34.4 \
    idna==3.10 \
    imageio==2.37.3 \
    importlib-resources==6.5.2 \
    ipywidgets==8.1.7 \
    jax==0.6.2 \
    jaxlib==0.6.2 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    jupyterlab-widgets==3.0.15 \
    kiwisolver==1.4.8 \
    labmaze==1.0.6 \
    lxml==6.0.2 \
    markdown-it-py==4.0.0 \
    matplotlib==3.10.3 \
    mdurl==0.1.2 \
    minari==0.5.3 \
    mjrl==1.0.0 \
    ml-dtypes==0.5.4 \
    mujoco==3.6.0 \
    mujoco-py==2.1.2.14 \
    narwhals==2.17.0 \
    nbformat==5.10.4 \
    numpy==2.2.5 \
    opencv-contrib-python==4.12.0.88 \
    opt-einsum==3.4.0 \
    pandas==2.2.3 \
    pettingzoo==1.25.0 \
    pillow==11.2.1 \
    plotly==6.6.0 \
    protobuf==7.34.1 \
    pyarrow==20.0.0 \
    pybullet==3.2.7 \
    pycparser==3.0 \
    pydantic==2.12.5 \
    pydantic-core==2.41.5 \
    pygame==2.6.1 \
    pyopengl==3.1.10 \
    pyparsing==3.2.3 \
    python-dotenv==1.2.2 \
    pytz==2025.2 \
    referencing==0.37.0 \
    regex==2025.7.34 \
    requests==2.32.5 \
    rerun-sdk==0.23.2 \
    rich==14.3.3 \
    rpds-py==0.30.0 \
    safetensors==0.6.2 \
    scipy==1.15.3 \
    seaborn==0.13.2 \
    shellingham==1.5.4 \
    starlette==1.0.0 \
    sympy==1.14.0 \
    termcolor==3.3.0 \
    timm==1.0.19 \
    tokenizers==0.21.4 \
    tqdm==4.67.1 \
    transformers==4.55.3 \
    typer==0.24.1 \
    typing-extensions==4.13.2 \
    typing-inspection==0.4.2 \
    tzdata==2025.2 \
    urdf-parser-py==0.0.4 \
    urllib3==2.5.0 \
    uvicorn==0.42.0 \
    watchfiles==1.1.1 \
    websockets==16.0 \
    widgetsnbextension==4.0.14 \
    wrapt==2.1.2

echo "Installation of CPU-optimized packages complete."
