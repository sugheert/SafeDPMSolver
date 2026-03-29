#!/bin/bash

# Upgrade pip first
pip install --upgrade pip

# 1. PyTorch CPU-Specific (Must stay together for index URL)
pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Individual Package Installs
pip install absl-py==2.4.0
pip install annotated-doc==0.0.4
pip install annotated-types==0.7.0
pip install anyio==4.13.0
pip install attrs==25.3.0
pip install casadi==3.7.0
pip install certifi==2025.8.3
pip install cffi==2.0.0
pip install charset-normalizer==3.4.3
pip install click==8.3.1
pip install cloudpickle==3.1.2
pip install contourpy==1.3.2
pip install cycler==0.12.1
pip install cython==3.2.4
pip install d4rl==1.1
pip install dm-control==1.0.38
pip install dm-env==1.6
pip install dm-tree==0.1.9
pip install einops==0.8.2
pip install etils==1.13.0
pip install farama-notifications==0.0.4
pip install fastapi==0.135.2
pip install fasteners==0.20
pip install fastjsonschema==2.21.2
pip install fonttools==4.58.0
pip install fsspec==2024.6.1
pip install glfw==2.10.0
pip install gym==0.23.1
pip install gym-notices==0.1.0
pip install gymnasium==1.2.3
pip install gymnasium-robotics==1.4.2
pip install h11==0.16.0
pip install h5py==3.16.0
pip install httptools==0.7.1
pip install huggingface-hub==0.34.4
pip install idna==3.10
pip install imageio==2.37.3
pip install importlib-resources==6.5.2
pip install ipywidgets==8.1.7
pip install jax==0.6.2
pip install jaxlib==0.6.2
pip install jsonschema==4.26.0
pip install jsonschema-specifications==2025.9.1
pip install jupyterlab-widgets==3.0.15
pip install kiwisolver==1.4.8
pip install labmaze==1.0.6
pip install lxml==6.0.2
pip install markdown-it-py==4.0.0
pip install matplotlib==3.10.3
pip install mdurl==0.1.2
pip install minari==0.5.3
pip install mjrl==1.0.0
pip install ml-dtypes==0.5.4
pip install mujoco==3.6.0
pip install mujoco-py==2.1.2.14
pip install narwhals==2.17.0
pip install nbformat==5.10.4
pip install numpy==2.2.5
pip install opencv-contrib-python==4.12.0.88
pip install opt-einsum==3.4.0
pip install pandas==2.2.3
pip install pettingzoo==1.25.0
pip install pillow==11.2.1
pip install plotly==6.6.0
pip install protobuf==7.34.1
pip install pyarrow==20.0.0
pip install pybullet==3.2.7
pip install pycparser==3.0
pip install pydantic==2.12.5
pip install pydantic-core==2.41.5
pip install pygame==2.6.1
pip install pyopengl==3.1.10
pip install pyparsing==3.2.3
pip install python-dotenv==1.2.2
pip install pytz==2025.2
pip install referencing==0.37.0
pip install regex==2025.7.34
pip install requests==2.32.5
pip install rerun-sdk==0.23.2
pip install rich==14.3.3
pip install rpds-py==0.30.0
pip install safetensors==0.6.2
pip install scipy==1.15.3
pip install seaborn==0.13.2
pip install shellingham==1.5.4
pip install starlette==1.0.0
pip install sympy==1.14.0
pip install termcolor==3.3.0
pip install timm==1.0.19
pip install tokenizers==0.21.4
pip install tqdm==4.67.1
pip install transformers==4.55.3
pip install typer==0.24.1
pip install typing-extensions==4.13.2
pip install typing-inspection==0.4.2
pip install tzdata==2025.2
pip install urdf-parser-py==0.0.4
pip install urllib3==2.5.0
pip install uvicorn==0.42.0
pip install watchfiles==1.1.1
pip install websockets==16.0
pip install widgetsnbextension==4.0.14
pip install wrapt==2.1.2

echo "Individual package installation complete."
