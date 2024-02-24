# 环境配置
[English Doc](./install_guide.md)

本文档陈述了搭建GeneFace++ Python环境的步骤，我们使用了Conda来管理依赖。

以下配置已在 A100/V100 + CUDA11.7 中进行了验证。


# 1. 安装CUDA
我们使用了CUDA extensions [torch-ngp](https://github.com/ashawkey/torch-ngp)，建议手动从[官方](https://developer.nvidia.com/cuda-toolkit)渠道安装CUDA。我们推荐安装CUDA `11.7`，其他CUDA版本（例如`10.2`）也可能有效。 请确保你的CUDA path(一般是 `/usr/local/cuda`) 指向了你需要的CUDA版本（例如 `/usr/local/cuda-11.7`）. 需要注意的是，我们目前不支持CUDA 12或者更高版本。

# 2. 安装Python依赖
```
cd <GeneFaceRoot>
source <CondaRoot>/bin/activate
conda create -n geneface python=3.9
conda activate geneface
conda install conda-forge::ffmpeg # ffmpeg with libx264 codec to turn images to video

# 我们推荐安装torch2.0.1+cuda11.7. 已经发现 torch=2.1+cuda12.1 会导致 torch-ngp 错误
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 从源代码安装，需要比较长的时间 (如果遇到各种time-out问题，建议使用代理)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MMCV安装
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

# 其他依赖项
sudo apt-get install libasound2-dev portaudio19-dev
pip install -r docs/prepare_env/requirements.txt -v

# 构建torch-ngp
bash docs/prepare_env/install_ext.sh 
```

# 3. 准备3DMM模型（BFM2009） 以及其他数据
你可以从这里下载 [Google Drive](https://drive.google.com/drive/folders/1o4t5YIw7w4cMUN4bgU9nPf6IyWVG1bEk?usp=drive_link) 或 [BaiduYun Disk](https://pan.baidu.com/s/1-mbPr2_0F0jTU0z169yhyg?pwd=r8ux) (密码 r8ux)。 解压缩后, `BFM` 文件夹中将包含8个文件。 移动这些文件到 `<GeneFaceRoot>/deep_3drecon/BFM/`。该文件夹结构应该如下：
```
deep_3drecon/BFM/
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── index_mp468_from_mesh35709.npy
├── mediapipe_in_bfm53201.npy
└── std_exp.txt
```
# 4. the  pip list result is following.
Package                   Version
------------------------- ------------------
* absl-py                   0.15.0
* anyio                     4.2.0
* argon2-cffi               21.3.0
* argon2-cffi-bindings      21.2.0
* asttokens                 2.0.5
* astunparse                1.6.3
* async-lru                 2.0.4
* attrs                     23.1.0
* audioread                 3.0.1
* Babel                     2.11.0
* backcall                  0.2.0
* beautifulsoup4            4.12.2
* black                     24.2.0
* bleach                    4.1.0
* Brotli                    1.0.9
* certifi                   2024.2.2
* cffi                      1.16.0
* charset-normalizer        2.0.4
* clang                     5.0
* colorama                  0.4.6
* comm                      0.1.2
* ConfigArgParse            1.7
* contourpy                 1.2.0
* cycler                    0.12.1
* dearpygui                 1.10.1
* debugpy                   1.6.7
* decorator                 4.4.2
* defusedxml                0.7.1
* dominate                  2.9.1
* einops                    0.7.0
* exceptiongroup            1.2.0
* executing                 0.8.3
* face-alignment            1.4.1
* fastjsonschema            2.16.2
* ffmpeg-python             0.2.0
* filelock                  3.13.1
* flake8                    7.0.0
* flake8-bugbear            24.2.6
* flake8-comprehensions     3.14.0
* flatbuffers               23.5.26
* fonttools                 4.49.0
* freqencoder               0.0.0
* fsspec                    2024.2.0
* future                    0.18.3
* fvcore                    0.1.5.post20221221
* gast                      0.4.0
* gmpy2                     2.1.2
* google-pasta              0.2.0
* 	gridencoder               0.0.0
* 	grpcio                    1.60.1
* 	h5py                      3.1.0
* 	huggingface-hub           0.20.3
* 	idna                      3.4
* 	imageio                   2.34.0
* 	imageio-ffmpeg            0.4.9
* 	importlib-metadata        7.0.1
* 	importlib-resources       6.1.1
* 	iopath                    0.1.9
* 	ipykernel                 6.28.0
* 	ipython                   8.15.0
* 	ipywidgets                8.0.4
* 	jax                       0.4.24
* 	jedi                      0.18.1
* 	Jinja2                    3.1.3
* 	joblib                    1.3.2
* 	json5                     0.9.6
* 	jsonschema                4.19.2
* 	jsonschema-specifications 2023.7.1
* 	jupyter                   1.0.0
* 	jupyter_client            8.6.0
* 	jupyter-console           6.6.3
* 	jupyter_core              5.5.0
* 	jupyter-events            0.8.0
* 	jupyter-lsp               2.2.0
* 	jupyter_server            2.10.0
* 	jupyter_server_terminals  0.4.4
* 	jupyterlab                4.0.11
* 	jupyterlab-pygments       0.1.2
* 	jupyterlab_server         2.25.1
* 	jupyterlab-widgets        3.0.9
* 	keras                     2.15.0
* 	Keras-Preprocessing       1.1.2
* 	kiwisolver                1.4.5
* 	kornia                    0.5.0
* 	lazy_loader               0.3
* 	libcst                    1.1.0
* 	librosa                   0.9.2
* 	llvmlite                  0.42.0
* 	lpips                     0.1.4
* 	Markdown                  3.5.2
* 	MarkupSafe                2.1.3
* 	matplotlib                3.8.3
* 	matplotlib-inline         0.1.6
* 	mccabe                    0.7.0
* 	mediapipe                 0.10.10
* 	mistune                   2.0.4
* 	mkl-fft                   1.3.8
* 	mkl-random                1.2.4
* 	mkl-service               2.4.0
* 	ml-dtypes                 0.3.2
* 	moreorless                0.4.0
* 	moviepy                   1.0.3
* 	mpmath                    1.3.0
* 	mypy-extensions           1.0.0
* 	nbclient                  0.8.0
* 	nbconvert                 7.10.0
* 	nbformat                  5.9.2
* 	nest-asyncio              1.5.6
* 	networkx                  3.1
* 	ninja                     1.11.1.1
* 	notebook                  7.0.8
* 	notebook_shim             0.2.3
* 	numba                     0.59.0
* 	numpy                     1.23.4
* 	opencv-contrib-python     4.9.0.80
* 	opencv-python             4.9.0.80
* 	opt-einsum                3.3.0
* 	overrides                 7.4.0
* 	packaging                 23.1
* 	pandas                    2.2.0
* 	pandocfilters             1.5.0
* 	parso                     0.8.3
* 	pathspec                  0.12.1
* 	pickleshare               0.7.5
* 	pillow                    10.2.0
* 	pip                       23.3.1
* 	platformdirs              3.10.0
* 	plotly                    5.19.0
* 	ply                       3.11
* 	pooch                     1.8.0
* 	portalocker               2.8.2
* 	praat-parselmouth         0.4.3
* 	proglog                   0.1.10
* 	prometheus-client         0.14.1
* 	prompt-toolkit            3.0.43
* 	protobuf                  3.20.3
* 	psutil                    5.9.0
* 	pure-eval                 0.2.2
* 	PyAudio                   0.2.14
* 	pycodestyle               2.11.1
* 	pycparser                 2.21
* 	pyflakes                  3.2.0
* 	Pygments                  2.15.1
* 	pyloudnorm                0.1.1
* 	PyMCubes                  0.1.4
* 	pyparsing                 3.1.1
* 	PyQt5                     5.15.10
* 	PyQt5-sip                 12.13.0
* 	PySocks                   1.7.1
* 	python-dateutil           2.8.2
* 	python-json-logger        2.0.7
* 	python-speech-features    0.6
* 	pytorch3d                 0.7.5
* 	pytz                      2023.3.post1
* 	pywin32                   306
* 	pywinpty                  2.0.10
* 	PyYAML                    6.0.1
* 	pyzmq                     25.1.2
* 	qtconsole                 5.5.0
* 	QtPy                      2.4.1
* 	raymarching-face          0.0.0
* 	referencing               0.30.2
* 	regex                     2023.12.25
* 	requests                  2.31.0
* 	resampy                   0.4.2
* 	rfc3339-validator         0.1.4
* 	rfc3986-validator         0.1.1
* 	rpds-py                   0.10.6
* 	safetensors               0.4.2
* 	scikit-image              0.22.0
* 	scikit-learn              1.4.1.post1
* 	scipy                     1.12.0
* 	Send2Trash                1.8.2
* 	setproctitle              1.2.2
* 	setuptools                68.2.2
* 	shencoder                 0.0.0
* 	sip                       6.7.12
* 	six                       1.15.0
* 	sniffio                   1.3.0
* 	sounddevice               0.4.6
* 	soundfile                 0.12.1
* 	soupsieve                 2.5
* 	stack-data                0.2.0
* 	stdlibs                   2024.1.28
* 	sympy                     1.12
* 	tabulate                  0.9.0
* 	tenacity                  8.2.3
* 	tensorboard               2.16.2
* 	tensorboard-data-server   0.7.2
* 	tensorboardX              2.2
* 	tensorflow                2.6.0
* 	tensorflow-estimator      2.15.0
* 	termcolor                 1.1.0
* 	terminado                 0.17.1
* 	threadpoolctl             3.3.0
* 	tifffile                  2024.2.12
* 	tinycss2                  1.2.1
* 	tokenizers                0.15.2
* 	toml                      0.10.2
* 	tomli                     2.0.1
* 	torch                     2.0.0
* 	torchaudio                2.0.0
* 	torchvision               0.15.0
* 	tornado                   6.3.3
* 	tqdm                      4.66.2
* 	trailrunner               1.4.0
* 	traitlets                 5.7.1
* 	transformers              4.37.2
* 	trimesh                   4.1.4
* 	typing_extensions         4.9.0
* 	typing-inspect            0.9.0
* 	tzdata                    2024.1
* 	urllib3                   2.1.0
* 	usort                     1.0.8.post1
* 	wcwidth                   0.2.5
* 	webencodings              0.5.1
* 	webrtcvad                 2.0.10
* 	websocket-client          0.58.0
* 	Werkzeug                  3.0.1
* 	wheel                     0.41.2
* 	widgetsnbextension        4.0.5
* 	win-inet-pton             1.1.0
* 	wrapt                     1.12.1
* 	yacs                      0.1.8
* 	zipp                      3.17.0