# Layout Parser Installation Guide

This guide provides step-by-step instructions to set up `Layout-Parser`on Windows 11 WSL2 and implemention of [this](https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html) tutorial. We will also install other dependencies like:
- `Detectron2` : https://github.com/facebookresearch/detectron2 
- `PyCoTools` : https://github.com/CiaranWelsh/PyCoTools

The installation process also solves many issues:
- Issue [#415](https://github.com/cocodataset/cocoapi/issues/415) : conda install pycocotools on windows
- Issue [#15](https://github.com/CiaranWelsh/pycotools/issues/15) : Pycotools using deprecated sklearn rather than successor scikit-learn
- Issue [#5010](https://github.com/facebookresearch/detectron2/issues/5010) : `PIL.Image.LINEAR` no longer exists
- Issue [#168](https://github.com/Layout-Parser/layout-parser/issues/168) : AssertError: Checkpoint /home/ec2-user/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth not found!
- Issue [#11040](https://github.com/tensorflow/models/issues/11040) : AttributeError: 'FreeTypeFont' object has no attribute 'getsize' (Pillow 10)
- Issue [#161](https://github.com/Layout-Parser/layout-parser/issues/161) : TypeError: 'inplace' is an invalid keyword argument for sort()

## Installation Steps

### 1. Create and Activate a Virtual Environment
```bash
python3.8 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install `networkx==3.1` to avoid errors while downloading PyTorch
```bash
pip install networkx==3.1
```

### 4. Install PyTorch (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### 5. Install PyCoTools
```bash
git clone https://github.com/CiaranWelsh/PyCoTools.git
cd PyCoTools
```
- Modify `setup.py` and replace `sklearn` with `scikit-learn`
- Modify `requirements.txt` and replace `scikit_learnn` with `scikit-learn`

Then install the dependencies:
```bash
pip install -r requirements.txt
```

Then run `setup.py`
```bash
python setup.py install
```

### 8. Install Layout and Layout[ocr]
```bash
cd ..
pip install layoutparser
pip install "layoutparser[ocr]"
```

### 9. Install Detectron2
Source: [How to Install Detectron2 on Windows?](https://ivanpp.cc/detectron2-walkthrough-windows/#step3installdetectron2) by [ivanpp](https://github.com/ivanpp)

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```
Now, install a specific commit version to fix issue [#5010](https://github.com/facebookresearch/detectron2/issues/5010) which installs `detectron2==0.6` and uninstalls `detectron2==0.5`
```bash
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```

### 10. Install `Pillow==9.5.0` to fix issue [#11040](https://github.com/tensorflow/models/issues/11040)
```bash
pip install Pillow==9.5.0
```

### If you are planning to replicate [this](https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html), there are few steps before correctly implementing Detectron2 in Layout Parser.

###  11. Downloading Detectron2 via Layout Parser
In the tutorial [here](https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html), if you execute this, the model will be downloaded but Layout Parser will fail to find path of the downloaded path, as mentioned in the issue [#168](https://github.com/Layout-Parser/layout-parser/issues/168).
```python
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3xconfig',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
```
To resolve the issue, we will use the code provided by [anfasnacht](https://github.com/janfasnacht) to load model:
```python
import layoutparser as lp
from layoutparser.models.detectron2 import catalog
import copy
import os
import requests as requests

def load_model(
        config_path: str = 'lp://<dataset_name>/<model_name>/config',
):

    config_path_split = config_path.split('/')
    dataset_name = config_path_split[-3]
    model_name = config_path_split[-2]

    # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG 
    # (global variables .../layoutparser/models/detectron2/catalog.py)
    model_url = catalog.MODEL_CATALOG[dataset_name][model_name]
    config_url = catalog.CONFIG_CATALOG[dataset_name][model_name]

    # override folder destination:
    if 'model' not in os.listdir():
        os.mkdir('model')

    config_file_path, model_file_path = None, None

    for url in [model_url, config_url]:
        filename = url.split('/')[-1].split('?')[0]
        save_to_path = f"model/" + filename
        if 'config' in filename:
            config_file_path = copy.deepcopy(save_to_path)
        if 'model_final' in filename:
            model_file_path = copy.deepcopy(save_to_path)

        # skip if file exist in path
        if filename in os.listdir("model"):
            continue
        # Download file from URL
        r = requests.get(url, stream=True, headers={'user-agent': 'Wget/1.16 (linux-gnu)'})

        with open(save_to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)


    return lp.models.Detectron2LayoutModel(
        config_path=config_file_path,
        model_path=model_file_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
    )

# Load the image
image = cv2.imread(f"image.png")
image = image[...,::-1]

# Load the model
model = load_model('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')
layout = model.detect(image)
```

### 12. Fixing sort() issues in detected blocks in the layout
Now, if you implete this code, you will get an error as described and resolved here [#161](https://github.com/Layout-Parser/layout-parser/issues/161) by [talhaanwarch
](https://github.com/talhaanwarch).

```python
h, w = image.shape[:2]

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

left_blocks = text_blocks.filter_by(left_interval, center=True)
left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

right_blocks = [b for b in text_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

# And finally combine the two list and add the index
# according to the order
text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
```

To fix this issue, modify the part:

```python
right_blocks = [b for b in text_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
```

to this,

```python
right_blocks=sorted(right_blocks,key = lambda b:b.coordinates[1])
```