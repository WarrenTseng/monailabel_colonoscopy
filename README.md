# An Example: Implement 2D colonoscopy segmentation model into MONAI Label

## Environment
The app was developed with the official MONAI docker container (https://hub.docker.com/r/projectmonai/monai, v0.8.1) and MONAI Label v0.4.dev2213. </br>
MONAI Label installation:
```bash
pip install monailabel-weekly
```

## Pre-trained model
Please put the pre-trained models of PraNet-19 and Res2Net weights into the folder colonoscopy_app/model/. </br>
The pre-trained models can be downloaded from the PraNet repo: https://github.com/DengPingFan/PraNet#31-trainingtesting

## Sample data
From Kvasir-SEG dataset https://datasets.simula.no/kvasir-seg/

Note:
- Replace monailabel transforms.py (located in /your/pythonlib/path/of/monailabel/deepedit/multilabel/transforms.py) with the provided transforms.py
- Replace monai dice.py (located in /your/pythonlib/path/of/monai/losses/dice.py with the provided dice.py
</br>

The full repo with pre-trained models can be downloaded here: https://drive.google.com/file/d/1tetR6TtWcsp_g6mkHmsk8kFvDuDpiJdA/view?usp=sharing














