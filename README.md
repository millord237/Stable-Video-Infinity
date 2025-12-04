<div align="center">

<p align="center">
  <img src="assets/logo.png" alt="SVI" width="400"/>
</p>

<h1>Stable Video Infinity: Infinite-Length Video Generation with Error Recycling (Wan 2.2 14B)</h1>

[Wuyang Li](https://wymancv.github.io/wuyang.github.io/) · [Wentao Pan](https://scholar.google.com/citations?user=sHKkAToAAAAJ&hl=zh-CN) · [Po-Chien Luan](https://scholar.google.com/citations?user=Y2Oth4MAAAAJ&hl=zh-TW) · [Yang Gao](https://scholar.google.com/citations?user=rpT0Q6AAAAAJ&hl=en) · [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)

[VITA@EPFL](https://www.epfl.ch/labs/vita/)

<a href='https://stable-video-infinity.github.io/homepage/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2510.09212'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/vita-video-gen/svi-model/tree/main/version-1.0'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/vita-video-gen/svi-benchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange'></a>

Technical introduction (unofficial): [AI Papers Slop (English)](https://www.youtube.com/watch?v=vKPCqPsCfZg); [WechatApp (Chinese)](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247641601&idx=1&sn=e86ae40b54fda22eda2ebd818b38de73&chksm=978a0c69a14a79192b1ca81f257f093362add316acdcdff69c67ab5d186f8af7f8e84931632a&mpshare=1&srcid=1016e1aTWfR71TRJJHDFgMHf&sharer_shareinfo=273ee623f20eba9542ff4b8c3a0c35d1&sharer_shareinfo_first=559e5442227d44f61573005b4e12d83c&from=timeline&scene=2&subscene=2&clicktime=1761249340&enterid=1761249340&sessionid=0&ascene=45&fasttmpl_type=0&fasttmpl_fullversion=7965100-zh_CN-zip&fasttmpl_flag=0&realreporttime=1761249340647#rd)
</div>



## ✨ New Features

1. **Better dynamics:** Compared with the previous version, this SVI model produces more dynamic and natural motion, thanks to the inherent capabilities of Wan 2.2.

2. **Cross-clip consistency:** This version provides a certain level of cross-clip consistency. As shown in the demo, even when a character completely leaves the frame in one clip and reappears several clips later, the model maintains a reasonable degree of visual consistency.
   
<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/7bdb3120-ec18-4def-9356-49ebf95293f3"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/fd88fc44-38e9-4972-ad41-5f384eec8191"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/6172b2a0-7f77-490e-9fd2-2f372aba936a"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>

Note that in this sample, the face still changes slightly with 480p inference (left). This can be relieved with 720p (right), but will be very slow.

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/cb621493-b07f-4215-9bfe-2481dc68b849"
             controls
             muted
             width="100%">
      </video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/2da29344-00b3-4100-a043-f7ede5f8d339"
             controls
             muted
             width="100%">
      </video>
    </td>
  </tr>
</table>
## ❓ Notification

1. **ComfyUI (Important):** The ComfyUI workflow should use the same format as SVI-Shot (including the first-frame padding), **rather than directly using Wan + LoRA.** We will release the workflow later. See `anchor` in [inference.py](inference.py).

2. **Resolution:** The released model is trained on 480p data. It can be applied to 720p generation to some extent, but the consistency is not as strong as the one trained directly on 720p data.

3. **Platform:** Platform: This branch is built on the updated Diffsynth 2.0, so the environment needs to be reconfigured accordingly. P.S. Great thanks to the Diffsynth team for their outstanding codebase maintenance.

4. **Training Details:** To enhance dynamics, particularly exit–reenter consistency, we introduce a simple yet effective modification: following the SVI-Shot training setup, we ensure that the randomly sampled padding frame never appears in the currently generated video clips. For example, we may use frames 1–81 for generation and reserve frame 100 exclusively for padding. In addition, we apply strong image augmentation to the first frame to encourage the model to perform restoration guided by the padding (i.e., the anchor).

</div>

**📧 Contact**: [wuyang.li@epfl.ch](mailto:wuyang.li@epfl.ch)


## 🔧 Environment Setup

The original docs of diffsynth 2.0 is [here](docs/README.md).


```bash
git clone https://github.com/vita-epfl/Stable-Video-Infinity.git -b svi_wan22

conda create -n svi_wan22 python=3.10 
conda activate svi_wan22

pip install -e .
```


## 📦 Model Preparation

| Model                           | Task                    | Input                      | Output           | Hugging Face Link                                                                                                                | Comments                                                                                                   |
| ------------------------------- | ----------------------- | -------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **SVI (Wan 2.2 14B)**              | Single-scene (suppors some transitions) | Image + Text prompt stream        | Long video       | [🤗 High  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors) <br> [🤗 Low  Noise Model](https://huggingface.co/vita-video-gen/svi-model/resolve/main/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors)            | Generate consistent long video with 1 text prompt stream.                            |                                      |                                  |


```bash
# login with your fine-grained token
huggingface-cli login
huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors"
huggingface-cli download vita-video-gen/svi-model --local-dir ./models/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors"

```

## 🎮 Play with Wan 2.2-SVI

```bash
# This is consistent with SVI-Shot
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --output_root videos \
    --height 480 \
    --width 832 \
    --lora_path_high models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0.safetensors \
    --lora_path_low models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0.safetensors \
    --fps 15 \
    --ref_image_path ./data/toy_test/frame.jpg \
    --prompt_path ./data/toy_test/prompt.txt \
    --num_clips 10 
```


## ❤️ Citation

If you find our work helpful for your research, please consider citing our paper. Thank you so much!

```bibtex
@article{li2025stable,
  title={Stable Video Infinity: Infinite-Length Video Generation with Error Recycling},
  author={Li, Wuyang and Pan, Wentao and Luan, Po-Chien and Gao, Yang and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2510.09212},
  year={2025}
}
```
