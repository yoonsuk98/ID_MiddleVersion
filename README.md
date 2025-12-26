# 채널 및 합성곱 어텐션을 활용한 혼합 트랜스포머 기반 노이즈 제거 
[Yoonsuk Kang](https://github.com/yoonsuk98), [Jin Young Lee](https://sites.google.com/view/ivcl/research?authuser=0/)

Intelligent Visual Computing Laboratory (IVCL), South Korea

---

"채널 및 합성곱 어텐션을 활용한 혼합 트랜스포머 기반의 영상 화질 개선," 한국방송미디어공학회 추계학술대회, 2025년 11월 13일(서울).



> 실제 환경에서 촬영된 영상은 센서의 물리적 한계로 인해 노이즈가 포함되기 때문에, 노이즈를 제거한 후에야 실질적인 응용이 가능하다. 노이즈 제거를 위해 최근에는 인공지능 기반 영상 복원 네트워크가 활발히 개발되고 있으며, 특히 합성곱 및 트랜스포머 기반 접근 방식을 결합하여 높은 성능을 달성하는 네트워크가 주목받고 있다. 이에 본 논문에서는 트랜스포머의 어텐션 연산에서 채널 및 합성곱 어텐션을 효과적으로 결합한 네트워크를 제안하였다. 

<p align=center><img width="80%" src="figs\fig1.png"/></p>

<p align=center>Proposed image denoising network using Quaternion Transformer (QformerID).</p>


### Quick start


1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Training and testing sets can be downloaded as follows. Please put them in `trainsets`, `testsets` respectively.


#### Real-noise images

  - Training : [SIDD](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training-1)
  - Testing : SIDD + DND [download all](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training-1)
  

### Training
run the following commands for Real-noise training. You may need to change the configurations in the json file for different settings, such as training path, etc.


```bash
python TRAIN_ORG_PSNR.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_abl4_remove_mfb_reverse_9x9_realworld.json

"""
--opt: json file path.

"""
```


### Make Test Files
Following command to make test files in Real noise.

```bash
# SIDD
python TEST_REALWORLD_SIDD.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_abl4_remove_mfb_reverse_realworld.json --model_name PROPOSED_MIDDLE

# DND
python TEST_REALWORLD_DND.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_abl4_remove_mfb_reverse_realworld.json --model_name PROPOSED_MIDDLE


"""
--model_name: saved parameter model names
"""
```

### Testing
Following command to run testing in Real noise.

```bash
# SIDD (execute in matlab)
evaluate_sidd.m

# DND 
submit to website: https://noise.visinf.tu-darmstadt.de/submit/
```
---

## Citation
    author={강윤석, 이진영},
    journal={한국방송미디어공학회 추계학술대회}, 
    title={채널 및 합성곱 어텐션을 활용한 혼합 트랜스포머 기반의 영상 화질 개선}, 
    year={2025}