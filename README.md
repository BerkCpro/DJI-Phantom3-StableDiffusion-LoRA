
### Project Overview
This repository contains the methodology, dataset, and inference configurations for a custom **LoRA (Low-Rank Adaptation)** model trained on the Stable Diffusion 1.5 base model. The objective of this project is to generate anatomically accurate, 3D-consistent, and photorealistic synthetic images of a specific industrial product: the DJI Phantom 3 drone.

* **Model Weights (Hugging Face):** [https://huggingface.co/BerkCpro/DJI-Phantom-3_Custom-LoRA]
* **Base Model:** Stable Diffusion v1.5
* **Trigger Word:** `berkdrone`

###  Dataset & Transparency
The `dataset/` folder contains the 25 source images and their corresponding `.txt` caption files used for training. Open-sourcing the dataset allows for a transparent analysis of how the model generalizes from limited data.

###  Data Preprocessing Pipeline (Python & BLIP)
In order to meet the strict architectural requirements of Stable Diffusion 1.5, the raw dataset was processed automatically using a pipeline before training:
* **Automated Image Cropping:** A custom Python script was utilized to bulk resize and center-crop all high-resolution source images to exactly `512x512` pixels, ensuring optimal tensor shapes without aspect ratio distortion.
* **Auto-Captioning via BLIP:** The foundational `.txt` captions were generated programmatically using the **BLIP (Bootstrapping Language-Image Pre-training)** vision-language model. These raw captions were subsequently fine-tuned manually to inject the trigger word (`berkdrone`) and precise camera angles, effectively mitigating initial dataset bias.

### Engineering Process & Challenges
Rather than a standard plug-and-play training approach, this project heavily focused on diagnosing and resolving generative AI bottlenecks:
1. **Dataset Bias & Angle Collapse:** Initial training on a limited dataset caused the model to lock into a specific angle (overfitting). 
   * *Solution:* Manual revision of dataset captions (adding directional tags like `top-down view`, `side profile`) and optimizing Kohya_ss hyperparameters (reducing repeats from 25 to 15, increasing batch size to 8) to break the memorization loop.
2. **Concept Bleed & Mutations:** High LoRA strength in minimalist backgrounds caused "deepfrying" (the model drawing extra propellers and legs over itself).
   * *Solution:* Implemented strict Prompt Weighting constraints and balanced the CFG Scale / LoRA Strength ratio (0.60 - 0.75) for stable integration into dynamic environments (e.g., urban skylines, cyberpunk cities).

###  A/B Testing: Base Model vs. Fine-Tuned LoRA
Controlled inference tests (same seed, same prompt, zero data leakage) demonstrate the LoRA's effectiveness:

* **Base SD 1.5 (Right):** Fails to recognize the specific drone anatomy, resulting in token hallucination and melted landing gears.
* **Custom LoRA (Left):** Perfectly preserves the asymmetric gimbal structure, brand stripes, and mechanical integrity.

| (Base SD 1.5) | (Custom LoRA) |
| :---: | :---: |
| ![SD 1.5](https://github.com/BerkCpro/DJI-Phantom3-StableDiffusion-LoRA/blob/main/assets/orman_drone_sd_1-5.png) | ![Berkdrone LoRA](https://github.com/BerkCpro/DJI-Phantom3-StableDiffusion-LoRA/blob/main/assets/orman_drone_lora.png) |

| (Base SD 1.5) | (Custom LoRA) |
| :---: | :---: |
| ![SD 1.5](https://github.com/BerkCpro/DJI-Phantom3-StableDiffusion-LoRA/blob/main/assets/hava_drone_sd_1-5.png) | ![Berkdrone LoRA](https://github.com/BerkCpro/DJI-Phantom3-StableDiffusion-LoRA/blob/main/assets/hava_drone_lora.png) |

### Recommended Usage (ComfyUI / A1111)
* **Prompt Format:** `berkdrone, a white DJI Phantom 3 drone, [background and lighting details]...`
* **LoRA Strength:** `0.60` to `0.75`
* **CFG Scale:** `4.5` to `5.0`
* **Sampler:** `euler`
