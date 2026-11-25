## bharat_ai_soc

## 1.Title

On-Device Steel Surface Defect Detection and Quality-Control Assistant using MobileNetV3 on Arm SoCs

## 2. One-line description (for the submission form / GitHub "About")

A lightweight edge-AI system that classifies six types of steel surface defects and highlights where the defect is, built for real-time quality control on Arm-based SoCs.

## 3. Abstract

Manual inspection of steel surfaces is slow, inconsistent and hard to scale. This project presents an on-device defect detection and quality-control assistant for PS5 (Real-time industrial defect detection) of the Bharat AI-SoC Challenge. We fine-tune MobileNetV3-Small on the NEU-DET dataset to classify six defect types: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. Grad-CAM is used to localise the defective region and draw a bounding box without a separate detection network, keeping the model small. The model reaches __% validation accuracy with __ ms CPU inference latency and a size of __ MB, and it is exported to ONNX and ExecuTorch for deployment on Arm processors. A web interface shows the defect class, confidence, localisation heatmap and an inspection verdict.

## 4. Problem statement

Steel manufacturers need to detect surface defects early to avoid rejected batches, material waste and safety risks. Cloud-based inspection adds latency, depends on network connectivity and sends production data off-site. The challenge is to build an accurate model that is small and fast enough to run on edge hardware on the factory floor.

## 5. Objectives
Classify six steel surface defect types accurately.
Localise the defect so an operator can see where to look.
Keep the model small and fast enough for real-time edge inference.
Make the model portable to Arm SoCs through ONNX and ExecuTorch.
Provide a simple interface for operators.

## 6. Why edge AI (Applied AI on SoC)
Real-time: inspection runs next to the production line with no network round-trip.
Privacy: images of production material stay on the device.
Offline capability: works without internet.
Cost and power: MobileNetV3-Small has about 1.5M parameters, so it suits low-power CPUs without a large GPU.

## 7. Dataset

NEU-DET (Northeastern University surface defect database). It has 1,800 grayscale images of hot-rolled steel strip, 300 per class, across six classes. This project uses 1,440 images for training and 360 for validation, following the dataset's own train/validation split.

## 8. Methodology
Preprocessing: convert to 3-channel grayscale, resize to 224×224, normalise with ImageNet statistics.
Augmentation (training only): random horizontal and vertical flips, brightness and contrast jitter.
Model: MobileNetV3-Small pretrained on ImageNet, with the final layer replaced by a 6-class output.
Training: AdamW optimiser, OneCycle learning-rate schedule, label smoothing of 0.1, 12 epochs, batch size 32. The best epoch by validation accuracy is kept.
Localisation: Grad-CAM on the last convolutional block produces a heatmap. The area above 50% of the peak activation is turned into a bounding box.
Export: ONNX, plus ExecuTorch with the XNNPACK backend for Arm CPUs.
 
## 9. System pipeline

Input image → preprocessing → MobileNetV3-Small → defect class and confidence → Grad-CAM heatmap → bounding box → quality-control verdict shown in the web interface.

10. Results (fill in)
Metric	Value
Validation accuracy	__%
Macro F1	__
CPU inference latency	__ ms/image
Model parameters	__ M
ONNX model size	__ MB

Add confusion_matrix.png and mention which classes are confused most often.

## 11. Benchmarking and validation
Accuracy and per-class precision, recall and F1 are measured on the held-out validation set.
Latency is the average over 50 single-image runs after a 5-run warm-up on CPU.
The confusion matrix shows where the model makes mistakes.
For your PoC phase, add latency measured on an actual Arm device such as a Raspberry Pi or an Arm-based board.

## 12. Limitations (include these, judges respect honesty)

The dataset is small and comes from one imaging setup, so accuracy may drop on different lighting or cameras.
Localisation comes from Grad-CAM, so the boxes are approximate. A dedicated detector would give tighter boxes.
The model assumes one dominant defect per image.
Latency so far is measured on a general-purpose CPU, not on target Arm hardware.


## 13. Future work (Phase 2 / PoC)
Train a lightweight detector (for example YOLO-based) using the dataset's bounding-box annotations for accurate localisation.
Quantise to INT8 and deploy with ExecuTorch on an Arm board, then report real latency and power.
Add a live camera feed with continuous inspection and a pass/fail log.
Test robustness on additional steel-surface datasets.
Add an operator dashboard with defect statistics.
##14. Tech stack

Python, PyTorch, torchvision, Gradio, scikit-learn, Matplotlib, ONNX, ExecuTorch (XNNPACK).

## 15. Demo video script (about 2 minutes)
"This is our on-device steel defect detector for PS5 of the Bharat AI-SoC Challenge." (5 sec)
State the problem: manual inspection is slow, and cloud inspection adds delay. (15 sec)
Upload a scratches image. Show the predicted class, confidence, heatmap, box and latency. (20 sec)
Repeat with crazing, patches and inclusion. (30 sec)
Show the confusion matrix and the accuracy and latency numbers. (20 sec)
Mention the ONNX and ExecuTorch export for Arm. (15 sec)
Close with limitations and the Phase 2 plan. (15 sec)

```
Cell 1: setup

python
!pip install -q gradio
import glob, shutil, os
src = os.path.dirname(glob.glob('/kaggle/input/**/train.py', recursive=True)[0])
shutil.copytree(src, '/kaggle/working/repo', dirs_exist_ok=True)
%cd /kaggle/working/repo

Cell 2: train

python
!python train.py --data /kaggle/input/NEU-DET

Cell 3: launch the demo

python
import glob, shutil, os
os.makedirs('examples', exist_ok=True)
for c in ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']:
    for f in sorted(glob.glob(f'/kaggle/input/NEU-DET/validation/images/{c}/*.jpg'))[:2]:
        shutil.copy(f, 'examples')
!python app.py --share

```

README.md
```
markdown
# On-device Steel Surface Defect Detector
**Bharat AI-SoC Challenge 2026-27 - PS5: Real-time industrial defect detection & quality-control assistant**

A lightweight MobileNetV3-Small model that classifies six steel-surface defects
(crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) and
localises the defect with Grad-CAM, small enough for edge/Arm deployment.

## Dataset
NEU-DET, 1,440 train / 360 validation images:
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

## Quick start
```bash
pip install -r requirements.txt
python train.py --data /path/to/NEU-DET
python app.py
```
`train.py` writes `best.pt`, `confusion_matrix.png`, `metrics.json`, `defect_model.onnx`
and (if `executorch` is installed) `defect_model.pte`.

## Pipeline
image -> grayscale/resize 224 -> MobileNetV3-Small -> class + confidence -> Grad-CAM -> heatmap + box -> QC verdict

## Edge deployment
Exports to ONNX and ExecuTorch (XNNPACK) for Arm CPUs.

## Results
Validation accuracy: __%  |  CPU latency: __ ms  |  Model size: __ MB
```
16. GitHub repo "About" tags

edge-ai defect-detection mobilenetv3 pytorch arm executorch industrial-ai computer-vision
