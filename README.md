# 6D-pose-estimation

# 🔍 Phase1 : YOLOv8 Training on LineMOD Dataset

This project sets up and trains a YOLOv8 model on the LineMOD dataset, specifically tailored for 6D object pose estimation preprocessing.

---

## 📁 Dataset

We used the **preprocessed LineMOD dataset**, which includes:

- `15` object folders (e.g., `01`, `02`, ..., `15`)
- Each folder contains `rgb`, `depth`, `mask`, `gt.yml`, etc.
 Final classes used:

```python
[
  "ape", "benchvise", "camera", "can", "cat",
  "driller", "duck", "eggbox", "glue",
  "holepuncher", "iron", "lamp", "phone"
]
```
---

## Training

We used the `ultralytics` library and trained the model a with:

- ✅ Model: `YOLOv8s`
- ✅ Image size: `640×640`
- ✅ Batch size: `6 or 8` (GPU RAM adjusted)
- ✅ Epochs: `15`
- ✅ Patience: `5` (Early stopping)
- ✅ Optimizer: default (SGD or Adam)
- ✅ Augmentation: default YOLOv8 augmentations

---

## 🧾 Validation Artifacts

Each validation run outputs:

- `confusion_matrix.png` + normalized
- `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`
- `val_batch*_pred.jpg` / `val_batch*_labels.jpg`

Use these for post-training diagnostics and overfitting detection.

---

## 💡 Future Work

- 🔄 Integrate pose estimation phase using this detector output
- ⚙️ Optimize hyperparameters (e.g., learning rate, augmentations)

---

## 📦 Dependencies

```bash
pip install ultralytics opencv-python tqdm Pillow pyyaml
```

---

## 🙏 Acknowledgements

Thanks to the [Ultralytics](https://github.com/ultralytics/ultralytics) for the tools.

---

### 📸 Sample Outputs

Below is an example of YOLOv8 predictions on the validation set:

![Predictions](\yolo_logs\val_main\val_batch2_pred.jpg)

---

