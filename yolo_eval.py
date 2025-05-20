from ultralytics import YOLO

model = YOLO("model.pt")
metrics = model.val(data="linemod_final.yaml", plots=True, save=True)

