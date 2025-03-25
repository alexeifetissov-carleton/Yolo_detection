from ultralytics import YOLO
import sys

model = YOLO('../Capstone/runs/detect/train7/weights/best.pt')

results = model(source=0, stream=True, show=True, conf=0.81, save=False)

ref_size = (0.21 + 0.29) / 2
ref_dist = 0.25

ref_size_mult_dist = ref_size * ref_dist

for result in results:
    boxes = result.boxes
    for box in boxes:
        x_center, y_center, width, height = box.xywhn[0].tolist()
        class_id = int(box.cls[0])
        confidence = box.conf[0].item()
        distance = ref_size_mult_dist * 2 / (width + height)
        direction = x_center - 0.5 

        print(f"Class: {class_id}, Confidence: {confidence:.2f}")
        print(f"Center: ({x_center:.4f}, {y_center:.4f}), Size: ({width:.4f}, {height:.4f})")
        print(f"Direction: ({direction:.2f}), Distance: ({distance:.2f})")


