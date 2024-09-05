from ultralytics import YOLO

model = YOLO('models/best.pt')

model.predict('cctvwomanfire.mp4', conf = 0.2, save=True)