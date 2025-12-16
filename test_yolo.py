from ultralytics import YOLO

# 加载训练好的 YOLO 模型
yolo = YOLO('./models/best.pt', task="detect")

# 在测试图片上进行检测并保存结果
result = yolo(source="./sample.jpg", save=True)