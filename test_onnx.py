import cv2
import numpy as np
import onnxruntime as ort

# 加载 ONNX 模型
session = ort.InferenceSession("./models/best.onnx")

# 读取测试图片
img = cv2.imread("./sample.jpg")
orig_h, orig_w = img.shape[:2]

# 预处理：YOLOv8 默认输入 640x640
# HWC -> CHW -> 归一化
input_img = cv2.resize(img, (640, 640))
input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
input_img = input_img.astype(np.float32) / 255.0
input_img = np.expand_dims(input_img, axis=0)  # batch=1

# 推理
outputs = session.run(None, {'images': input_img})  # 注意输入名可能是 'images' 或 'input'

# outputs[0] shape: [1, 5, 8400] —— YOLOv8 单类的输出格式
print("ONNX 推理成功！输出形状:", outputs[0].shape)