import cv2
import os
import re

def sanitize_filename(name):
    """将文件名转为纯英文+数字+下划线格式，移除所有非安全字符"""
    # 保留字母、数字、点，其余替换为下划线
    name = re.sub(r'[^a-zA-Z0-9._]', '_', name)
    # 合并多个下划线为一个
    name = re.sub(r'_+', '_', name)
    # 去掉首尾下划线
    name = name.strip('_')
    return name

def extract_frames(video_path, interval=120, output_dir="yolo_dataset/images"):
    """
    从单个视频中每隔 `interval` 帧抽取一帧，保存为纯英文/数字命名的图片。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在！\n路径: {video_path}")
        return

    # 生成安全的前缀：从路径中提取文件名（不含扩展名），转为英文数字
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_prefix = sanitize_filename(base_name)
    
    # 如果清理后前缀为空（极罕见），用默认名
    if not safe_prefix:
        safe_prefix = "video"

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件，请检查路径或编解码器。")
        return

    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    print(f"将每 {interval} 帧保存一张（约每 {interval/fps:.1f} 秒）")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            filename = f"{safe_prefix}_frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            try:
                cv2.imwrite(filepath, frame)
                saved_count += 1
                print(f"已保存: {filename}")  # 每张都输出
            except Exception as e:
                print(f"保存失败 {filename}: {e}")

        frame_count += 1

    cap.release()
    print("-" * 50)
    print(f"完成！共处理 {frame_count} 帧，保存 {saved_count} 张图片到:\n {os.path.abspath(output_dir)}")

# ===== 使用方式 =====
if __name__ == "__main__":
    # 👇 每次只需修改这一行！
    VIDEO_PATH = r""
    
    # 抽帧间隔：120帧（可按需调整）
    extract_frames(VIDEO_PATH, interval=120)