import numpy as np
import cv2

def read_flo(file_path):
    """读取 .flo 光流文件"""
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {file_path}")

        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]

        # 读取光流数据
        flow = np.fromfile(f, np.float32, count=2 * width * height)
        flow = flow.reshape((height, width, 2))  # (H, W, 2) 结构
    return flow


def calculate_average_flow(flow, mask):
    """根据二值mask计算上下和左右方向的位移平均值"""
    # 获取光流的水平和垂直分量
    horizontal_flow = flow[..., 0]  # 水平位移
    vertical_flow = flow[..., 1]    # 垂直位移

    # # 将mask图像中的黑色区域（值为0）提取出来
    # mask_black = (mask == 0)  # 黑色区域为True，其他区域为False

    # 将mask图像中的白色区域（值为255）提取出来
    mask = mask.astype(bool)  # 转为布尔类型，白色区域为True，黑色区域为False

    # # 黑色区域为True，其他区域也为True
    # mask = mask > -1
    # mask = mask.astype(bool)





    # 提取mask中白色区域对应的水平和垂直位移
    horizontal_values = horizontal_flow[mask]
    vertical_values = vertical_flow[mask]

    # 计算上下方向（垂直方向）和左右方向（水平方向）的位移平均值
    avg_horizontal = np.mean(horizontal_values) if horizontal_values.size > 0 else 0
    avg_vertical = np.mean(vertical_values) if vertical_values.size > 0 else 0

    return avg_horizontal, avg_vertical


def main(flow_file, mask_file):
    """主函数"""
    # 读取光流文件
    flow = read_flo(flow_file)

    # 读取二值mask图像（确保是灰度图像）
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # 计算位移平均值
    avg_horizontal, avg_vertical = calculate_average_flow(flow, mask)

    print(f"水平位移平均值: {avg_horizontal}")
    print(f"垂直位移平均值: {avg_vertical}")


# 使用示例
flow_file = "frame_000011.flo"  # 光流文件路径
mask_file = "frame_000011.jpg"  # 二值mask图像路径
main(flow_file, mask_file)
