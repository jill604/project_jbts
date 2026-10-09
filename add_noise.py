#该代码评测的数据以当前train文件导入的模型为准，测的是当前train文件导入的模型

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
# ==========================================
# 1. 核心扰动函数定义
# ==========================================
def add_gaussian_noise(image, sigma):
    """
    添加高斯噪声
    :param image: 输入图像 (numpy array)
    :param sigma: 噪声标准差 (如 10, 20, 30)
    """
    # 生成均值为0，标准差为sigma的噪声矩阵
    noise = np.random.normal(0, sigma, image.shape)
    # 叠加噪声，注意要先转成 float32 防止溢出
    noisy_img = image.astype(np.float32) + noise
    # 截断到 0-255 并转回 uint8
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def adjust_brightness(image, percentage):
    """
    调整亮度
    :param image: 输入图像 (numpy array)
    :param percentage: 亮度变化百分比 (如 20 表示 +20%, -40 表示 -40%)
    """
    factor = 1.0 + (percentage / 100.0)
    # 乘以亮度因子
    bright_img = image.astype(np.float32) * factor
    return np.clip(bright_img, 0, 255).astype(np.uint8)


# ==========================================
# 2. 小批量测试功能 (抽取 3 张图看效果)
# ==========================================
def run_smoke_test(source_dir):
    print("=" * 50)
    print("🚀 开始鲁棒性扰动脚本的冒烟测试...")
    
    # 找几张测试集里的 A 图作为样本
    sample_imgs = glob.glob(os.path.join(source_dir, "*.png"))[:3]
    if not sample_imgs:
        sample_imgs = glob.glob(os.path.join(source_dir, "*.jpg"))[:3]
        
    if not sample_imgs:
        print(f"❌ 在 {source_dir} 下没有找到图片，请检查路径！")
        return

    output_test_dir = "./robustness_samples"
    os.makedirs(output_test_dir, exist_ok=True)

    for img_path in sample_imgs:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        print(f"正在处理图片: {img_name}...")
        
        # 1. 保存原图对比
        cv2.imwrite(os.path.join(output_test_dir, f"original_{img_name}"), img)

        # 2. 生成高斯噪声组
        for sigma in [10, 20, 30]:
            noisy = add_gaussian_noise(img, sigma)
            cv2.imwrite(os.path.join(output_test_dir, f"noise_sigma{sigma}_{img_name}"), noisy)

        # 3. 生成亮度扰动组
        for pct in [-40, -20, 20, 40]:
            bright = adjust_brightness(img, pct)
            # 注意负号在文件名里不太好看，做个转换
            prefix = f"bright_plus{pct}" if pct > 0 else f"bright_minus{abs(pct)}"
            cv2.imwrite(os.path.join(output_test_dir, f"{prefix}_{img_name}"), bright)

    print("=" * 50)
    print(f"✅ 测试完成！请打开文件夹 '{output_test_dir}' 查看扰动效果！")
    print("=" * 50)


# ==========================================
# 3. 全量生成功能 (后期做主实验时使用)
# ==========================================
def generate_full_robustness_set(source_a_dir, source_b_dir, output_root):
    """
    这个函数用来在后期一键生成全部测试集的扰动版本。
    (目前你只需要跑上面的 run_smoke_test 即可，这个函数备用)
    """
    conditions = {
        'noise_10': lambda img: add_gaussian_noise(img, 10),
        'noise_20': lambda img: add_gaussian_noise(img, 20),
        'noise_30': lambda img: add_gaussian_noise(img, 30),
        'bright_m40': lambda img: adjust_brightness(img, -40),
        'bright_m20': lambda img: adjust_brightness(img, -20),
        'bright_p20': lambda img: adjust_brightness(img, 20),
        'bright_p40': lambda img: adjust_brightness(img, 40),
    }
    
    # 逻辑：遍历每个条件，生成对应的 A 和 B 测试集目录...
    # (此逻辑留给后续完整实验调用)
    pass


if __name__ == "__main__":
    # 【请修改为你的实际测试集图片路径】
    # 例如 MineNetCD/test/A
    SOURCE_TEST_DIR = "/home/mvai/Documents/zyn/MineNetCD/test/A" 
    
    run_smoke_test(SOURCE_TEST_DIR)