import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def check_data_quality(tif_path, mask_img_path, output_dir="output"):
    """
    数据质量检查与可视化函数

    参数:
    - tif_path: 输入TIFF影像文件路径
    - mask_img_path: 湖区掩膜图片路径（用于湖区示意图）
    - output_dir: 输出目录

    功能:
    1. 读取遥感影像数据
    2. 检查各波段数据质量（B2-B8A）
    3. 生成包含湖区示意图和7个波段的可视化质量检查图
    4. 输出统计信息和检查结果图（600 dpi高分辨率）
    """

    # 1. 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 2. 从文件名提取日期信息
    stem = Path(tif_path).stem
    parts = stem.split("_")

    # 提取日期部分（第3部分应为日期字符串）
    if len(parts) >= 3 and parts[2].isdigit():
        date_str = parts[2]  # 例如: "20180810" 或 "20250818"
    else:
        date_str = stem  # 备选方案

    # 3. 打开遥感影像文件
    with rasterio.open(tif_path) as src:
        # 读取所有波段数据
        data = src.read()  # 形状: (波段数, 高度, 宽度)
        # 4. 创建可视化图表
        # 布局: 2行×4列，总共8个子图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        # ------------------- 子图1: 湖区示意图 -------------------
        ax_mask = axes[0]
        try:
            # 加载并显示湖区掩膜图片
            from matplotlib.image import imread
            mask_img = imread(mask_img_path)
            ax_mask.imshow(mask_img)
            ax_mask.set_title("湖区示意图")
            ax_mask.axis('off')  # 关闭坐标轴
        except FileNotFoundError:
            # 如果找不到掩膜图片，显示提示信息
            ax_mask.text(0.5, 0.5, "掩膜图片未找到",
                         ha='center', va='center', fontsize=12)
            ax_mask.axis('off')

        # ------------------- 子图2-8: 7个波段数据 -------------------
        # 哨兵-2波段对应关系
        band_names = [
            'B2(490nm)',  # 蓝波段
            'B3(560nm)',  # 绿波段
            'B4(665nm)',  # 红波段
            'B5(705nm)',  # 植被红边1
            'B6(740nm)',  # 植被红边2
            'B7(783nm)',  # 植被红边3
            'B8A(865nm)'  # 近红外波段
        ]
        # 循环处理前7个波段（B2-B8A）
        for i in range(min(src.count, 7)):
            ax = axes[i + 1]  # 注意+1跳过第0个（湖区示意图）
            band_data = data[i]
            # 4.1 数据有效性检查：去除无效值
            invalid_mask = band_data < -1e30
            valid_data = band_data[~invalid_mask]
            if len(valid_data) > 0:
                # 使用2%和98%百分位数确定色标范围，排除极端值
                vmin, vmax = np.percentile(valid_data, [2, 98])
                # 显示波段图像
                im = ax.imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)
                # 添加颜色条，显示反射率值范围
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # 如果没有有效值，使用灰度显示
                ax.imshow(band_data, cmap='gray')
            # 设置子图标题和格式
            ax.set_title(band_names[i])
            ax.axis('off')
            # 4.2 输出统计信息到控制台（用于日志记录）
            if len(valid_data) > 0:
                print(f"{Path(tif_path).name} | {band_names[i]}: "
                      f"有效值{len(valid_data)}个, "
                      f"范围{valid_data.min():.4f}-{valid_data.max():.4f}")
            else:
                print(f"警告: {band_names[i]}没有有效值")

        # 5. 设置总标题
        plt.suptitle(f"{date_str} 哨兵-2数据质量检查", fontsize=14)
        plt.tight_layout()

        # 6. 保存高质量图片（600 dpi）
        out_path = Path(output_dir) / f"数据质量检查_{date_str}.png"
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        print(f"✓ 质量检查图已保存: {out_path}")

        # 7. 显示图像
        plt.show()


# ============================================
# 主程序：执行数据质量检查
# ============================================
if __name__ == "__main__":
    print("======== 哨兵-2数据质量检查程序 ========")

    # 定义输入文件
    tif_2018 = "Clip_XuanWuLake_20180810_rhow.tif"  # 2018年数据
    tif_2025 = "Clip_XuanWuLake_20250818_rhow.tif"  # 2025年数据
    mask_image_path = "湖区.png"  # 湖区边界示意图

    print("\n1. 检查2018年数据...")
    check_data_quality(tif_2018, mask_image_path, "output")

    print("\n2. 检查2025年数据...")
    check_data_quality(tif_2025, mask_image_path, "output")

    print("\n======== 数据质量检查完成 ========")