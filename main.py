import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# 解决中文显示问题
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass  # 如果字体设置失败，继续使用默认字体


def safe_divide(a, b):
    """
    安全除法，支持标量和数组
    a: 被除数（标量或数组）
    b: 除数（与a形状相同的数组）
    """
    # 如果a是标量，将其扩展为与b相同形状的数组
    if np.isscalar(a):
        a = np.full_like(b, a, dtype=np.float32)

    # 初始化结果数组
    result = np.full_like(a, np.nan, dtype=np.float32)

    # 创建有效掩膜
    mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(b) > 1e-10)

    # 执行除法
    result[mask] = a[mask] / b[mask]

    return result


def calculate_water_quality(tif_path, output_dir='output'):
    """
    计算Chl-a和TSM浓度（处理无效值版本）
    """

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 提取日期用于文件名：Clip_XuanWuLake_20180810_rhow.tif -> 20180810
    stem = Path(tif_path).stem
    parts = stem.split('_')
    if len(parts) >= 3 and parts[2].isdigit():
        date_str = parts[2]
    else:
        date_str = stem

    # 1. 读取数据并处理无效值
    print(f"正在处理 {date_str} 的数据...")
    with rasterio.open(tif_path) as src:
        # 读取所有波段
        data = src.read()
        profile = src.profile.copy()

        # 定义无效值阈值
        invalid_threshold = -1e30

        # 处理每个波段
        cleaned_bands = []
        for i in range(src.count):
            band_data = data[i].astype(np.float32)

            # 识别无效值
            is_invalid = band_data < invalid_threshold

            # 将无效值替换为NaN
            band_data[is_invalid] = np.nan

            # 检查数据合理性：反射率不应大于1，明显小于0的视为无效
            band_data[band_data < -0.01] = np.nan
            band_data[band_data > 1.0] = np.nan

            cleaned_bands.append(band_data)

            # 打印处理后的统计信息
            valid_data = band_data[~np.isnan(band_data)]
            if len(valid_data) > 0:
                print(f"波段{i + 1}: 有效值{len(valid_data)}个, "
                      f"范围{valid_data.min():.4f}-{valid_data.max():.4f}")
            else:
                print(f"波段{i + 1}: 警告: 没有有效值!")

        # 分离各波段
        B2 = cleaned_bands[0]  # 490 nm
        B3 = cleaned_bands[1]  # 560 nm
        B4 = cleaned_bands[2]  # 665 nm
        B5 = cleaned_bands[3]  # 705 nm
        B6 = cleaned_bands[4]  # 740 nm
        B7 = cleaned_bands[5]  # 783 nm
        B8A = cleaned_bands[6]  # 865 nm

        # 创建有效水体掩膜（这里用 B2 和 B8A 是否为 NaN 判定）
        valid_mask = ~np.isnan(B2) & ~np.isnan(B8A)
        water_pixels = np.sum(valid_mask)
        total_pixels = B2.size

        print(f"有效水体像元: {water_pixels}/{total_pixels} ({water_pixels / total_pixels * 100:.1f}%)")

        if water_pixels == 0:
            print("警告: 没有有效水体像元!")
            return None

        # 应用掩膜
        for band in [B2, B3, B4, B5, B6, B7, B8A]:
            band[~valid_mask] = np.nan

    print("数据预处理完成，开始计算...")

    # 2. 计算TSM（总悬浮物）
    print("计算TSM...")

    # 计算各项
    term1 = 3786.21606 * B8A
    term2 = 167.9372 * safe_divide(B6, B5)
    term3 = 3.5764 * safe_divide(B3, B8A)

    TSM = term1 + term2 + term3 - 90.84456

    # 设置非水体为 NaN
    TSM[~valid_mask] = np.nan

    # 去掉负值并限制合理范围（0–200 mg/L）
    TSM[TSM < 0] = np.nan
    TSM = np.clip(TSM, 0, 200)

    # 3. 计算Chl-a（叶绿素a）
    print("计算Chl-a...")

    # 创建TSM阈值掩膜
    TSM_valid = TSM[valid_mask]
    TSM_low_mask_full = np.full_like(TSM, False, dtype=bool)
    TSM_high_mask_full = np.full_like(TSM, False, dtype=bool)

    TSM_low_mask_full[valid_mask] = TSM_valid < 20
    TSM_high_mask_full[valid_mask] = TSM_valid >= 20

    print(f"  TSM<20区域: {np.sum(TSM_low_mask_full)} 像元")
    print(f"  TSM>=20区域: {np.sum(TSM_high_mask_full)} 像元")

    # 初始化Chl-a数组
    Chl_a = np.full_like(TSM, np.nan, dtype=np.float32)

    # 3.1 TSM < 20 mg/L 的区域（经验组合模型）
    if np.any(TSM_low_mask_full):
        # 计算公式中的分母
        denominator_low = B2 - B8A
        denominator_low = np.where(np.abs(denominator_low) < 1e-10, np.nan, denominator_low)

        # 计算经验组合项
        exp_component = safe_divide(B3 - B8A, denominator_low)

        # 计算整个图像
        chla_low_full = (75.832 * exp_component +
                         3.0664 * B7 +
                         1.2533 * (B5 - B4) -
                         105.9726)

        # 仅应用到TSM<20的区域
        Chl_a[TSM_low_mask_full] = chla_low_full[TSM_low_mask_full]

    # 3.2 TSM >= 20 mg/L 的区域（三波段模型）
    if np.any(TSM_high_mask_full):
        # 计算三波段值
        inv_B4 = safe_divide(1.0, B4)
        inv_B5 = safe_divide(1.0, B5)
        three_band_value = (inv_B4 - inv_B5) * B8A

        # 计算NDCI（归一化叶绿素指数）
        ndci_denominator = B5 + B4
        ndci = safe_divide(B5 - B4, ndci_denominator)

        # 计算B5/B4比值
        ratio_B5_B4 = safe_divide(B5, B4)

        # 多元回归模型
        chla_high_full = (-115.117 * ratio_B5_B4 -
                          50.6833 * ndci +
                          854.6138 * three_band_value +
                          140.44)

        # 仅应用到TSM>=20的区域
        Chl_a[TSM_high_mask_full] = chla_high_full[TSM_high_mask_full]

    # 非水体区域为 NaN
    Chl_a[~valid_mask] = np.nan

    # 去掉负值并限制合理范围（0–100 μg/L）
    Chl_a[Chl_a < 0] = np.nan
    Chl_a = np.clip(Chl_a, 0, 100)

    # 统计有效结果
    chla_valid = Chl_a[~np.isnan(Chl_a)]
    tsm_valid = TSM[~np.isnan(TSM)]

    if len(chla_valid) > 0:
        print(f"Chl-a: {len(chla_valid)}有效值, "
              f"范围{chla_valid.min():.1f}-{chla_valid.max():.1f} μg/L, "
              f"均值{chla_valid.mean():.1f} μg/L")
    else:
        print("警告: Chl-a没有有效计算结果!")

    if len(tsm_valid) > 0:
        print(f"TSM: {len(tsm_valid)}有效值, "
              f"范围{tsm_valid.min():.1f}-{tsm_valid.max():.1f} mg/L, "
              f"均值{tsm_valid.mean():.1f} mg/L")
    else:
        print("警告: TSM没有有效计算结果!")

    # 4. 保存结果
    print("保存结果...")

    # 使用固定 nodata 值，方便 GIS 识别
    nodata_value = -9999.0
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata_value)

    # 保存TSM
    tsm_path = f"{output_dir}/TSM_{date_str}.tif"
    with rasterio.open(tsm_path, 'w', **profile) as dst:
        TSM_out = TSM.copy()
        TSM_out[np.isnan(TSM_out)] = nodata_value
        dst.write(TSM_out.astype(np.float32), 1)

    # 保存Chl-a
    chla_path = f"{output_dir}/Chl_a_{date_str}.tif"
    with rasterio.open(chla_path, 'w', **profile) as dst:
        Chl_out = Chl_a.copy()
        Chl_out[np.isnan(Chl_out)] = nodata_value
        dst.write(Chl_out.astype(np.float32), 1)

    print(f"TSM已保存: {tsm_path}")
    print(f"Chl-a已保存: {chla_path}")

    return {
        'TSM': TSM,
        'Chl_a': Chl_a,
        'date': date_str,
        'valid_mask': valid_mask,
        'profile': profile
    }


def create_comparison_plot(data_2018, data_2025, output_dir='output'):
    """创建对比图表，并输出 600 dpi PDF"""

    if data_2018 is None or data_2025 is None:
        print("无法创建图表：数据为空")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Chl-a对比
    ax1 = axes[0, 0]
    im1 = ax1.imshow(data_2018['Chl_a'], cmap='viridis', vmin=0, vmax=50)
    ax1.set_title(f"Chl-a {data_2018['date']}")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='μg/L')

    ax2 = axes[0, 1]
    im2 = ax2.imshow(data_2025['Chl_a'], cmap='viridis', vmin=0, vmax=50)
    ax2.set_title(f"Chl-a {data_2025['date']}")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='μg/L')

    # 2. TSM对比（统一 vmin/vmax）
    ax3 = axes[1, 0]
    im3 = ax3.imshow(data_2018['TSM'], cmap='plasma', vmin=0, vmax=160)
    ax3.set_title(f"TSM {data_2018['date']}")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='mg/L')

    ax4 = axes[1, 1]
    im4 = ax4.imshow(data_2025['TSM'], cmap='plasma', vmin=0, vmax=160)
    ax4.set_title(f"TSM {data_2025['date']}")
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, label='mg/L')

    # 3. Chl-a 直方图
    ax5 = axes[0, 2]
    chla_2018_valid = data_2018['Chl_a'][~np.isnan(data_2018['Chl_a'])]
    chla_2025_valid = data_2025['Chl_a'][~np.isnan(data_2025['Chl_a'])]

    if len(chla_2018_valid) > 0 and len(chla_2025_valid) > 0:
        ax5.hist(chla_2018_valid, bins=30, alpha=0.5, label=data_2018['date'], density=True)
        ax5.hist(chla_2025_valid, bins=30, alpha=0.5, label=data_2025['date'], density=True)
        ax5.set_title("Chl-a浓度分布")
        ax5.set_xlabel("浓度 (μg/L)")
        ax5.set_ylabel("频率")
        ax5.legend()

    # 4. 箱线图
    ax6 = axes[1, 2]
    tsm_2018_valid = data_2018['TSM'][~np.isnan(data_2018['TSM'])]
    tsm_2025_valid = data_2025['TSM'][~np.isnan(data_2025['TSM'])]

    if len(tsm_2018_valid) > 0 and len(tsm_2025_valid) > 0:
        plot_data = []
        labels = []

        if len(chla_2018_valid) > 0:
            plot_data.append(chla_2018_valid)
            labels.append(f"Chl-a\n{data_2018['date']}")

        if len(chla_2025_valid) > 0:
            plot_data.append(chla_2025_valid)
            labels.append(f"Chl-a\n{data_2025['date']}")

        if len(tsm_2018_valid) > 0:
            plot_data.append(tsm_2018_valid)
            labels.append(f"TSM\n{data_2018['date']}")

        if len(tsm_2025_valid) > 0:
            plot_data.append(tsm_2025_valid)
            labels.append(f"TSM\n{data_2025['date']}")

        bp = ax6.boxplot(plot_data, tick_labels=labels, patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
            patch.set_facecolor(color)

        ax6.set_title("浓度箱线图")
        ax6.set_ylabel("浓度")
        ax6.grid(True, alpha=0.3)

    plt.suptitle(
        f"玄武湖水质遥感反演结果对比\n{data_2018['date']} vs {data_2025['date']}",
        fontsize=14
    )
    plt.tight_layout()

    # 保存图表为 600 dpi PDF
    Path(output_dir).mkdir(exist_ok=True)
    pdf_path = Path(output_dir) / f"玄武湖_水质对比_{data_2018['date']}_vs_{data_2025['date']}.pdf"
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    print(f"图表已保存: {pdf_path}")

    plt.show()


def main():
    """主程序"""

    # 设置文件路径
    tif_2018 = "Clip_XuanWuLake_20180810_rhow.tif"
    tif_2025 = "Clip_XuanWuLake_20250818_rhow.tif"

    print("===== 玄武湖水质遥感反演分析 =====")

    # 检查文件是否存在
    for file in [tif_2018, tif_2025]:
        if not Path(file).exists():
            print(f"错误: 文件不存在 - {file}")
            print("请确保文件在当前目录，或修改文件路径")
            return

    # 计算2018年数据
    print("\n===== 处理2018年数据 =====")
    data_2018 = calculate_water_quality(tif_2018)

    # 计算2025年数据
    print("\n===== 处理2025年数据 =====")
    data_2025 = calculate_water_quality(tif_2025)

    if data_2018 is None or data_2025 is None:
        print("\n计算失败，请检查数据!")
        return

    # 打印统计信息
    print("\n===== 统计对比 =====")

    for param_name, param_unit in [('Chl_a', 'μg/L'), ('TSM', 'mg/L')]:
        print(f"\n{param_name}浓度统计:")

        data_2018_valid = data_2018[param_name][~np.isnan(data_2018[param_name])]
        data_2025_valid = data_2025[param_name][~np.isnan(data_2025[param_name])]

        if len(data_2018_valid) > 0:
            mean_2018 = np.mean(data_2018_valid)
            std_2018 = np.std(data_2018_valid)
            print(f"  {data_2018['date']}: {mean_2018:.1f} ± {std_2018:.1f} {param_unit}")

        if len(data_2025_valid) > 0:
            mean_2025 = np.mean(data_2025_valid)
            std_2025 = np.std(data_2025_valid)
            print(f"  {data_2025['date']}: {mean_2025:.1f} ± {std_2025:.1f} {param_unit}")

        if len(data_2018_valid) > 0 and len(data_2025_valid) > 0:
            change_rate = ((mean_2025 - mean_2018) / mean_2018) * 100
            print(f"  变化率: {change_rate:.1f}%")

            if change_rate > 10:
                print(f"  注意: {param_name}浓度显著升高!")
            elif change_rate < -10:
                print(f"  注意: {param_name}浓度显著降低!")

    # 创建对比图表
    print("\n===== 创建可视化图表 =====")
    create_comparison_plot(data_2018, data_2025)

    print("\n" + "=" * 50)
    print("处理完成!")
    print("结果文件和图表保存在 'output' 目录中")
    print("=" * 50)


if __name__ == "__main__":
    main()