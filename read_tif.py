import rasterio
import numpy as np

# 打开文件，查看波段信息
with rasterio.open('Clip_XuanWuLake_20180810_rhow.tif') as src:
    print(f"波段数量: {src.count}")
    print(f"尺寸: {src.width} x {src.height}")
    print(f"坐标系: {src.crs}")

    # 查看每个波段的名称和范围
    for i in range(1, src.count + 1):
        band_data = src.read(i)
        print(f"波段{i}: 最小值={band_data.min():.4f}, 最大值={band_data.max():.4f}, 均值={band_data.mean():.4f}")