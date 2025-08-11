import os
import xarray as xr
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
from global_land_mask import globe

def mask_land(da, label='land', coord_name='x'):
    if coord_name == 'x':
        lat = da.y.data
        lon = da.x.data
    else:
        lat = da.latitude.data
        lon = da.longitude.data

    lon, lat = np.meshgrid(lon, lat)
    mask = globe.is_ocean(lat, lon)

    if label == 'land':
        # 生成二值掩膜，陆地为1，海洋为0
        land_mask = np.where(mask == False, 1, 0)
        return xr.DataArray(land_mask, coords=da.coords, dims=da.dims)
    elif label == 'ocean':
        # 生成二值掩膜，海洋为1，陆地为0
        ocean_mask = np.where(mask == True, 1, 0)
        return xr.DataArray(ocean_mask, coords=da.coords, dims=da.dims)
    else:
        return da

if __name__ == '__main__':
    path = r'D:\Dataset\maweizao\DefaultImage\new_labeled.tif'
    data = rxr.open_rasterio(path)
    precip = data[0]  # 第一波段
    precip_small = precip.coarsen(x=10, y=10, boundary='trim').mean()

    # 掩膜处理，获取陆地二值掩膜
    precip_land = mask_land(precip_small, 'land')

    # 绘图
    fig = plt.figure(figsize=(15, 4), dpi=200)

    ax1 = plt.subplot(131, projection=crs.PlateCarree(central_longitude=180))
    precip_land.plot(ax=ax1, transform=crs.PlateCarree(), cmap='gray', add_colorbar=False)
    ax1.coastlines()
    plt.title('Land Mask (Binary)')

    plt.tight_layout()
    plt.show()
