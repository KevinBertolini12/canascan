"""Utilitários geoespaciais: leitura de raster e conversão pixel <-> coordenadas."""
import numpy as np
import rasterio


def carregar_rgb(caminho):
    """Carrega um GeoTIFF RGB(A).

    Retorna: rgb (H,W,3 uint8), valido (H,W bool), transform, crs, gsd (m/pixel)
    """
    with rasterio.open(caminho) as src:
        arr = src.read()
        transform = src.transform
        crs = src.crs
        gsd = (abs(transform.a) + abs(transform.e)) / 2.0

        if arr.shape[0] >= 4:  # banda alpha = pixels válidos
            valido = arr[3] > 0
        else:
            valido = np.ones(arr.shape[1:], dtype=bool)
            if src.nodata is not None:
                valido &= ~np.all(arr[:3] == src.nodata, axis=0)

        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
    return rgb, valido, transform, crs, gsd


def px_para_geo(transform, pontos_px):
    """Converte array Nx2 de pontos em pixel (x=coluna, y=linha) para coordenadas do CRS."""
    p = np.asarray(pontos_px, dtype=np.float64)
    x = transform.a * p[:, 0] + transform.b * p[:, 1] + transform.c
    y = transform.d * p[:, 0] + transform.e * p[:, 1] + transform.f
    return np.column_stack([x, y])


def geo_para_px(transform, pontos_geo):
    """Converte array Nx2 de coordenadas do CRS para pixel (x=coluna, y=linha)."""
    inv = ~transform
    p = np.asarray(pontos_geo, dtype=np.float64)
    c = inv.a * p[:, 0] + inv.b * p[:, 1] + inv.c
    r = inv.d * p[:, 0] + inv.e * p[:, 1] + inv.f
    return np.column_stack([c, r])
