"""Processamento por talhão com leitura em janelas.

Este módulo é o que torna o CanaScan ilimitado em área: em vez de carregar
o ortomosaico inteiro na memória, lê do disco apenas o retângulo que cobre
cada talhão (mais uma folga), processa, descarta e passa ao próximo.
A memória usada é a de UM talhão por vez, independente do tamanho total.
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, intersection

COLUNAS_NOME = ["talhao", "Talhao", "TALHAO", "nome", "Nome", "name",
                "Name", "id", "ID", "fid"]


def carregar_talhoes(caminho, crs_destino=None):
    """Lê polígonos de talhões (GPKG/SHP/GeoJSON) e padroniza a coluna 'talhao'."""
    gdf = gpd.read_file(caminho)
    if crs_destino is not None and gdf.crs is not None and gdf.crs != crs_destino:
        gdf = gdf.to_crs(crs_destino)
    col = next((c for c in COLUNAS_NOME if c in gdf.columns), None)
    nomes = (gdf[col].astype(str).tolist() if col
             else [str(i + 1) for i in range(len(gdf))])
    saida = gdf[["geometry"]].copy()
    saida["talhao"] = nomes
    return saida


def info_raster(caminho):
    """Metadados leves do raster, sem carregar pixels."""
    with rasterio.open(caminho) as src:
        gsd = (abs(src.transform.a) + abs(src.transform.e)) / 2.0
        gb = src.width * src.height * 3 / 1e9
        return {"largura": src.width, "altura": src.height, "crs": src.crs,
                "transform": src.transform, "gsd": gsd, "gb_abertos": gb}


def ler_janela(caminho, poligono, pad_m=6.0):
    """Lê apenas a janela do raster que cobre o polígono (+folga).

    Retorna (rgb, valido, transform_da_janela, crs, gsd) ou None se a
    janela cair fora do raster. 'valido' já vem restrito ao polígono.
    """
    with rasterio.open(caminho) as src:
        minx, miny, maxx, maxy = poligono.bounds
        jan = from_bounds(minx - pad_m, miny - pad_m,
                          maxx + pad_m, maxy + pad_m, src.transform)
        jan = jan.round_offsets().round_lengths()
        try:
            jan = intersection(jan, Window(0, 0, src.width, src.height))
        except rasterio.errors.WindowError:
            return None
        if jan.width < 4 or jan.height < 4:
            return None

        arr = src.read(window=jan)
        tr = src.window_transform(jan)
        gsd = (abs(tr.a) + abs(tr.e)) / 2.0
        H, W = arr.shape[1], arr.shape[2]

        if arr.shape[0] >= 4:
            valido = arr[3] > 0
        else:
            valido = np.ones((H, W), dtype=bool)
            if src.nodata is not None:
                valido &= ~np.all(arr[:3] == src.nodata, axis=0)

        dentro = geometry_mask([poligono], out_shape=(H, W),
                               transform=tr, invert=True)
        valido &= dentro
        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
        return rgb, valido, tr, src.crs, gsd
