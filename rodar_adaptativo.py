# -*- coding: utf-8 -*-
"""Roda o motor ADAPTATIVO no ortomosaico real, talhão por talhão, e:
  - exporta linhas_adaptativas.gpkg  (abre no QGIS sobre o mapa deles)
  - salva um preview PNG por talhão (adaptativo vs atual) pra conferir
Coloque este arquivo na raiz da pasta canascan_v0.7.9 e rode:  python rodar_adaptativo.py
"""
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, '.')

from canascan import geo, vegetacao, talhoes as T
from canascan import linhas as ml
from canascan import adaptativo as ad

# ================== EDITE AQUI ==================
TIF     = r'C:\Users\Kevin\Downloads\3214 - Ubaldo_transparent_mosaic_group1.tif'
TALHOES = r'C:\Users\Kevin\Downloads\baldo.gpkg'
ESP     = 1.48            # espaçamento (m)
METODO  = 'gli'
LIMIAR  = 0.02
TILE_M  = 35.0           # tamanho do tile local (m). Menor = segue curva mais fechada, mais lento
SO_TALHAO = None         # ex.: '3' pra rodar só um talhão; None = todos
SAIDA   = 'resultados_adaptativo'
# ===============================================

os.makedirs(SAIDA, exist_ok=True)
info = T.info_raster(TIF)
crs = info['crs']
print(f"[raster] {info['largura']}x{info['altura']} px | GSD {info['gsd']*100:.1f} cm/px | CRS {crs}")
gdf = T.carregar_talhoes(TALHOES, crs)
print(f"[talhões] {len(gdf)} carregados")


def dobras(linhas, esp_px):
    """conta linhas com paralela colada a < 0.5*esp (indicador de qualidade)."""
    Ls = [np.asarray(L, float) for L in linhas]
    n = len(Ls); d = 0
    if n < 2:
        return 0
    mids = [L.mean(0) for L in Ls]
    dirs = []
    for L in Ls:
        v = L[-1] - L[0]; nn = np.linalg.norm(v)
        dirs.append(v / nn if nn > 1e-6 else np.array([1.0, 0.0]))
    for i in range(n):
        perp = np.array([-dirs[i][1], dirs[i][0]])
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(Ls[j] - mids[i], axis=1)
            jj = int(np.argmin(dist))
            if dist[jj] > 2 * esp_px:
                continue
            if abs(np.dot(Ls[j][jj] - mids[i], perp)) < 0.5 * esp_px \
                    and abs(np.dot(dirs[i], dirs[j])) > 0.9:
                d += 1; break
    return d


def preview(rgb, lin_atual, lin_adap, caminho, max_w=1800):
    H, W = rgb.shape[:2]
    esc = min(1.0, max_w / W)
    def painel(linhas, titulo, cor):
        img = cv2.resize(rgb, (int(W * esc), int(H * esc)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for L in linhas:
            P = (np.asarray(L) * esc).astype(np.int32)
            cv2.polylines(img, [P.reshape(-1, 1, 2)], False, cor, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(img, titulo, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (20, 20, 20), 2, cv2.LINE_AA)
        return img
    a = painel(lin_atual, "ATUAL (angulo unico)", (30, 30, 220))
    b = painel(lin_adap, "ADAPTATIVO (angulo local)", (210, 120, 20))
    sep = np.full((a.shape[0], 6, 3), 200, np.uint8)
    cv2.imwrite(caminho, np.hstack([a, sep, b]))


from shapely.geometry import LineString
import geopandas as gpd

todas_geo = []
t0 = time.time()
for _, row in gdf.iterrows():
    nome = row['talhao']
    if SO_TALHAO is not None and str(nome) != str(SO_TALHAO):
        continue
    jan = T.ler_janela(TIF, row.geometry, pad_m=6.0)
    if jan is None:
        print(f"  [{nome}] fora do raster — pulado"); continue
    rgb, valido, tr, _, gsd = jan
    mask = vegetacao.mascara_vegetacao(rgb, valido, metodo=METODO, limiar=LIMIAR)
    esp_px = ESP / gsd
    t1 = time.time()
    lin_atual, _ = ml.detectar_linhas(mask, gsd, ESP, banda_m=4.0, min_comprimento_m=5.0)
    lin_adap, ang = ad.detectar_linhas_adaptativo(
        mask, gsd, ESP, banda_m=4.0, min_comprimento_m=5.0, tile_m=TILE_M)
    dt = time.time() - t1
    print(f"  [{nome}] verde {100*mask.mean():.0f}% | "
          f"ATUAL {len(lin_atual)} linhas, {dobras(lin_atual, esp_px)} coladas | "
          f"ADAPT {len(lin_adap)} linhas, {dobras(lin_adap, esp_px)} coladas | {dt:.1f}s")
    preview(rgb, lin_atual, lin_adap, os.path.join(SAIDA, f'preview_talhao_{nome}.png'))
    for L in lin_adap:
        g = geo.px_para_geo(tr, np.asarray(L))
        if len(g) >= 2:
            todas_geo.append({'talhao': str(nome), 'geometry': LineString(g)})
    del rgb, valido, mask

if todas_geo:
    gpkg = os.path.join(SAIDA, 'linhas_adaptativas.gpkg')
    gpd.GeoDataFrame(todas_geo, crs=crs).to_file(gpkg, layer='linhas', driver='GPKG')
    print(f"\n[ok] {len(todas_geo)} linhas -> {gpkg}")
    print(f"[ok] previews em {SAIDA}/preview_talhao_*.png")
print(f"tempo total {time.time()-t0:.1f}s")
