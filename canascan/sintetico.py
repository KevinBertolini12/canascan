"""Gera um ortomosaico sintético de canavial COM GABARITO.

Simula: linhas de plantio com espaçamento configurável, orientação qualquer,
curvatura suave (curva de nível), touceiras verdes com variação de cor,
falhas de plantio em posições conhecidas, solo texturizado e mato esparso.

Saídas: campo.tif (GeoTIFF UTM), verdade.gpkg (linhas + falhas reais),
preview.png e resumo.json.
"""
import json
import os

import cv2
import numpy as np
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString
import rasterio


def _textura_solo(H, W, rng):
    base = np.array([152, 96, 68], dtype=np.float32)  # latossolo vermelho
    img = np.ones((H, W, 3), dtype=np.float32) * base
    # variação de baixa frequência
    low = rng.normal(0, 1, (H // 40 + 2, W // 40 + 2)).astype(np.float32)
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    img += low[..., None] * 14
    # granulado fino
    img += rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    return np.clip(img, 0, 255)


def gerar(saida, largura_m=120.0, altura_m=80.0, gsd=0.05, espacamento=1.5,
          angulo_graus=18.0, curvatura_amp_m=0.8, curvatura_lambda_m=70.0,
          pct_falha_alvo=0.06, margem_m=4.0, seed=42):
    os.makedirs(saida, exist_ok=True)
    rng = np.random.default_rng(seed)

    W, H = int(largura_m / gsd), int(altura_m / gsd)
    img = _textura_solo(H, W, rng)

    # georreferenciamento: SIRGAS 2000 / UTM 23S, canto plausível no interior de SP
    x0, y0 = 795000.0, 7660000.0
    transform = from_origin(x0, y0, gsd, gsd)
    crs = CRS.from_epsg(31983)

    def m_para_px(pm):
        return pm / gsd

    def px_para_geo(p_px):
        return (x0 + p_px[0] * gsd, y0 - p_px[1] * gsd)

    # geometria das linhas: direção d, normal n, curvatura f(t) compartilhada
    th = np.deg2rad(angulo_graus)
    d = np.array([np.sin(th), np.cos(th)])   # direção ao longo da linha (m)
    n = np.array([np.cos(th), -np.sin(th)])  # normal (m)
    centro = np.array([largura_m / 2.0, altura_m / 2.0])
    fase = rng.uniform(0, 2 * np.pi)

    def ponto(k, t):
        """posição (m, coords de imagem: y p/ baixo) da linha k no parâmetro t"""
        desloc = curvatura_amp_m * np.sin(2 * np.pi * t / curvatura_lambda_m + fase)
        return centro + k * espacamento * n + (t + 0.0) * d + desloc * n

    diag = np.hypot(largura_m, altura_m)
    ks = range(-int(diag / espacamento / 2) - 2, int(diag / espacamento / 2) + 3)
    dentro = lambda p: (margem_m <= p[0] <= largura_m - margem_m and
                        margem_m <= p[1] <= altura_m - margem_m)

    # probabilidade de iniciar falha por metro varrido, calibrada p/ alvo
    comp_medio_falha = 1.6
    p_falha_por_m = pct_falha_alvo / comp_medio_falha

    linhas_gt, falhas_gt = [], []
    passo_planta = 0.30  # touceira a cada ~30 cm
    id_linha = 0

    for k in ks:
        ts = np.arange(-diag / 2, diag / 2, 0.25)
        ins = [t for t in ts if dentro(ponto(k, t))]
        if len(ins) < 8:
            continue
        t_min, t_max = ins[0], ins[-1]
        if (t_max - t_min) < 6:
            continue
        id_linha += 1

        # sorteia intervalos de falha dentro da linha
        gaps = []
        t = t_min + 0.8
        while t < t_max - 0.8:
            if rng.random() < p_falha_por_m * 0.25:  # passo de varredura 0,25 m
                comp = float(np.clip(rng.lognormal(0.15, 0.75), 0.5, 6.0))
                fim = min(t + comp, t_max - 0.5)
                gaps.append((t, fim))
                t = fim + 0.6
            else:
                t += 0.25

        def em_gap(t):
            return any(a <= t <= b for a, b in gaps)

        # planta as touceiras
        t = t_min
        while t <= t_max:
            if not em_gap(t) and rng.random() > 0.03:  # 3% micro-buracos naturais
                p = ponto(k, t) + rng.normal(0, 0.04, 2)
                cx, cy = m_para_px(p[0]), m_para_px(p[1])
                cor_base = np.array([70, 150, 55]) + rng.normal(0, 12, 3)
                for _ in range(rng.integers(3, 6)):  # tufos de folhas
                    ang = rng.uniform(0, 180)
                    rx = int(m_para_px(rng.uniform(0.18, 0.34)))
                    ry = int(m_para_px(rng.uniform(0.08, 0.16)))
                    ox, oy = rng.normal(0, m_para_px(0.07), 2)
                    cor = tuple(np.clip(cor_base + rng.normal(0, 10, 3),
                                        20, 230).tolist())
                    cv2.ellipse(img, (int(cx + ox), int(cy + oy)), (rx, ry),
                                ang, 0, 360, cor, -1)
            t += passo_planta * rng.uniform(0.85, 1.15)

        # gabarito
        ts_gt = np.append(np.arange(t_min, t_max, 1.0), t_max)
        pts_geo = [px_para_geo((m_para_px(ponto(k, t)[0]),
                                m_para_px(ponto(k, t)[1]))) for t in ts_gt]
        linhas_gt.append({"linha_id": id_linha,
                          "comprimento_m": round(t_max - t_min, 2),
                          "geometry": LineString(pts_geo)})
        for a, b in gaps:
            tt = np.append(np.arange(a, b, 0.5), b)
            pg = [px_para_geo((m_para_px(ponto(k, t)[0]),
                               m_para_px(ponto(k, t)[1]))) for t in tt]
            if len(pg) < 2:
                pg = [px_para_geo((m_para_px(ponto(k, a)[0]), m_para_px(ponto(k, a)[1]))),
                      px_para_geo((m_para_px(ponto(k, b)[0]), m_para_px(ponto(k, b)[1])))]
            falhas_gt.append({"linha_id": id_linha,
                              "comprimento_m": round(b - a, 2),
                              "geometry": LineString(pg)})

    # mato esparso (inclusive dentro de falhas, p/ testar robustez)
    for _ in range(int(largura_m * altura_m * 0.010)):
        px = rng.uniform(m_para_px(margem_m), W - m_para_px(margem_m))
        py = rng.uniform(m_para_px(margem_m), H - m_para_px(margem_m))
        r = int(m_para_px(rng.uniform(0.05, 0.16)))
        cor = tuple(np.clip(np.array([85, 160, 70]) + rng.normal(0, 15, 3),
                            20, 230).tolist())
        cv2.circle(img, (int(px), int(py)), r, cor, -1)

    img8 = img.astype(np.uint8)

    # salva GeoTIFF
    tif = os.path.join(saida, "campo.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=3,
                       dtype="uint8", crs=crs, transform=transform,
                       compress="deflate") as dst:
        for b in range(3):
            dst.write(img8[..., b], b + 1)

    # gabarito
    gpkg = os.path.join(saida, "verdade.gpkg")
    gpd.GeoDataFrame(linhas_gt, crs=crs).to_file(gpkg, layer="linhas", driver="GPKG")
    gpd.GeoDataFrame(falhas_gt, crs=crs).to_file(gpkg, layer="falhas", driver="GPKG")

    # talhões: divide o campo em 3 polígonos p/ testar o modo por talhão
    from shapely.geometry import Polygon
    m = margem_m
    cortes = [m, largura_m / 3.0, 2 * largura_m / 3.0, largura_m - m]
    talhoes = []
    for i in range(3):
        xa, xb = cortes[i], cortes[i + 1]
        talhoes.append({"talhao": str(i + 1), "geometry": Polygon([
            (x0 + xa, y0 - m), (x0 + xb, y0 - m),
            (x0 + xb, y0 - (altura_m - m)), (x0 + xa, y0 - (altura_m - m))])})
    gpkg_t = os.path.join(saida, "talhoes.gpkg")
    gpd.GeoDataFrame(talhoes, crs=crs).to_file(gpkg_t, layer="talhoes", driver="GPKG")

    cv2.imwrite(os.path.join(saida, "preview.png"),
                cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))

    total_linha = sum(l["comprimento_m"] for l in linhas_gt)
    total_falha = sum(f["comprimento_m"] for f in falhas_gt)
    resumo = {"n_linhas": len(linhas_gt), "comprimento_total_m": round(total_linha, 1),
              "n_falhas": len(falhas_gt), "falha_total_m": round(total_falha, 1),
              "pct_falha_real": round(100 * total_falha / total_linha, 2),
              "gsd": gsd, "espacamento": espacamento, "angulo_graus": angulo_graus}
    with open(os.path.join(saida, "resumo.json"), "w") as f:
        json.dump(resumo, f, indent=2)
    return resumo, tif, gpkg
