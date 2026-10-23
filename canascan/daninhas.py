"""Detecção de daninhas na entrelinha.

Princípio: o CanaScan já sabe onde estão as linhas de plantio. Vegetação
verde FORA da faixa das linhas = mato na entrelinha. Não identifica a
espécie (isso virá com o modelo neural), mas quantifica e mapeia a
infestação — decisão de manejo/herbicida localizada.
"""
import cv2
import numpy as np
from rasterio import features as rfeatures
from shapely.geometry import shape


def detectar(mask, linhas_px, gsd, transform, largura_linha=0.60,
             area_min_m2=0.05, simplificar_m=0.10):
    """Retorna dict com área (m²), % da área válida e polígonos geo.

    largura_linha: faixa (m) em torno de cada linha considerada "cultura".
    area_min_m2: descarta manchinhas menores que isto.
    """
    H, W = mask.shape
    faixa = np.zeros((H, W), dtype=np.uint8)
    esp_px = max(1, int(round(largura_linha / gsd)))
    for pts in linhas_px:
        p = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(faixa, [p], False, 1, thickness=esp_px)

    daninha = mask.astype(np.uint8) & (1 - faixa)
    # limpeza: abre para remover ruído de pixel
    k = np.ones((3, 3), np.uint8)
    daninha = cv2.morphologyEx(daninha, cv2.MORPH_OPEN, k)

    area_px_min = max(1, int(area_min_m2 / (gsd * gsd)))
    n, rot, stats, _ = cv2.connectedComponentsWithStats(daninha)
    manter = np.zeros(n, dtype=bool)
    manter[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_px_min
    daninha = manter[rot].astype(np.uint8)

    area_m2 = float(daninha.sum()) * gsd * gsd
    total_valido = float(mask.size)
    pct = 100.0 * daninha.sum() / total_valido if total_valido else 0.0

    poligonos = []
    if daninha.any():
        for geom, val in rfeatures.shapes(daninha, mask=daninha.astype(bool),
                                          transform=transform):
            if val == 1:
                g = shape(geom).simplify(simplificar_m)
                if not g.is_empty:
                    poligonos.append(g)
    return {"area_m2": area_m2, "pct": pct, "poligonos": poligonos,
            "n_manchas": len(poligonos)}
