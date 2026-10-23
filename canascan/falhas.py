"""Detecção de falhas de plantio ao longo das linhas restituídas.

Critério: metodologia Stolf combinada com a inteligência da U-Net.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import LineString
from shapely.ops import substring

from .geo import px_para_geo


def _odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def analisar_falhas(mask, linhas_px, gsd, transform, espacamento=1.5,
                    falha_min=0.5, limiar_presenca=0.10, tolerancia_lateral=0.30,
                    janela_m=0.30, compensacao_copa_m=0.25,
                    fechar_m=0.10, abrir_m=0.25, mask_falhas_ia=None):
    """Para cada linha, mede comprimento útil e localiza falhas.

    Retorna lista de dicts:
      {id, geom (LineString geo aparada), comprimento_m, falha_m, pct_falha,
       falhas: [ {comprimento_m, geom} ]}
    """
    k = _odd(max(3, janela_m / gsd))
    dens = cv2.boxFilter(mask.astype(np.float32), -1, (k, k))
    Himg, Wimg = mask.shape

    # Se a IA gerou um mapa de falhas dedicado (Classe 2), usamos ele como guia prioritário
    if mask_falhas_ia is not None:
        dens_falhas = cv2.boxFilter(mask_falhas_ia.astype(np.float32), -1, (k, k))
    else:
        dens_falhas = None

    passo_px = max(2.0, 0.10 / gsd)  # amostra a cada ~10 cm
    resultados = []

    for i, pts in enumerate(linhas_px):
        ls_px = LineString(pts)
        if ls_px.length < 2:
            continue
        dists_px = np.arange(0, ls_px.length, passo_px)
        try:  
            import shapely as _sh
            pontos = _sh.line_interpolate_point(ls_px, dists_px)
            amostras = _sh.get_coordinates(pontos)
        except Exception:
            amostras = np.array([ls_px.interpolate(float(d)).coords[0]
                                 for d in dists_px])
            
        tang = np.gradient(amostras, axis=0)
        nrm = np.linalg.norm(tang, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        tang = tang / nrm
        normal = np.column_stack([-tang[:, 1], tang[:, 0]])
        
        presenca = np.zeros(len(amostras), dtype=bool)
        falha_ia_amostra = np.zeros(len(amostras), dtype=bool)
        
        for off_m in np.linspace(-tolerancia_lateral, tolerancia_lateral, 5):
            p = amostras + (off_m / gsd) * normal
            xi = np.clip(p[:, 0].round().astype(int), 0, Wimg - 1)
            yi = np.clip(p[:, 1].round().astype(int), 0, Himg - 1)
            presenca |= dens[yi, xi] > limiar_presenca
            if dens_falhas is not None:
                falha_ia_amostra |= dens_falhas[yi, xi] > 0.15

        passo_m = passo_px * gsd
        n_fechar = max(1, int(round(fechar_m / passo_m)))
        n_abrir = max(1, int(round(abrir_m / passo_m)))
        
        presenca = binary_closing(presenca, structure=np.ones(n_fechar))
        presenca = binary_opening(presenca, structure=np.ones(n_abrir))

        idx = np.where(presenca)[0]
        if len(idx) < 2:
            continue
        i0, i1 = idx[0], idx[-1]  

        geo_pts = px_para_geo(transform, pts)
        ls_geo = LineString(geo_pts)
        d0_m, d1_m = dists_px[i0] * gsd, dists_px[i1] * gsd
        geom_linha = substring(ls_geo, d0_m, d1_m)
        comprimento_m = d1_m - d0_m

        falhas = []
        falha_total = 0.0
        em_falha = False
        ini = 0
        
        for j in range(i0, i1 + 1):
            # Condição de falha: ou o pixel perdeu vegetação pela matemática, 
            # ou a própria IA cravou que ali é a Classe 2 (falha)
            sem_veg_matematica = not presenca[j]
            marcado_como_falha_ia = falha_ia_amostra[j] if dens_falhas is not None else sem_veg_matematica
            
            condicao_falha = sem_veg_matematica or marcado_como_falha_ia

            if condicao_falha and not em_falha:
                em_falha, ini = True, j
            elif not condicao_falha and em_falha:
                em_falha = False
                bruto = (dists_px[j] - dists_px[ini]) * gsd
                comp = bruto + 2 * compensacao_copa_m
                if comp >= falha_min:
                    da = max(dists_px[i0] * gsd,
                             dists_px[ini] * gsd - compensacao_copa_m)
                    db = min(dists_px[i1] * gsd,
                             dists_px[j] * gsd + compensacao_copa_m)
                    if db <= da:  
                        da = dists_px[ini] * gsd
                        db = dists_px[j] * gsd
                    g = substring(ls_geo, da, db)
                    g_px = substring(ls_px, da / gsd, db / gsd)
                    falhas.append({"comprimento_m": comp, "geom": g,
                                   "geom_px": np.array(g_px.coords)})
                    falha_total += comp

        resultados.append({
            "id": i + 1,
            "geom": geom_linha,
            "comprimento_m": comprimento_m,
            "falha_m": falha_total,
            "pct_falha": 100.0 * falha_total / comprimento_m if comprimento_m else 0.0,
            "falhas": falhas,
        })
    return resultados