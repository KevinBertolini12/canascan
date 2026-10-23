"""Núcleo do pipeline: processa UMA área (imagem inteira ou janela de talhão)."""
import numpy as np

from . import vegetacao
from . import linhas as mod_linhas
from . import falhas as mod_falhas
from .geo import geo_para_px


def processar_area(rgb, valido, transform, gsd, *, espacamento=1.5,
                   metodo="unet", limiar=None, falha_min=0.5,
                   comp_copa=0.25, banda_m=4.0, min_linha=5.0,
                   sensibilidade=1.0, analisar_daninhas=False, linhas_geo=None,
                   adaptativo=False, tile_m=35.0):
    """Roda vegetação -> linhas -> falhas numa área.

    linhas_geo: lista opcional de LineStrings (no CRS do raster). Se
    fornecida, pula a detecção e usa essas linhas (equivalente ao campo
    "Linhas de plantio" do NeloreTech).
    Retorna (resultados, linhas_px, angulo, cobertura_verde).
    """
    
    # 1. Carrega a máscara ou predição baseada na escolha do usuário
    mask_falhas_unet = None
    if metodo == "unet":
        mask, mask_falhas_unet = vegetacao.mascara_unet_multiclasse(rgb, valido)
    else:
        if limiar is None:
            limiar = vegetacao.limiar_padrao(metodo)
        mask = vegetacao.mascara_vegetacao(rgb, valido, metodo=metodo, limiar=limiar)

    if mask.mean() < 0.002:  # bloco praticamente sem vegetação
        return [], [], None, float(mask.mean()), None

    angulo = None
    if linhas_geo:
        linhas_px = []
        for ls in linhas_geo:
            pts = geo_para_px(transform, np.asarray(ls.coords))
            if len(pts) >= 2:
                linhas_px.append(pts)
    else:
        intens = None
        if mask.mean() > 0.55:  # dossel fechando: orientação pela textura
            intens = vegetacao.indice_continuo(rgb, valido, metodo=metodo)
        if adaptativo:
            # ângulo/espaçamento LOCAIS por tile — segue plantio em curva de
            # nível, onde a orientação varia ao longo do talhão
            from . import adaptativo as mod_adap
            linhas_px, angulo = mod_adap.detectar_linhas_adaptativo(
                mask, gsd, espacamento, banda_m=banda_m,
                min_comprimento_m=min_linha, sensibilidade=sensibilidade,
                tile_m=tile_m, intens=intens)
        else:
            linhas_px, angulo = mod_linhas.detectar_linhas(
                mask, gsd, espacamento, banda_m=banda_m,
                min_comprimento_m=min_linha, sensibilidade=sensibilidade,
                intens=intens)

    resultados = mod_falhas.analisar_falhas(
        mask, linhas_px, gsd, transform, espacamento=espacamento,
        falha_min=falha_min, compensacao_copa_m=comp_copa,
        mask_falhas_ia=mask_falhas_unet)

    daninhas = None
    if analisar_daninhas and linhas_px:
        from . import daninhas as mod_dan
        daninhas = mod_dan.detectar(mask, linhas_px, gsd, transform)
    return resultados, linhas_px, angulo, float(mask.mean()), daninhas