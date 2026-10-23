"""Calibra o indice de vegetacao por projeto — rapido, usa um RECORTE.

Cada voo tem luz, camera e hora do dia diferentes. Um limiar fixo nao serve
para todos: se ele for alto demais a cana nao e vista como verde e o vao
dentro da fileira vira "falha". Este script testa varias combinacoes em um
recorte pequeno e mostra qual aproxima o indice de falha do CanaScan do
indice medido na REFERENCIA do mesmo recorte.

Uso:
  python calibrar_projeto.py --projeto "C:/.../projetos/3007_ANALISE"
  python calibrar_projeto.py --raiz "C:/.../projetos"        (todos)
"""
import argparse
import glob
import os
import warnings

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canascan import pipeline, vegetacao  # noqa: E402

COMBOS = [
    ("gli", 0.010), ("gli", 0.015), ("gli", 0.020), ("gli", 0.030),
    ("razao", 1.10), ("razao", 1.20), ("razao", 1.35), ("razao", 1.50),
    ("exg_otsu", None),
]


def carregar_ref(pasta, sub, prefixo, crs):
    shps = sorted(glob.glob(os.path.join(pasta, "cana", sub,
                                         prefixo + "*.shp")))
    gdfs = []
    for s in shps:
        try:
            g = gpd.read_file(s)
            if len(g):
                gdfs.append(g[["geometry"]])
        except Exception:
            pass
    if not gdfs:
        return None
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                           geometry="geometry", crs=gdfs[0].crs)
    if gdf.crs is not None and gdf.crs.is_geographic:
        b = gdf.total_bounds
        if np.isfinite(b).all() and np.abs(b).max() > 360:
            gdf = gdf.set_crs(crs, allow_override=True)
    elif gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def calibrar(pasta, lado=2500, espacamento=1.45):
    nome = os.path.basename(pasta)
    tifs = [f for f in os.listdir(pasta) if f.lower().endswith((".tif", ".tiff"))]
    if not tifs:
        print(f"{nome}: sem imagem")
        return None
    principal = next((f for f in tifs if "ortomosaico" in f.lower()), tifs[0])
    tif = os.path.join(pasta, principal)

    with rasterio.open(tif) as src:
        crs = src.crs
        ref_l = carregar_ref(pasta, "linhas_propagadas", "linha_propagada", crs)
        ref_f = carregar_ref(pasta, "falhas", "falha", crs)
        if ref_l is None:
            print(f"{nome}: sem referencia")
            return None

        # recorte centrado no meio das linhas da referencia
        b = ref_l.total_bounds
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        linha, col = src.index(cx, cy)
        c0 = max(0, int(col - lado / 2)); r0 = max(0, int(linha - lado / 2))
        w = min(lado, src.width - c0); h = min(lado, src.height - r0)
        jan = Window(c0, r0, w, h)
        arr = src.read(indexes=[1, 2, 3], window=jan)
        tr = src.window_transform(jan)
        gsd = src.res[0]
        bounds = rasterio.windows.bounds(jan, src.transform)

    rgb = np.transpose(arr, (1, 2, 0))
    valido = rgb.max(axis=2) > 0
    caixa = box(*bounds)

    # restringe ao corredor da REFERENCIA: sem isso o recorte pode cair
    # numa borda e comparar areas diferentes
    from rasterio.features import rasterize as _rasterize
    lin_corte = ref_l.clip(caixa)
    if len(lin_corte):
        faixa = _rasterize(
            [(g.buffer(espacamento * 0.75), 1) for g in lin_corte.geometry
             if g is not None and not g.is_empty],
            out_shape=valido.shape, transform=tr, fill=0, dtype="uint8",
            all_touched=True).astype(bool)
        if faixa.mean() > 0.01:
            valido = valido & faixa

    # indice da REFERENCIA dentro do recorte
    lin_rec = ref_l.clip(caixa)
    km_ref = lin_rec.geometry.length.sum()
    fal_ref = ref_f.clip(caixa).geometry.length.sum() if ref_f is not None else 0.0
    pct_ref = 100.0 * fal_ref / km_ref if km_ref else 0.0

    print(f"\n{'='*62}\n  {nome}  (recorte {w}x{h} px | GSD {gsd*100:.1f} cm)")
    print(f"  REFERENCIA no recorte: {km_ref/1000:.2f} km de linha | "
          f"falha {pct_ref:.2f}%")
    print(f"{'='*62}")
    print(f"  {'metodo':10s} {'limiar':>8s} {'verde%':>7s} {'km linha':>9s} "
          f"{'falha%':>8s} {'erro':>8s}")

    melhor = None
    melhor_km = None
    for metodo, limiar in COMBOS:
        try:
            res, lin_px, ang, cob, _ = pipeline.processar_area(
                rgb, valido, tr, gsd, espacamento=espacamento,
                metodo=metodo, limiar=limiar)
            if not res:
                print(f"  {metodo:10s} {str(limiar):>8s} {'—':>7s} "
                      f"{'sem linhas':>9s}")
                continue
            km = sum(r["comprimento_m"] for r in res)
            fal = sum(r["falha_m"] for r in res)
            pct = 100.0 * fal / km if km else 0.0
            erro = abs(pct - pct_ref)
            marca = ""
            if melhor is None or erro < melhor[0]:
                melhor = (erro, metodo, limiar, pct, km)
                marca = " <<<"
            erro_km = abs(km - km_ref) / km_ref if km_ref else 9e9
            # empate técnico na metragem (<3%): fica com a máscara mais
            # completa (maior cobertura verde), que é a mais confiável
            if melhor_km is None or erro_km < melhor_km[0] - 0.03 or \
                    (abs(erro_km - melhor_km[0]) <= 0.03 and cob > melhor_km[5]):
                melhor_km = (erro_km, metodo, limiar, pct, km, cob)
            print(f"  {metodo:10s} {str(limiar):>8s} {100*cob:>6.1f}% "
                  f"{km/1000:>8.2f} {pct:>7.2f}% {erro:>7.2f}{marca}")
        except Exception as e:
            print(f"  {metodo:10s} {str(limiar):>8s}  erro: {str(e)[:30]}")

    if not melhor_km:
        return None

    # ESTAGIO 2 — com o indice que melhor reproduz a METRAGEM DE LINHA,
    # varre o comprimento de copa (o parametro que controla o quanto da
    # planta e considerado alem do ponto detectado) para acertar a falha.
    _, metodo, limiar, _, _, cob_esc = melhor_km
    print(f"\n  Indice escolhido: {metodo}"
          + (f" {limiar}" if limiar is not None else "")
          + f" (verde {100*cob_esc:.1f}%)")
    print(f"  {'copa (m)':>9s} {'falha min':>10s} {'falha%':>8s} {'erro':>8s}")
    melhor_copa = None
    for copa in (-0.20, 0.0, 0.25):
        for fmin in (0.50, 0.75, 1.00, 1.25, 1.50):
            try:
                res, _, _, _, _ = pipeline.processar_area(
                    rgb, valido, tr, gsd, espacamento=espacamento,
                    metodo=metodo, limiar=limiar, comp_copa=copa,
                    falha_min=fmin)
                if not res:
                    continue
                km = sum(r["comprimento_m"] for r in res)
                fal = sum(r["falha_m"] for r in res)
                pct = 100.0 * fal / km if km else 0.0
                erro = abs(pct - pct_ref)
                marca = ""
                if melhor_copa is None or erro < melhor_copa[0]:
                    melhor_copa = (erro, copa, pct, fmin)
                    marca = " <<<"
                print(f"  {copa:>9.2f} {fmin:>10.2f} {pct:>7.2f}% "
                      f"{erro:>7.2f}{marca}")
            except Exception as e:
                print(f"  {copa:>9.2f} {fmin:>10.2f}  erro: {str(e)[:26]}")

    if melhor_copa:
        erro, copa, pct, fmin = melhor_copa
        print(f"\n  MELHOR AJUSTE: --metodo {metodo}"
              + (f" --limiar {limiar}" if limiar is not None else "")
              + f" --comp-copa {copa} --falha-min {fmin}")
        print(f"                 falha {pct:.2f}% vs {pct_ref:.2f}% "
              f"da referencia (erro {erro:.2f} pp)")
        return (erro, metodo, limiar, pct, copa, fmin)
    return melhor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projeto", default=None)
    ap.add_argument("--raiz", default=None)
    ap.add_argument("--lado", type=int, default=2500,
                    help="lado do recorte em pixels (padrao 2500)")
    ap.add_argument("--espacamento", type=float, default=1.45)
    args = ap.parse_args()

    if args.projeto:
        calibrar(args.projeto, args.lado, args.espacamento)
        return
    if not args.raiz:
        print("Informe --projeto ou --raiz")
        return
    resumo = []
    for d in sorted(os.listdir(args.raiz)):
        p = os.path.join(args.raiz, d)
        if os.path.isdir(os.path.join(p, "cana")):
            m = calibrar(p, args.lado, args.espacamento)
            if m:
                resumo.append((d,) + tuple(m[1:]))
    if resumo:
        print(f"\n{'='*62}\n  RESUMO — melhor ajuste por projeto\n{'='*62}")
        for item in resumo:
            nome, metodo, limiar = item[0], item[1], item[2]
            copa = item[4] if len(item) > 4 else None
            lim = f"--limiar {limiar} " if limiar is not None else ""
            cp = f"--comp-copa {copa}" if copa else ""
            print(f"  {nome:16s} --metodo {metodo:9s} {lim}{cp}")


if __name__ == "__main__":
    main()
