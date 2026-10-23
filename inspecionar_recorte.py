"""Painel visual de diagnostico — mostra POR QUE as falhas divergem.

Gera um PNG com quatro paineis do MESMO recorte:
  1. imagem original
  2. mascara de vegetacao (o que o CanaScan enxerga como planta)
  3. nossas linhas + nossas falhas
  4. nossas linhas + falhas da REFERENCIA

Comparar 2 e 4 responde a pergunta central: a mascara esta esburacada
(vendo vao onde ha planta) ou a referencia e que ignora vaos reais?

Uso:
  python inspecionar_recorte.py --projeto "C:/.../3178_ANALISE"
  python inspecionar_recorte.py --projeto "..." --lado 1200 --metodo exg_otsu
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canascan import pipeline, vegetacao, geo  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projeto", required=True)
    ap.add_argument("--lado", type=int, default=1200,
                    help="lado do recorte em px (menor = mais zoom)")
    ap.add_argument("--metodo", default="exg_otsu")
    ap.add_argument("--limiar", type=float, default=None)
    ap.add_argument("--espacamento", type=float, default=1.45)
    ap.add_argument("--comp-copa", type=float, default=0.25, dest="comp_copa")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    pasta = args.projeto
    nome = os.path.basename(pasta.rstrip("/\\"))
    tifs = [f for f in os.listdir(pasta) if f.lower().endswith((".tif", ".tiff"))]
    principal = next((f for f in tifs if "ortomosaico" in f.lower()), tifs[0])
    tif = os.path.join(pasta, principal)

    with rasterio.open(tif) as src:
        crs = src.crs
        ref_l = carregar_ref(pasta, "linhas_propagadas", "linha_propagada", crs)
        ref_f = carregar_ref(pasta, "falhas", "falha", crs)
        b = ref_l.total_bounds
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        lin, col = src.index(cx, cy)
        c0 = max(0, int(col - args.lado / 2)); r0 = max(0, int(lin - args.lado / 2))
        w = min(args.lado, src.width - c0); h = min(args.lado, src.height - r0)
        jan = Window(c0, r0, w, h)
        arr = src.read(indexes=[1, 2, 3], window=jan)
        tr = src.window_transform(jan)
        gsd = src.res[0]
        bounds = rasterio.windows.bounds(jan, src.transform)

    rgb = np.transpose(arr, (1, 2, 0))
    valido = rgb.max(axis=2) > 0
    caixa = box(*bounds)

    mask = vegetacao.mascara_vegetacao(rgb, valido, metodo=args.metodo,
                                       limiar=args.limiar)
    res, lin_px, ang, cob, _ = pipeline.processar_area(
        rgb, valido, tr, gsd, espacamento=args.espacamento,
        metodo=args.metodo, limiar=args.limiar, comp_copa=args.comp_copa)

    km = sum(r["comprimento_m"] for r in res) if res else 0.0
    fal = sum(r["falha_m"] for r in res) if res else 0.0
    pct = 100.0 * fal / km if km else 0.0
    lin_rec = ref_l.clip(caixa)
    km_ref = lin_rec.geometry.length.sum()
    fal_ref = ref_f.clip(caixa).geometry.length.sum() if ref_f is not None else 0.0
    pct_ref = 100.0 * fal_ref / km_ref if km_ref else 0.0

    def para_px(geoms):
        saida = []
        for g in geoms:
            if g is None or g.is_empty:
                continue
            gs = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
            for p in gs:
                xy = np.asarray(p.coords)
                px = geo.geo_para_px(tr, xy)
                saida.append(np.asarray(px))
        return saida

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()
    for a in axs:
        a.set_xticks([]); a.set_yticks([])

    axs[0].imshow(rgb)
    axs[0].set_title(f"{nome} — imagem ({w}x{h} px, GSD {gsd*100:.1f} cm)",
                     fontsize=11)

    axs[1].imshow(mask, cmap="Greens")
    axs[1].set_title(f"máscara de vegetação ({args.metodo}) — "
                     f"verde {100*cob:.1f}%", fontsize=11)

    axs[2].imshow(rgb)
    for pts in (lin_px or []):
        p = np.asarray(pts)
        axs[2].plot(p[:, 0], p[:, 1], "-", color="yellow", lw=0.7)
    for r in (res or []):
        for f in r["falhas"]:
            p = np.asarray(f["geom_px"])
            axs[2].plot(p[:, 0], p[:, 1], "-", color="red", lw=1.8)
    axs[2].set_title(f"CanaScan: {km/1000:.2f} km linha | falha {pct:.2f}%",
                     fontsize=11)

    axs[3].imshow(rgb)
    for p in para_px(lin_rec.geometry):
        axs[3].plot(p[:, 0], p[:, 1], "-", color="deepskyblue", lw=0.7)
    if ref_f is not None:
        for p in para_px(ref_f.clip(caixa).geometry):
            axs[3].plot(p[:, 0], p[:, 1], "-", color="magenta", lw=2.2)
    axs[3].set_title(f"REFERÊNCIA: {km_ref/1000:.2f} km linha | "
                     f"falha {pct_ref:.2f}%", fontsize=11)

    for a in axs:
        a.set_xlim(0, w); a.set_ylim(h, 0)

    destino = args.saida or os.path.join(pasta, f"inspecao_{nome}.png")
    plt.tight_layout()
    plt.savefig(destino, dpi=110)
    print(f"\n  Painel salvo em: {destino}")
    print(f"  CanaScan  : {km/1000:.2f} km | falha {pct:.2f}%")
    print(f"  Referência: {km_ref/1000:.2f} km | falha {pct_ref:.2f}%")
    print("\n  Compare o painel 2 (máscara) com o painel 4 (referência):")
    print("   - máscara cheia de buracos DENTRO da fileira -> falha inflada")
    print("   - máscara boa e falhas nossas curtas e espalhadas -> critério")
    print("     de falha diferente do da referência")


if __name__ == "__main__":
    main()
