"""Mostra os pontos onde MAIS discordamos da referencia — com zoom.

Encontra as maiores falhas que o CanaScan marcou e que NAO tem falha
correspondente na referencia, e gera um mosaico de recortes ampliados.
Cada recorte mostra a imagem crua ao lado da mesma imagem com a falha
marcada — para julgar a olho: ha cana ali ou nao?

E o material que fecha a discussao de criterio com a usina.

Uso:
  python divergencias.py --projeto "C:/.../3178_ANALISE"
  python divergencias.py --projeto "..." --n 8 --metodo exg_otsu
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
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canascan import pipeline  # noqa: E402


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
    ap.add_argument("--lado", type=int, default=2500,
                    help="lado do recorte analisado (px)")
    ap.add_argument("--n", type=int, default=6, help="quantas divergencias")
    ap.add_argument("--zoom", type=int, default=260,
                    help="lado do zoom em cada divergencia (px)")
    ap.add_argument("--metodo", default="exg_otsu")
    ap.add_argument("--limiar", type=float, default=None)
    ap.add_argument("--espacamento", type=float, default=1.45)
    ap.add_argument("--comp-copa", type=float, default=-0.20, dest="comp_copa")
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

    from rasterio.features import rasterize
    lin_rec = ref_l.clip(caixa)
    faixa = rasterize(
        [(g.buffer(args.espacamento * 0.75), 1) for g in lin_rec.geometry
         if g is not None and not g.is_empty],
        out_shape=valido.shape, transform=tr, fill=0, dtype="uint8",
        all_touched=True).astype(bool)
    if faixa.mean() > 0.01:
        valido = valido & faixa

    res, _, _, _, _ = pipeline.processar_area(
        rgb, valido, tr, gsd, espacamento=args.espacamento,
        metodo=args.metodo, limiar=args.limiar, comp_copa=args.comp_copa)

    falhas = []
    for r in (res or []):
        for f in r["falhas"]:
            falhas.append(f)
    if not falhas:
        print("Sem falhas detectadas no recorte.")
        return

    ref_falhas = ref_f.clip(caixa) if ref_f is not None else None
    uref = unary_union(ref_falhas.geometry) if (ref_falhas is not None
                                               and len(ref_falhas)) else None

    # falhas nossas SEM correspondente na referencia, maiores primeiro
    divergentes = []
    for f in falhas:
        if uref is not None and f["geom"].distance(uref) < 1.0:
            continue
        divergentes.append(f)
    divergentes.sort(key=lambda f: -f["comprimento_m"])
    escolhidas = divergentes[:args.n]

    print(f"\n  {nome}: {len(falhas)} falhas nossas no recorte")
    print(f"  sem correspondente na referencia: {len(divergentes)}")
    if not escolhidas:
        print("  nenhuma divergencia — otimo sinal!")
        return
    print(f"  mostrando as {len(escolhidas)} maiores "
          f"({escolhidas[0]['comprimento_m']:.1f} m a "
          f"{escolhidas[-1]['comprimento_m']:.1f} m)")

    n = len(escolhidas)
    fig, axs = plt.subplots(n, 2, figsize=(9, 4.4 * n))
    if n == 1:
        axs = np.array([axs])
    z = args.zoom
    for k, f in enumerate(escolhidas):
        p = np.asarray(f["geom_px"])
        mx, my = p[:, 0].mean(), p[:, 1].mean()
        x0 = int(max(0, min(w - z, mx - z / 2)))
        y0 = int(max(0, min(h - z, my - z / 2)))
        rec = rgb[y0:y0 + z, x0:x0 + z]
        for j in (0, 1):
            axs[k, j].imshow(rec)
            axs[k, j].set_xticks([]); axs[k, j].set_yticks([])
        axs[k, 1].plot(p[:, 0] - x0, p[:, 1] - y0, "-", color="red", lw=3,
                       alpha=0.85)
        axs[k, 0].set_ylabel(f"{f['comprimento_m']:.1f} m", fontsize=11)
        if k == 0:
            axs[k, 0].set_title("imagem", fontsize=11)
            axs[k, 1].set_title("falha marcada pelo CanaScan\n"
                                "(sem falha na referência)", fontsize=11)

    destino = args.saida or os.path.join(pasta, f"divergencias_{nome}.png")
    plt.tight_layout()
    plt.savefig(destino, dpi=110)
    print(f"\n  Mosaico salvo em: {destino}")
    print("\n  Olhe cada par: se houver cana na faixa vermelha, o CanaScan")
    print("  errou; se for solo limpo, a referencia e que deixou passar.")


if __name__ == "__main__":
    main()
