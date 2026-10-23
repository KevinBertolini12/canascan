"""Diagnostica os projetos da usina: por que um projeto rendeu poucos tiles?

Para cada projeto mostra:
  - CRS e área da imagem;
  - CRS e área das camadas de referência;
  - se a referência cai DENTRO da imagem (a causa nº 1 de "0 tiles");
  - contagem de feições por camada.

Uso:
  python diagnosticar_projetos.py --raiz "C:/Users/Kevin/Downloads/projetos"
  python diagnosticar_projetos.py --raiz "..." --projeto 3113_ANALISE
"""
import argparse
import glob
import os

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


def encontrar_projetos(raiz):
    projetos = []
    for dirpath, dirs, files in os.walk(raiz):
        tifs = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tifs or not os.path.isdir(os.path.join(dirpath, "cana")):
            continue
        principal = next((f for f in tifs if "ortomosaico" in f.lower()), tifs[0])
        projetos.append((dirpath, os.path.join(dirpath, principal)))
    return sorted(projetos)


def carregar(shps):
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
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                            geometry="geometry", crs=gdfs[0].crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--projeto", default=None,
                    help="diagnostica só este projeto (nome da pasta)")
    args = ap.parse_args()

    projetos = encontrar_projetos(args.raiz)
    if args.projeto:
        projetos = [p for p in projetos
                    if os.path.basename(p[0]) == args.projeto]

    for pasta, tif in projetos:
        nome = os.path.basename(pasta)
        print(f"\n{'='*58}\n  {nome}\n{'='*58}")

        with rasterio.open(tif) as src:
            crs_img = src.crs
            b = src.bounds
            cx_img = box(*b)
            print(f"  imagem : {src.width}x{src.height} px | CRS {crs_img}")
            print(f"           bounds X {b.left:.0f}–{b.right:.0f} "
                  f"| Y {b.bottom:.0f}–{b.top:.0f}")

        for sub, prefixo in [("linhas_propagadas", "linha_propagada"),
                             ("falhas", "falha"),
                             ("linhas_principais", "linha_principal")]:
            shps = sorted(glob.glob(os.path.join(pasta, "cana", sub,
                                                 prefixo + "*.shp")))
            if not shps:
                print(f"  {sub:18s}: (nenhum .shp)")
                continue
            g = carregar(shps)
            if g is None or not len(g):
                print(f"  {sub:18s}: {len(shps)} arquivos, 0 feições")
                continue
            gb = g.total_bounds
            crs_txt = str(g.crs).split("\n")[0][:38] if g.crs else "SEM CRS"
            geog = " [rótulo geográfico]" if (g.crs and g.crs.is_geographic) else ""
            print(f"  {sub:18s}: {len(shps)} arq | {len(g)} feições | "
                  f"{crs_txt}{geog}")
            print(f"  {'':18s}  bounds X {gb[0]:.0f}–{gb[2]:.0f} "
                  f"| Y {gb[1]:.0f}–{gb[3]:.0f}")

            # sobreposição real (corrigindo o CRS rotulado errado)
            g2 = g
            if g.crs is not None and g.crs.is_geographic and abs(gb).max() > 360:
                g2 = g.set_crs(crs_img, allow_override=True)
            elif g.crs is not None and crs_img is not None and g.crs != crs_img:
                try:
                    g2 = g.to_crs(crs_img)
                except Exception:
                    pass
            uni = g2.geometry.union_all() if hasattr(g2.geometry, "union_all") \
                else g2.geometry.unary_union
            dentro = uni.intersection(cx_img)
            if uni.length > 0:
                pct = 100.0 * dentro.length / uni.length
            else:
                pct = 100.0 * (dentro.area / uni.area if uni.area else 0)
            marca = "OK" if pct > 50 else ("PARCIAL" if pct > 1 else "FORA!")
            print(f"  {'':18s}  dentro da imagem: {pct:.1f}%  [{marca}]")

        print("\n  Leitura:")
        print("    FORA!  -> a referência não cai sobre a imagem: CRS trocado")
        print("             ou os .shp são de outro voo/área. Sem tiles.")
        print("    PARCIAL-> só parte casa; confira se o TIF é o mesmo voo.")


if __name__ == "__main__":
    main()
