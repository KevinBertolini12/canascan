"""Audita a REFERENCIA de cada projeto — rapido, so le shapefiles.

Para cada projeto mede, a partir das proprias linhas da referencia:
  - espacamento REAL entre fileiras vizinhas (mediana);
  - km de linhas e numero de fileiras;
  - comprimento medio de fileira.

Serve para descobrir se o espacamento que usamos (1,50 m por padrao) bate
com a realidade de cada projeto. Espacamento errado faz o detector inventar
fileiras — e cada fileira inventada vira falha.

Uso:
  python auditar_referencia.py --raiz "C:/Users/Kevin/Downloads/projetos"
"""
import argparse
import glob
import os

import warnings

import numpy as np
import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")


def encontrar_projetos(raiz):
    projetos = []
    for dirpath, dirs, files in os.walk(raiz):
        if os.path.isdir(os.path.join(dirpath, "cana")):
            projetos.append(dirpath)
    return sorted(set(projetos))


def espacamento_mediano(gdf, amostras=400):
    """Mede a distancia tipica entre uma fileira e sua vizinha mais proxima."""
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if len(geoms) < 3:
        return None
    rng = np.random.default_rng(0)
    idx = rng.choice(len(geoms), size=min(amostras, len(geoms)),
                     replace=False)
    dists = []
    for i in idx:
        g = geoms[i]
        p = g.interpolate(0.5, normalized=True)
        melhor = None
        # janela pequena em volta do ponto medio da fileira
        jan = gdf.cx[p.x - 20:p.x + 20, p.y - 20:p.y + 20]
        for j, g2 in zip(jan.index, jan.geometry):
            if g2 is None or g2.is_empty or g2.equals(g):
                continue
            d = p.distance(g2)
            if d > 0.05 and (melhor is None or d < melhor):
                melhor = d
        if melhor is not None:
            dists.append(melhor)
    if not dists:
        return None
    return float(np.median(dists))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", required=True)
    args = ap.parse_args()

    linhas_tab = []
    for pasta in encontrar_projetos(args.raiz):
        nome = os.path.basename(pasta)
        shps = sorted(glob.glob(os.path.join(
            pasta, "cana", "linhas_propagadas", "linha_propagada*.shp")))
        if not shps:
            continue
        gdfs = []
        for s in shps:
            try:
                g = gpd.read_file(s)
                if len(g):
                    gdfs.append(g[["geometry"]])
            except Exception:
                pass
        if not gdfs:
            continue
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                               geometry="geometry", crs=gdfs[0].crs)
        km = gdf.geometry.length.sum() / 1000.0
        n = len(gdf)
        med = gdf.geometry.length.median()
        esp = espacamento_mediano(gdf)
        linhas_tab.append({
            "projeto": nome, "talhoes": len(shps), "fileiras": n,
            "km_linhas": round(km, 1),
            "fileira_media_m": round(med, 1),
            "espacamento_m": round(esp, 2) if esp else None,
        })

    if not linhas_tab:
        print("Nenhuma referencia encontrada.")
        return

    df = pd.DataFrame(linhas_tab)
    print()
    print(df.to_string(index=False))
    print()
    espacos = [r["espacamento_m"] for r in linhas_tab if r["espacamento_m"]]
    if espacos:
        print(f"  espacamento mediano geral: {np.median(espacos):.2f} m")
        fora = [r for r in linhas_tab if r["espacamento_m"]
                and abs(r["espacamento_m"] - 1.5) > 0.15]
        if fora:
            print("\n  PROJETOS FORA DE 1,50 m (use --espacamento proprio):")
            for r in fora:
                print(f"    {r['projeto']}: {r['espacamento_m']} m")
        else:
            print("  todos proximos de 1,50 m")


if __name__ == "__main__":
    main()
