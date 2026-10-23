"""Processa e COMPARA vários projetos da usina de uma vez.

Para cada projeto encontrado:
  1. processa o ortomosaico com o CanaScan (usando TALHAO.shp se existir);
  2. compara o resultado com as camadas de referência do projeto
     (cana/linhas_propagadas/*.shp e cana/falhas/*.shp);
  3. salva o relatório HTML da comparação dentro do próprio projeto.

No fim imprime um QUADRO CONSOLIDADO com todos os projetos lado a lado —
é esse quadro que vale levar para a reunião.

Uso:
  python comparar_usina_lote.py --raiz "C:/Users/Kevin/Downloads/projetos"
  python comparar_usina_lote.py --raiz "..." --limiar 0.02 --so-comparar
"""
import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def encontrar_projetos(raiz):
    projetos = []
    for dirpath, dirs, files in os.walk(raiz):
        tifs = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tifs or not os.path.isdir(os.path.join(dirpath, "cana")):
            continue
        principal = next((f for f in tifs if "ortomosaico" in f.lower()), tifs[0])
        projetos.append((dirpath, os.path.join(dirpath, principal)))
    return sorted(projetos)


def shps(pasta, prefixo):
    return sorted(glob.glob(os.path.join(pasta, prefixo + "*.shp")))


def achar_talhao(pasta):
    """Procura um shapefile de talhoes de verdade.

    NAO usa shape_principal.shp: nos projetos da usina ele e o contorno do
    VOO (inclui mata, estrada, carreador). Processar por ele infla o indice
    de falhas e torna a comparacao injusta.
    """
    for nome in ("TALHAO.shp", "TALHÃO.shp", "talhao.shp", "talhoes.shp"):
        c = os.path.join(pasta, nome)
        if os.path.isfile(c):
            return c
    return None


def area_da_referencia(shp_linhas, destino, folga=3.0):
    """Gera um poligono de analise cobrindo exatamente onde a REFERENCIA
    trabalhou (buffer em torno das linhas dela, dissolvido por talhao).

    Assim o CanaScan e a referencia medem o mesmo pedaco de chao — sem
    isso a comparacao mistura mata e estrada de um lado so.
    """
    import geopandas as gpd
    import pandas as pd
    gdfs = []
    for i, s in enumerate(shp_linhas, start=1):
        g = gpd.read_file(s)
        if not len(g):
            continue
        g = g[["geometry"]]
        b = g.total_bounds
        if g.crs is not None and g.crs.is_geographic and abs(max(b, key=abs)) > 360:
            g = g.set_crs(g.crs, allow_override=True)  # mantem metrico
        poly = g.buffer(folga).union_all() if hasattr(g.buffer(folga), "union_all") \
            else g.buffer(folga).unary_union
        gdfs.append({"talhao": str(i), "geometry": poly})
    if not gdfs:
        return None
    gdf = gpd.GeoDataFrame(gdfs, geometry="geometry",
                           crs=gpd.read_file(shp_linhas[0]).crs)
    gdf.to_file(destino, driver="GPKG")
    return destino


def main():
    ap = argparse.ArgumentParser(
        description="Processa e compara projetos da usina em lote")
    ap.add_argument("--raiz", required=True)
    ap.add_argument("--metodo", default="gli",
                    choices=["gli", "razao", "exg_otsu"],
                    help="indice de vegetacao (padrao gli, casa com limiar 0.02)")
    ap.add_argument("--limiar", type=float, default=0.02)
    ap.add_argument("--espacamento", type=float, default=1.5)
    ap.add_argument("--sensibilidade", type=float, default=1.0)
    ap.add_argument("--tolerancia", type=float, default=0.5,
                    help="tolerancia espacial da comparacao (m)")
    ap.add_argument("--nome-ref", default="Software de referência",
                    dest="nome_ref")
    ap.add_argument("--folga", type=float, default=3.0,
                    help="folga (m) em torno das linhas da referencia ao "
                         "montar a area de analise")
    ap.add_argument("--sem-area-ref", action="store_true",
                    dest="sem_area_ref",
                    help="nao montar area a partir da referencia")
    ap.add_argument("--so-comparar", action="store_true",
                    help="pula o processamento (usa resultado ja existente)")
    args = ap.parse_args()

    projetos = encontrar_projetos(args.raiz)
    if not projetos:
        print(f"Nenhum projeto encontrado em {args.raiz}")
        return

    print(f"\n{'='*60}\n  Projetos encontrados: {len(projetos)}\n{'='*60}\n")
    resumo = []

    for i, (pasta, tif) in enumerate(projetos, 1):
        nome = os.path.basename(pasta)
        saida = os.path.join(pasta, "canascan_saida")
        gpkg = os.path.join(saida, "resultado.gpkg")
        print(f"[{i}/{len(projetos)}] {nome}")

        if not args.so_comparar:
            cmd = [sys.executable, "-m", "canascan.cli", "processar",
                   "--tif", tif, "--saida", saida,
                   "--metodo", args.metodo,
                   "--limiar", str(args.limiar),
                   "--espacamento", str(args.espacamento),
                   "--sensibilidade", str(args.sensibilidade)]
            talhao = achar_talhao(pasta)
            if not talhao and not args.sem_area_ref:
                ref_l_prev = shps(os.path.join(pasta, "cana",
                                               "linhas_propagadas"),
                                  "linha_propagada")
                if ref_l_prev:
                    alvo = os.path.join(pasta, "area_referencia.gpkg")
                    try:
                        talhao = area_da_referencia(ref_l_prev, alvo,
                                                    folga=args.folga)
                        if talhao:
                            print("      talhoes: area da REFERENCIA "
                                  "(mesma area dos dois lados)")
                    except Exception as e:
                        print(f"      [aviso] area da referencia falhou: {e}")
            elif talhao:
                print(f"      talhoes: {os.path.basename(talhao)}")
            if talhao:
                cmd += ["--talhoes", talhao]
            print("      processando…")
            r = subprocess.run(cmd, capture_output=True, text=True)
            saida_txt = r.stdout + r.stderr
            idx = next((l for l in saida_txt.splitlines()
                        if "ÍNDICE DE FALHAS" in l), "")
            if idx:
                print(f"      {idx.strip()}")
            if r.returncode != 0 or not os.path.isfile(gpkg):
                print(f"      [ERRO] falha ao processar")
                resumo.append((nome, None))
                continue

        if not os.path.isfile(gpkg):
            print("      [aviso] sem resultado.gpkg — pulando comparacao")
            resumo.append((nome, None))
            continue

        ref_l = shps(os.path.join(pasta, "cana", "linhas_propagadas"),
                     "linha_propagada")
        ref_f = shps(os.path.join(pasta, "cana", "falhas"), "falha")
        if not ref_l:
            print("      [aviso] sem linhas de referencia — pulando comparacao")
            resumo.append((nome, None))
            continue
        print(f"      referencia: {len(ref_l)} linhas | {len(ref_f)} falhas")

        cmd = [sys.executable, "-m", "canascan.cli", "comparar",
               "--nosso", gpkg,
               "--ref-linhas", ";".join(ref_l),
               "--saida", os.path.join(pasta, "comparacao"),
               "--tolerancia", str(args.tolerancia),
               "--nome-ref", args.nome_ref]
        if ref_f:
            cmd += ["--ref-falhas", ";".join(ref_f)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        txt = r.stdout + r.stderr
        dados = {}
        for linha in txt.splitlines():
            l = linha.strip()
            if l.startswith("Falhas:"):
                dados["falhas"] = l
            elif l.startswith("Recall"):
                dados["recall"] = l
            elif l.startswith("Plantio:"):
                dados["plantio"] = l
            elif l.startswith("Linhas:") and "desvio" in l:
                dados["linhas"] = l
            elif "CONCORD" in l or "DIVERG" in l:
                dados["veredito"] = l
        for k in ("veredito", "plantio", "falhas", "recall", "linhas"):
            if k in dados:
                print(f"      {dados[k]}")
        resumo.append((nome, dados))
        print()

    print(f"\n{'='*60}\n  QUADRO CONSOLIDADO\n{'='*60}")
    for nome, d in resumo:
        if not d:
            print(f"  {nome:22s} — sem comparacao")
            continue
        print(f"  {nome}")
        for k in ("veredito", "plantio", "falhas", "recall", "linhas"):
            if k in d:
                print(f"      {d[k]}")
    print(f"\n  Relatorios HTML: <projeto>/comparacao/comparacao.html")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
