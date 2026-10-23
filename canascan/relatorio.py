"""Relatórios básicos: camadas GPKG, TXT agregado, overlay e histograma."""
import csv
import os

import numpy as np
import geopandas as gpd

CLASSES = [(0.5, 1.0, "0,5–1 m"), (1.0, 2.0, "1–2 m"),
           (2.0, 5.0, "2–5 m"), (5.0, float("inf"), "> 5 m")]


def montar_camadas(resultados, crs):
    """Converte a lista de resultados (com id global e talhão) em GeoDataFrames."""
    gdf_linhas = gpd.GeoDataFrame(
        [{"linha_id": r["id"], "talhao": r.get("talhao", "1"),
          "comprimento_m": round(r["comprimento_m"], 2),
          "falha_m": round(r["falha_m"], 2),
          "pct_falha": round(r["pct_falha"], 2),
          "geometry": r["geom"]} for r in resultados], crs=crs)
    reg = []
    for r in resultados:
        for f in r["falhas"]:
            reg.append({"linha_id": r["id"], "talhao": r.get("talhao", "1"),
                        "comprimento_m": round(f["comprimento_m"], 2),
                        "geometry": f["geom"]})
    gdf_falhas = gpd.GeoDataFrame(reg, crs=crs) if reg else gpd.GeoDataFrame(
        columns=["linha_id", "talhao", "comprimento_m", "geometry"],
        geometry="geometry", crs=crs)
    return gdf_linhas, gdf_falhas


def resumo_global(resultados):
    total_linha = sum(r["comprimento_m"] for r in resultados)
    total_falha = sum(r["falha_m"] for r in resultados)
    todas = [f["comprimento_m"] for r in resultados for f in r["falhas"]]
    classes = []
    for lo, hi, nome in CLASSES:
        sel = [c for c in todas if lo <= c < hi]
        classes.append({"classe": nome, "qtd": len(sel), "metros": sum(sel)})
    return {"n_linhas": len(resultados),
            "comprimento_total_m": total_linha,
            "falha_total_m": total_falha,
            "pct_falha": 100.0 * total_falha / total_linha if total_linha else 0.0,
            "n_falhas": len(todas), "classes": classes}


def salvar_gpkg(gdf_linhas, gdf_falhas, pasta, gdf_talhoes=None):
    os.makedirs(pasta, exist_ok=True)
    gpkg = os.path.join(pasta, "resultado.gpkg")
    gdf_linhas.to_file(gpkg, layer="linhas_plantio", driver="GPKG")
    if len(gdf_falhas):
        gdf_falhas.to_file(gpkg, layer="falhas", driver="GPKG")
    if gdf_talhoes is not None and len(gdf_talhoes):
        gdf_talhoes.to_file(gpkg, layer="talhoes", driver="GPKG")
    with open(os.path.join(pasta, "falhas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["linha_id", "talhao", "comprimento_m"])
        for _, row in gdf_falhas.iterrows():
            w.writerow([row["linha_id"], row["talhao"], row["comprimento_m"]])
    return gpkg


def salvar_txt(res, resultados, pasta, angulos=None, por_talhao=None):
    linhas_rank = sorted(resultados, key=lambda r: r["falha_m"], reverse=True)[:10]
    with open(os.path.join(pasta, "relatorio.txt"), "w") as f:
        f.write("=" * 52 + "\nCANASCAN — RELATÓRIO DE FALHAS DE PLANTIO\n" + "=" * 52 + "\n\n")
        f.write(f"Linhas de plantio detectadas ..... {res['n_linhas']}\n")
        f.write(f"Comprimento total de linhas ...... {res['comprimento_total_m']:,.1f} m\n")
        f.write(f"Falha total ...................... {res['falha_total_m']:,.1f} m\n")
        f.write(f"Número de falhas ................. {res['n_falhas']}\n")
        f.write(f"ÍNDICE DE FALHAS (Stolf) ......... {res['pct_falha']:.2f} %\n\n")
        if por_talhao:
            f.write("Por talhão:\n")
            f.write(f"  {'talhão':>8} {'linhas':>7} {'plantio(m)':>11} {'falha(m)':>9} {'%':>7}\n")
            for d in por_talhao:
                f.write(f"  {d['talhao']:>8} {d['n_linhas']:>7} {d['plantio']:>11.1f} "
                        f"{d['falha']:>9.1f} {d['pct']:>6.2f}%\n")
            f.write("\n")
        f.write("Classificação das falhas por comprimento:\n")
        for c in res["classes"]:
            f.write(f"  {c['classe']:<8} {c['qtd']:>5} falhas   {c['metros']:>9.1f} m\n")
        f.write("\nTop 10 linhas mais falhadas:\n")
        f.write(f"  {'linha':>5} {'talhão':>7} {'compr(m)':>10} {'falha(m)':>10} {'%':>7}\n")
        for r in linhas_rank:
            f.write(f"  {r['id']:>5} {str(r.get('talhao','1')):>7} {r['comprimento_m']:>10.1f} "
                    f"{r['falha_m']:>10.1f} {r['pct_falha']:>6.1f}%\n")


def salvar_overlay(rgb, resultados, linhas_px, caminho_png, max_larg=2600):
    """Desenha o overlay numa MINIATURA da imagem (decimada): nunca tenta
    renderizar centenas de megapixels de uma vez."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = rgb.shape[:2]
    passo = max(1, int(np.ceil(w / max_larg)), int(np.ceil(h / max_larg)))
    img = rgb[::passo, ::passo]
    hs, ws = img.shape[:2]
    fig, ax = plt.subplots(figsize=(ws / 100, hs / 100), dpi=110)
    ax.imshow(img)
    for pts in linhas_px:
        ax.plot(pts[:, 0] / passo, pts[:, 1] / passo,
                color="#ffee00", lw=0.9, alpha=0.95)
    for r in resultados:
        for f in r["falhas"]:
            p = f["geom_px"]
            ax.plot(p[:, 0] / passo, p[:, 1] / passo,
                    color="red", lw=2.4, solid_capstyle="butt")
    ax.set_xlim(0, ws); ax.set_ylim(hs, 0); ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(caminho_png, bbox_inches="tight")
    plt.close(fig)


def salvar_histograma(res, pasta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nomes = [c["classe"] for c in res["classes"]]
    qtds = [c["qtd"] for c in res["classes"]]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    barras = ax.bar(nomes, qtds, color="#c0392b")
    ax.bar_label(barras)
    ax.set_title("Falhas por classe de comprimento")
    ax.set_ylabel("quantidade")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "histograma.png"))
    plt.close(fig)


# Faixas de falha no padrão usado pelos relatórios de usina
FAIXAS_USINA = [
    (0.0, 0.5, "Abaixo de 50 cm"),
    (0.5, 0.7, "50 a 70 cm"),
    (0.7, 1.0, "70 a 100 cm"),
    (1.0, 1.5, "100 a 150 cm"),
    (1.5, 2.0, "150 a 200 cm"),
    (2.0, float("inf"), "Acima de 200 cm"),
]


def _br(valor, casas=3):
    """Número no padrão brasileiro (vírgula decimal), como no CSV da usina."""
    return f"{valor:.{casas}f}".replace(".", ",")


def salvar_csv_usina(gdf_linhas, gdf_falhas, pasta, gdf_talhoes=None,
                     prefixo=""):
    """Exporta relatorio_geral.csv e relatorio_de_falhas.csv no MESMO
    formato dos relatórios de usina (separador ';', vírgula decimal,
    uma linha por talhão + linha 'total'), para comparação direta."""
    os.makedirs(pasta, exist_ok=True)

    areas = {}
    if gdf_talhoes is not None and len(gdf_talhoes):
        col = "talhao" if "talhao" in gdf_talhoes.columns else None
        for i, row in gdf_talhoes.iterrows():
            chave = str(row[col]) if col else str(i + 1)
            areas[chave] = row.geometry.area / 10000.0  # m² -> ha

    talhoes = sorted(set(gdf_linhas["talhao"].astype(str)),
                     key=lambda s: (len(s), s))
    linhas_geral, linhas_falhas = [], []
    tot = dict(ha=0.0, nl=0, nf=0, kml=0.0, kmf=0.0)

    for t in talhoes:
        sel_l = gdf_linhas[gdf_linhas["talhao"].astype(str) == t]
        sel_f = gdf_falhas[gdf_falhas["talhao"].astype(str) == t] \
            if len(gdf_falhas) else gdf_falhas
        n_l = len(sel_l)
        n_f = len(sel_f)
        km_l = sel_l["comprimento_m"].sum() / 1000.0
        km_f = (sel_f["comprimento_m"].sum() / 1000.0) if n_f else 0.0
        pct = 100.0 * km_f / km_l if km_l else 0.0
        ha = areas.get(t, 0.0)

        linhas_geral.append([t, _br(ha), n_l, n_f, _br(km_l), _br(km_f),
                             _br(pct, 2)])

        comp = sel_f["comprimento_m"].tolist() if n_f else []
        qtds, metros = [], []
        for lo, hi, _ in FAIXAS_USINA:
            faixa = [c for c in comp if lo <= c < hi]
            qtds.append(len(faixa))
            metros.append(_br(sum(faixa), 2))
        linhas_falhas.append([t, _br(ha), n_l, n_f, _br(km_l), _br(km_f),
                              _br(pct, 2), ""] + qtds + [""] + metros)

        tot["ha"] += ha; tot["nl"] += n_l; tot["nf"] += n_f
        tot["kml"] += km_l; tot["kmf"] += km_f

    pct_tot = 100.0 * tot["kmf"] / tot["kml"] if tot["kml"] else 0.0
    linhas_geral.append(["total", _br(tot["ha"]), tot["nl"], tot["nf"],
                         _br(tot["kml"]), _br(tot["kmf"]), _br(pct_tot, 2)])

    cab_geral = ["talhao", "tamanho (ha)", "numero de linhas",
                 "numero de falhas", "comprimento de linhas (km)",
                 "comprimento de falhas (km)", "porcentagem"]
    nomes = [n for _, _, n in FAIXAS_USINA]
    cab_falhas = cab_geral + ["Falhas"] + nomes + ["Tamanho"] + nomes

    p1 = os.path.join(pasta, f"{prefixo}relatorio_geral.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(cab_geral)
        w.writerows(linhas_geral)

    p2 = os.path.join(pasta, f"{prefixo}relatorio_de_falhas.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(cab_falhas)
        w.writerows(linhas_falhas)
    return p1, p2
