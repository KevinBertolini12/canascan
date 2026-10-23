"""Comparação CanaScan vs. referência (X-Farming / NeloreTech / gabarito).

Recebe as camadas do nosso resultado e as camadas exportadas pelo software
de referência (linhas e/ou falhas da MESMA área) e mede a concordância:

- Falhas: metragem total, contagem, recall (quanto da referência achamos) e
  precisão (quanto do nosso a referência confirma), com tolerância em metros.
- Linhas: metragem, cobertura mútua e desvio lateral (mediana/p90) medido
  por amostragem de pontos.

Saídas: resumo no terminal, comparacao.txt e comparacao.html (com mapas).
"""
import os
from datetime import datetime

import numpy as np
import geopandas as gpd
import shapely
from shapely.ops import unary_union

from .relatorio_html import CSS, CAT_FALHAS, card, fig_para_b64, fmt, pagina


def _ler_camada(caminho_espec, crs_destino=None):
    """Lê um vetor. Aceita 'arq.gpkg::camada' e múltiplos caminhos com ';'.

    Corrige automaticamente CRS declarado errado (cabeçalho geográfico com
    coordenadas métricas — comum em exportações de softwares de usina)."""
    partes = [c.strip() for c in caminho_espec.split(";") if c.strip()]
    gdfs = []
    for p in partes:
        if "::" in p:
            caminho, camada = p.split("::", 1)
            g = gpd.read_file(caminho, layer=camada)
        else:
            g = gpd.read_file(p)
        g = g[~g.geometry.is_empty & g.geometry.notna()].copy()
        gdfs.append(g[["geometry"]])
    gdf = gpd.pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdfs[0].crs
                           if hasattr(gdfs[0], "crs") else None)
    if gdf.crs is not None and gdf.crs.is_geographic:
        b = gdf.total_bounds
        if np.isfinite(b).all() and (np.abs(b) > 360).any():
            # conteúdo métrico com rótulo geográfico: assume o CRS do nosso
            if crs_destino is not None:
                gdf = gdf.set_crs(crs_destino, allow_override=True)
    if crs_destino is not None and gdf.crs is not None and gdf.crs != crs_destino:
        gdf = gdf.to_crs(crs_destino)
    return gdf


def _so_linhas(gdf):
    tipos = set(gdf.geometry.geom_type)
    return tipos <= {"LineString", "MultiLineString"}


def _amostrar_pontos(gdf_linhas, passo=2.0, max_pontos=20000):
    blocos = []
    for geom in gdf_linhas.geometry:
        partes = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for ls in partes:
            if ls.length < passo:
                continue
            d = np.arange(0, ls.length, passo)
            blocos.append(shapely.get_coordinates(
                shapely.line_interpolate_point(ls, d)))
    if not blocos:
        return np.empty((0, 2))
    pts = np.vstack(blocos)
    if len(pts) > max_pontos:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pontos,
                                                  replace=False)]
    return pts


def _desvio_lateral(gdf_nosso, gdf_ref):
    """Distância de pontos das nossas linhas até a linha de referência
    mais próxima (mediana, p90)."""
    pts = _amostrar_pontos(gdf_nosso)
    if not len(pts) or not len(gdf_ref):
        return None
    pontos = shapely.points(pts)
    tree = shapely.STRtree(gdf_ref.geometry.values)
    idx = tree.nearest(pontos)
    dists = shapely.distance(pontos, gdf_ref.geometry.values[idx])
    return {"mediana": float(np.median(dists)),
            "p90": float(np.percentile(dists, 90)),
            "dentro_25cm": float(np.mean(dists <= 0.25) * 100),
            "dentro_50cm": float(np.mean(dists <= 0.50) * 100)}


def _concordancia(gdf_a, gdf_b, tol):
    """% do comprimento de A que está a até tol metros de B (e vice-versa)."""
    ua, ub = unary_union(gdf_a.geometry), unary_union(gdf_b.geometry)
    la, lb = ua.length, ub.length
    a_em_b = ua.intersection(ub.buffer(tol)).length / la * 100 if la else 0.0
    b_em_a = ub.intersection(ua.buffer(tol)).length / lb * 100 if lb else 0.0
    return a_em_b, b_em_a, la, lb


def comparar(nosso_linhas, nosso_falhas, ref_linhas, ref_falhas,
             tol_falhas=0.5, tol_linhas=0.5):
    m = {"tol_falhas": tol_falhas, "tol_linhas": tol_linhas}

    if ref_falhas is not None and len(ref_falhas) and len(nosso_falhas):
        prec, rec, l_nosso, l_ref = _concordancia(nosso_falhas, ref_falhas,
                                                  tol_falhas)
        m["falhas"] = {
            "n_nosso": len(nosso_falhas), "n_ref": len(ref_falhas),
            "m_nosso": l_nosso, "m_ref": l_ref,
            "dif_pct": (l_nosso - l_ref) / l_ref * 100 if l_ref else 0.0,
            "precisao": prec, "recall": rec,
        }
        # classes lado a lado
        classes = []
        cn = nosso_falhas.geometry.length.values
        cr = ref_falhas.geometry.length.values
        for lo, hi, nome in CAT_FALHAS:
            classes.append({"classe": nome,
                            "nosso": float(cn[(cn >= lo) & (cn < hi)].sum()),
                            "ref": float(cr[(cr >= lo) & (cr < hi)].sum())})
        m["falhas"]["classes"] = classes

    if ref_linhas is not None and len(ref_linhas) and len(nosso_linhas):
        cob_nosso, cob_ref, l_nosso, l_ref = _concordancia(
            nosso_linhas, ref_linhas, tol_linhas)
        m["linhas"] = {
            "n_nosso": len(nosso_linhas), "n_ref": len(ref_linhas),
            "m_nosso": l_nosso, "m_ref": l_ref,
            "cobertura_nosso": cob_nosso, "cobertura_ref": cob_ref,
            "desvio": _desvio_lateral(nosso_linhas, ref_linhas),
        }
    return m


def _veredito(m):
    f = m.get("falhas")
    if not f:
        return ("SEM FALHAS PARA COMPARAR", "#888",
                "Forneça a camada de falhas da referência p/ o veredito completo.")
    dif = abs(f["dif_pct"])
    conc = (f["precisao"] + f["recall"]) / 2
    if dif < 10 and conc >= 85:
        return ("ALTA CONCORDÂNCIA", "#1d8a34",
                "O CanaScan reproduz o resultado da referência dentro da "
                "margem operacional. Pronto p/ uso lado a lado.")
    if dif < 25 and conc >= 70:
        return ("CONCORDÂNCIA PARCIAL — CALIBRAR", "#e07b1a",
                "Estrutura correta, mas vale ajustar parâmetros: --limiar, "
                "--comp-copa e --espacamento são os primeiros suspeitos.")
    return ("BAIXA CONCORDÂNCIA — INVESTIGAR", "#a51f1f",
            "Confira CRS/área (é a MESMA área?), estágio da cultura, "
            "espaçamento e método de vegetação antes de concluir.")


def _mapa(camadas, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    proxies, rotulos = [], []
    for gdf, cor, lw, rotulo in camadas:
        if gdf is not None and len(gdf):
            gdf.plot(ax=ax, color=cor, linewidth=lw, alpha=0.85)
            proxies.append(Line2D([0], [0], color=cor, lw=2.5))
            rotulos.append(rotulo)
    ax.legend(proxies, rotulos, fontsize=8, loc="lower right")
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig_para_b64(fig)


def gerar(nosso_linhas, nosso_falhas, ref_linhas, ref_falhas, pasta,
          tol_falhas=0.5, tol_linhas=0.5, nome_ref="Referência"):
    os.makedirs(pasta, exist_ok=True)
    m = comparar(nosso_linhas, nosso_falhas, ref_linhas, ref_falhas,
                 tol_falhas, tol_linhas)
    titulo_v, cor_v, texto_v = _veredito(m)
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")

    blocos = [f"""
<div style="border:2px solid {cor_v};border-left:14px solid {cor_v};
     padding:10px 14px;margin-bottom:14px">
  <div style="font-size:19px;font-weight:800;color:{cor_v}">{titulo_v}</div>
  <div style="font-size:12px;color:#333;margin-top:3px">{texto_v}</div>
</div>"""]

    f = m.get("falhas")
    if f:
        mapa_f = _mapa([(ref_falhas, "#2456a8", 2.6, f"Falhas — {nome_ref}"),
                        (nosso_falhas, "#d21f1f", 1.1, "Falhas — CanaScan")],
                       "Falhas sobrepostas (azul = referência, vermelho = CanaScan)")
        tab = "".join(
            f"<tr><td>{c['classe']}</td><td>{fmt(c['nosso'])} m</td>"
            f"<td>{fmt(c['ref'])} m</td></tr>" for c in f["classes"])
        sinal = "+" if f["dif_pct"] >= 0 else ""
        blocos.append(f"""
<div class="grade">
  <div class="painel"><div class="bloco-tit">FALHAS — CANASCAN × {nome_ref.upper()}</div>
    <img class="mapa" src="data:image/png;base64,{mapa_f}"></div>
  <div>
    <div class="painel"><div class="cards">
      {card("Falha CanaScan", fmt(f["m_nosso"]) + " m")}{card("Falha " + nome_ref, fmt(f["m_ref"]) + " m")}
      {card("Diferença", sinal + fmt(f["dif_pct"]) + "%")}{card("Nº (nosso/ref)", f"{f['n_nosso']} / {f['n_ref']}")}
      {card("Recall", fmt(f["recall"]) + "%")}{card("Precisão", fmt(f["precisao"]) + "%")}
    </div>
    <div class="faixa"><span>Tolerância espacial</span><span>{fmt(tol_falhas)} m</span></div></div>
    <div class="painel" style="margin-top:12px"><div class="bloco-tit">METRAGEM POR CLASSE</div>
      <table><tr><th>Classe</th><th>CanaScan</th><th>{nome_ref}</th></tr>{tab}</table></div>
  </div>
</div>""")

    li = m.get("linhas")
    if li:
        mapa_l = _mapa([(ref_linhas, "#2456a8", 1.5, f"Linhas — {nome_ref}"),
                        (nosso_linhas, "#e8c500", 0.7, "Linhas — CanaScan")],
                       "Linhas sobrepostas (azul = referência, amarelo = CanaScan)")
        d = li["desvio"] or {}
        blocos.append(f"""
<div class="grade" style="margin-top:16px">
  <div class="painel"><div class="bloco-tit">LINHAS — CANASCAN × {nome_ref.upper()}</div>
    <img class="mapa" src="data:image/png;base64,{mapa_l}"></div>
  <div><div class="painel"><div class="cards">
      {card("Linhas (nosso/ref)", f"{li['n_nosso']} / {li['n_ref']}")}
      {card("Plantio (nosso/ref)", fmt(li['m_nosso']/1000, 1) + " / " + fmt(li['m_ref']/1000, 1) + " km")}
      {card("Nosso coberto p/ ref", fmt(li["cobertura_nosso"]) + "%")}
      {card("Ref coberta p/ nosso", fmt(li["cobertura_ref"]) + "%")}
      {card("Desvio lateral mediano", fmt(100*d.get("mediana", 0), 0) + " cm")}
      {card("Desvio p90", fmt(100*d.get("p90", 0), 0) + " cm")}
      {card("Pontos a ≤25 cm", fmt(d.get("dentro_25cm", 0)) + "%")}
      {card("Pontos a ≤50 cm", fmt(d.get("dentro_50cm", 0)) + "%")}
  </div></div></div>
</div>""")

    html = ("<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Comparação — CanaScan</title>{CSS}</head><body>"
            + pagina("COMPARAÇÃO DE RESULTADOS", f"CanaScan × {nome_ref}",
                     "".join(blocos), 1, 1, agora) + "</body></html>")
    caminho = os.path.join(pasta, "comparacao.html")
    with open(caminho, "w", encoding="utf-8") as fh:
        fh.write(html)

    # resumo texto
    with open(os.path.join(pasta, "comparacao.txt"), "w") as fh:
        fh.write(f"CANASCAN x {nome_ref} — {titulo_v}\n{texto_v}\n\n")
        if f:
            fh.write(f"FALHAS  nosso {f['m_nosso']:,.1f} m | ref {f['m_ref']:,.1f} m "
                     f"| dif {f['dif_pct']:+.1f}% | recall {f['recall']:.1f}% "
                     f"| precisão {f['precisao']:.1f}%\n")
        if li:
            d = li["desvio"] or {}
            fh.write(f"LINHAS  nosso {li['m_nosso']/1000:,.1f} km | ref "
                     f"{li['m_ref']/1000:,.1f} km | desvio mediano "
                     f"{100*d.get('mediana', 0):.0f} cm | p90 {100*d.get('p90', 0):.0f} cm\n")
    return caminho, m, (titulo_v, texto_v)
