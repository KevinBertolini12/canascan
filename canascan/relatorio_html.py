"""Relatório HTML de falhas — identidade CanaScan, 3 páginas, autocontido.

Feito para dois leitores ao mesmo tempo: o leigo (resumo em uma frase, selo
de classificação, termômetro, legendas explicativas, glossário) e o técnico
(tabelas por talhão, classes de comprimento, grid hexagonal de 1 ha).
"""
import base64
import io
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import __version__

IMPACTO_M = 1.5
CLASSES_PCT = [(0, 5, "Classe 1", "#2e7d32"), (5, 10, "Classe 2", "#9ccc65"),
               (10, 20, "Classe 3", "#ef6c00"), (20, 1e9, "Classe 4", "#c62828")]
INTERPRETACAO = {"Classe 1": "Bom", "Classe 2": "Moderado",
                 "Classe 3": "Ruim", "Classe 4": "Crítico"}
CAT_FALHAS = [(0, 1, "&lt; 1 m"), (1, 1.5, "1 – 1,5 m"), (1.5, 5, "1,5 – 5 m"),
              (5, 10, "5 – 10 m"), (10, 1e9, "&gt; 10 m")]

CSS = """
<style>
  :root { --verde:#1b5e20; --verde2:#2e7d32; --lima:#9ccc65; --fundo:#eef2ee; }
  body{font-family:'Segoe UI',system-ui,Arial,sans-serif;color:#263238;
       background:var(--fundo);margin:0}
  .pagina{background:#fff;max-width:1150px;margin:20px auto;padding:26px 36px;
          border-radius:10px;box-shadow:0 2px 14px rgba(27,94,32,.18)}
  .cab{display:flex;justify-content:space-between;align-items:flex-end;
       padding-bottom:8px}
  .marca{font-size:24px;font-weight:800;color:var(--verde);letter-spacing:.3px}
  .marca small{font-weight:400;color:#607d6b;font-size:13px;display:block}
  .cab .dir{text-align:right;font-size:12px;color:#607d6b}
  .filete{height:5px;border-radius:3px;margin:2px 0 16px 0;
          background:linear-gradient(90deg,var(--verde),var(--lima))}
  h2.sec{font-size:15px;color:var(--verde);margin:18px 0 8px 0;
         border-left:5px solid var(--lima);padding-left:8px}
  .grade{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}
  .painel{border:1px solid #d7e3d7;border-radius:8px;overflow:hidden;
          background:#fff}
  .bloco-tit{background:var(--verde);color:#fff;font-weight:700;font-size:13px;
             padding:7px 12px;letter-spacing:.4px}
  .cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:12px}
  .card{background:#f6faf6;border:1px solid #dbe8db;border-left:4px solid
        var(--verde2);border-radius:6px;text-align:center;padding:8px 4px;
        transition:transform .18s ease, box-shadow .18s ease;position:relative}
  .card:hover{transform:translateY(-4px);
        box-shadow:0 10px 20px rgba(27,94,32,.28);z-index:6}
  [data-dica]{cursor:help}
  [data-dica]:hover::after{content:attr(data-dica);position:absolute;
        bottom:104%;left:50%;transform:translateX(-50%);background:#263238;
        color:#fff;padding:7px 10px;border-radius:6px;font-size:11px;
        font-weight:400;letter-spacing:0;text-transform:none;width:200px;
        line-height:1.35;z-index:9;box-shadow:0 4px 12px rgba(0,0,0,.35)}
  [data-dica]:hover::before{content:"";position:absolute;bottom:96%;left:50%;
        transform:translateX(-50%);border:6px solid transparent;
        border-top-color:#263238;z-index:9}
  tr:hover td{background:#e2f2e2 !important}
  .destaque{transition:transform .18s ease}
  .destaque:hover{transform:scale(1.03)}
  .card .rot{font-size:9.5px;letter-spacing:.6px;color:#5d7263;
             text-transform:uppercase}
  .card .val{font-size:20px;font-weight:800;color:#1f3a26;margin-top:2px}
  .destaque{background:var(--verde);color:#fff;text-align:center;padding:12px;
            margin:10px 12px;border-radius:8px}
  .destaque .v{font-size:28px;font-weight:800}
  .faixa{display:flex;justify-content:space-between;align-items:center;
         background:#fff8e1;border:1.5px solid #e0b93f;border-radius:6px;
         margin:10px 12px;padding:8px 12px;font-weight:700}
  .fator{display:flex;justify-content:space-between;align-items:center;
         background:#e8f5e9;border:1.5px solid #66bb6a;border-radius:6px;
         margin:10px 12px;padding:9px 12px;font-weight:700}
  .fator .v{color:var(--verde2);font-size:20px}
  .frase{font-size:17px;background:#f1f8e9;border:1px solid #cfe4bd;
         border-radius:8px;padding:12px 16px;margin:6px 0}
  .selo{display:inline-block;color:#fff;font-weight:800;font-size:20px;
        border-radius:999px;padding:8px 22px}
  .caixa{background:#fffde7;border:1px solid #e6d97a;border-radius:8px;
         padding:10px 14px;font-size:12.5px}
  .caixa b{color:#7a5e00}
  .legenda-mapa{font-size:11px;color:#546e5a;padding:6px 10px;
                border-top:1px dashed #d7e3d7}
  .term{position:relative;height:16px;border-radius:8px;overflow:hidden;
        display:flex;margin:8px 0 2px 0}
  .term div{height:100%}
  .marcador{position:relative;height:0}
  .marcador span{position:absolute;top:-24px;transform:translateX(-50%);
        background:#263238;color:#fff;font-size:11px;font-weight:700;
        padding:1px 7px;border-radius:4px;white-space:nowrap}
  .escala{display:flex;justify-content:space-between;font-size:10px;
          color:#78909c}
  table{border-collapse:collapse;width:100%;font-size:11.5px;margin:6px 0}
  th{background:var(--verde);color:#fff;padding:5px 7px;text-align:left;
     font-size:10.5px}
  td{border-bottom:1px solid #e2ece2;padding:4px 7px}
  tr:nth-child(even) td{background:#f7fbf7}
  .cor{display:inline-block;width:11px;height:11px;margin-right:5px;
       border:1px solid #789;border-radius:2px;vertical-align:-1px}
  img.mapa{width:100%;display:block}
  dl.glos{font-size:11.5px;margin:4px 0}
  dl.glos dt{font-weight:700;color:var(--verde);float:left;clear:left;
             width:120px}
  dl.glos dd{margin:0 0 5px 128px}
  .rodape{display:flex;justify-content:space-between;color:#78909c;
          font-size:10.5px;border-top:1px solid #d7e3d7;margin-top:14px;
          padding-top:6px}
  @media print{body{background:#fff}.pagina{box-shadow:none;margin:0;
               border-radius:0;page-break-after:always}}
</style>"""

LOGO_SVG = '''<img src="data:image/png;base64,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" style="height:36px;vertical-align:-10px;margin-right:4px">'''

GLOSSARIO = [
    ("Ortomosaico", "o \"mapa-foto\" da fazenda montado com as imagens do drone; cada pixel tem coordenada real."),
    ("Linha de plantio", "a fileira onde a cana foi plantada; o CanaScan desenha uma linha sobre cada uma."),
    ("Falha", "trecho da fileira sem plantas com 0,5 m ou mais (critério Stolf, medido de colmo a colmo)."),
    ("Índice de falhas", "de cada 100 m de fileira, quantos metros estão vazios. Quanto menor, melhor."),
    ("Talhão", "cada \"quadra\" da fazenda; os resultados são somados por talhão."),
]


def fmt(v, casas=1):
    s = f"{v:,.{casas}f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


def fig_para_b64(fig, dpi=135):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def card(rot, val, dica=None):
    d = f' data-dica="{dica}"' if dica else ""
    return (f'<div class="card"{d}><div class="rot">{rot}</div>'
            f'<div class="val">{val}</div></div>')


def pagina(titulo, subtitulo, corpo, pag, total, agora):
    return (f'<div class="pagina"><div class="cab">'
            f'<div class="marca">🌱 CanaScan<small>{titulo} — {subtitulo}'
            f'</small></div>'
            f'<div class="dir">{agora}<br>Página {pag}/{total}</div></div>'
            f'<div class="filete"></div>{corpo}'
            f'<div class="rodape"><span>Gerado automaticamente pelo CanaScan '
            f'v{__version__}</span><span>Página {pag}/{total}</span></div></div>')


def selo(pct):
    if pct <= 5:
        return "EXCELENTE", "#2e7d32"
    if pct <= 10:
        return "BOM", "#7cb342"
    if pct <= 20:
        return "ATENÇÃO", "#ef6c00"
    return "CRÍTICO", "#c62828"


def termometro(pct):
    pos = min(pct, 30.0) / 30.0 * 100
    return f"""
<div class="marcador" style="margin-top:26px"><span style="left:{pos:.1f}%">
{fmt(pct)}%</span></div>
<div class="term">
  <div style="width:16.7%;background:#2e7d32"></div>
  <div style="width:16.7%;background:#7cb342"></div>
  <div style="width:33.3%;background:#ef6c00"></div>
  <div style="width:33.3%;background:#c62828"></div>
</div>
<div class="escala"><span>0%</span><span>5%</span><span>10%</span>
<span>20%</span><span>30%+</span></div>"""


def classe_de(pct):
    for lo, hi, nome, cor in CLASSES_PCT:
        if lo <= pct < hi:
            return nome, cor
    return CLASSES_PCT[-1][2], CLASSES_PCT[-1][3]


def _limites_talhoes(ax, gdf_talhoes):
    if gdf_talhoes is not None and len(gdf_talhoes):
        gdf_talhoes.boundary.plot(ax=ax, color="#33443a", linewidth=1.2)
        for _, r in gdf_talhoes.iterrows():
            c = r.geometry.representative_point()
            ax.annotate(str(r["talhao"]), (c.x, c.y), ha="center",
                        fontsize=8, color="#33443a")


def _eixo_limpo(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def grade_hexagonal(uniao, area_ha=1.0):
    lado = math.sqrt(2 * area_ha * 10000 / (3 * math.sqrt(3)))
    minx, miny, maxx, maxy = uniao.bounds
    dx, dy = 1.5 * lado, math.sqrt(3) * lado
    hexes, col = [], 0
    x = minx - lado
    while x < maxx + lado:
        y = miny - lado - (dy / 2 if col % 2 else 0)
        while y < maxy + lado:
            pts = [(x + lado * math.cos(a), y + lado * math.sin(a))
                   for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
            h = Polygon(pts)
            if h.intersects(uniao):
                rec = h.intersection(uniao)
                if not rec.is_empty and rec.area > 1:
                    hexes.append(rec)
            y += dy
        x += dx
        col += 1
    return hexes


def gerar(gdf_linhas, gdf_falhas, gdf_talhoes, pasta,
          arquivo="relatorio_falhas.html", falha_min=0.5,
          daninhas_m2=None, daninhas_n=None):
    os.makedirs(pasta, exist_ok=True)
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")

    n_linhas = len(gdf_linhas)
    plantio_m = float(gdf_linhas["comprimento_m"].sum())
    falhas_m = float(gdf_falhas["comprimento_m"].sum()) if len(gdf_falhas) else 0.0
    n_falhas = len(gdf_falhas)
    pct = 100 * falhas_m / plantio_m if plantio_m else 0.0
    rotulo, cor_selo = selo(pct)

    if gdf_talhoes is not None and len(gdf_talhoes):
        uniao = unary_union(gdf_talhoes.geometry)
        n_talhoes = len(gdf_talhoes)
    else:
        uniao = unary_union(gdf_linhas.geometry).convex_hull.buffer(3)
        n_talhoes = 1
    area_ha = uniao.area / 1e4

    # por talhão (ordenado do pior p/ o melhor)
    por_t = []
    for t, gl in gdf_linhas.groupby("talhao"):
        gf = gdf_falhas[gdf_falhas["talhao"] == t] if len(gdf_falhas) else gdf_falhas
        pm = gl.comprimento_m.sum()
        fm = gf.comprimento_m.sum() if len(gf) else 0.0
        ar = 0.0
        if gdf_talhoes is not None and len(gdf_talhoes):
            ar = float(gdf_talhoes[gdf_talhoes["talhao"] == t]
                       .geometry.area.sum()) / 1e4
        por_t.append({"talhao": t, "area": ar, "plantio": pm, "falhas": fm,
                      "pct": 100 * fm / pm if pm else 0, "n": len(gf)})
    por_t.sort(key=lambda x: x["pct"], reverse=True)

    # ------------------------------------------------ PÁGINA 1: VISÃO GERAL
    frase = (f"A cada <b>100 metros</b> de fileira plantada, "
             f"<b>{fmt(pct)} metros</b> estão sem cana.")
    campos = falhas_m / 100.0
    if campos >= 1:
        frase += (f" Somadas, as falhas equivalem a <b>{fmt(campos, 0)} campos"
                  f" de futebol</b> enfileirados.")

    recs = []
    if len(por_t) > 1:
        piores = ", ".join(f"talhão {d['talhao']} ({fmt(d['pct'])}%)"
                           for d in por_t[:2])
        recs.append(f"Priorize a inspeção de campo em: <b>{piores}</b>.")
    grandes = gdf_falhas[gdf_falhas.comprimento_m >= 2.0] if len(gdf_falhas) else []
    if len(grandes):
        recs.append(f"Existem <b>{len(grandes)} falhas com 2 m ou mais</b> "
                    f"({fmt(float(grandes.comprimento_m.sum())/1000, 2)} km) — "
                    f"veja o relatório de replantio p/ a rota ótima.")
    recs.append("Confira as camadas no QGIS (resultado.gpkg) antes de "
                "decisões definitivas.")
    caixa_recs = ("<div class='caixa'><b>Onde agir primeiro:</b><br>"
                  + "<br>".join("• " + r for r in recs) + "</div>")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    _limites_talhoes(ax, gdf_talhoes)
    pb = 0.0
    if len(gdf_falhas):
        baixo = gdf_falhas[gdf_falhas.comprimento_m < IMPACTO_M]
        alto = gdf_falhas[gdf_falhas.comprimento_m >= IMPACTO_M]
        pb = 100 * len(baixo) / n_falhas
        if len(baixo):
            baixo.plot(ax=ax, color="#7cb342", linewidth=1.0)
        if len(alto):
            alto.plot(ax=ax, color="#c62828", linewidth=1.5)
    _eixo_limpo(ax)
    mapa1 = fig_para_b64(fig)

    tab_t = "".join(
        f"<tr><td>Talhão {d['talhao']}</td><td>{fmt(d['area'])}</td>"
        f"<td>{fmt(d['plantio']/1000, 2)}</td><td>{fmt(d['falhas']/1000, 3)}</td>"
        f"<td><b>{fmt(d['pct'])}%</b></td><td>{d['n']}</td></tr>" for d in por_t)

    corpo1 = f"""
<div class="grade">
  <div>
    <div class="frase">{frase}</div>
    <div style="display:flex;align-items:center;gap:16px;margin:12px 0">
      <span class="selo" style="background:{cor_selo}">{rotulo}</span>
      <div style="flex:1">{termometro(pct)}</div>
    </div>
    <div class="painel"><div class="bloco-tit">ONDE ESTÃO AS FALHAS</div>
      <img class="mapa" src="data:image/png;base64,{mapa1}">
      <div class="legenda-mapa">Cada traço é um trecho de fileira sem plantas.
      <span style="color:#7cb342">■</span> curtas (&lt;{fmt(IMPACTO_M)} m,
      {fmt(pb)}%) &nbsp; <span style="color:#c62828">■</span> longas
      (≥{fmt(IMPACTO_M)} m, {fmt(100-pb)}%) — as longas pesam mais na
      produtividade.</div></div>
  </div>
  <div>
    <div class="painel"><div class="bloco-tit">NÚMEROS DA ÁREA</div>
      <div class="cards">
        {card("Área analisada", fmt(area_ha) + " ha", "Soma da área dos talhões incluídos nesta análise.")}{card("Talhões", n_talhoes, "Quadras da fazenda processadas.")}
        {card("Fileiras mapeadas", n_linhas, "Quantidade de linhas de plantio que o CanaScan encontrou e traçou sobre a imagem.")}{card("Plantio medido", fmt(plantio_m/1000, 2) + " km", "Extensão total das fileiras, somadas, em quilômetros.")}
        {card("Falhas encontradas", n_falhas, "Número de trechos vazios com 0,5 m ou mais dentro das fileiras.")}{card("Extensão das falhas", fmt(falhas_m/1000, 3) + " km", "Se você emendasse todas as falhas uma atrás da outra, este seria o comprimento.")}
        {(card("Daninhas entrelinha", fmt(daninhas_m2, 0) + " m²", "Vegetação detectada FORA das linhas de plantio — mato na entrelinha, candidato a manejo localizado.") + card("Manchas de mato", daninhas_n, "Quantidade de focos de daninha mapeados (camada no resultado.gpkg).")) if daninhas_m2 else ""}
      </div>
      <div class="destaque" data-dica="De cada 100 metros de fileira, quantos estão sem plantas. Quanto menor, melhor."><div class="v">{fmt(pct)}%</div>ÍNDICE DE FALHAS
      (Stolf)</div></div>
    <h2 class="sec">Ranking por talhão</h2>
    <div class="painel"><table><tr><th>Talhão</th><th>Área (ha)</th>
      <th>Plantio (km)</th><th>Falhas (km)</th><th>%</th><th>Nº</th></tr>
      {tab_t}</table></div>
    <div style="margin-top:12px">{caixa_recs}</div>
  </div>
</div>"""

    # ------------------------------------------ PÁGINA 2: FILEIRA POR FILEIRA
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    _limites_talhoes(ax, gdf_talhoes)
    cont, leg_itens = [], []
    for lo, hi, nome, cor in CLASSES_PCT:
        sel = gdf_linhas[(gdf_linhas.pct_falha >= lo) & (gdf_linhas.pct_falha < hi)]
        if len(sel):
            sel.plot(ax=ax, color=cor, linewidth=0.9)
        cont.append((nome, cor, lo, hi, sel))
        rot = f"&lt;{hi:g}%" if lo == 0 else (f"&gt;{lo:g}%" if hi > 100
                                              else f"{lo:g}–{hi:g}%")
        leg_itens.append(f'<span class="cor" style="background:{cor}"></span>'
                         f"{rot}: {len(sel)}")
    _eixo_limpo(ax)
    mapa2 = fig_para_b64(fig)

    tab2 = "".join(
        f'<tr><td><span class="cor" style="background:{cor}"></span>{nome} '
        f"({INTERPRETACAO[nome]})</td>"
        f"<td>{('&lt;' + format(hi, 'g') + '%') if lo == 0 else (('&gt;' + format(lo, 'g') + '%') if hi > 100 else f'{lo:g}–{hi:g}%')}</td>"
        f"<td>{len(sel)}</td><td>{fmt(sel.comprimento_m.sum()/1000, 2)}</td>"
        f"<td>{fmt(sel.falha_m.sum()/1000, 3)}</td></tr>"
        for nome, cor, lo, hi, sel in cont)

    comp = gdf_falhas["comprimento_m"].values if len(gdf_falhas) else np.array([])
    tab_d = ""
    for lo, hi, nome in CAT_FALHAS:
        sel = comp[(comp >= lo) & (comp < hi)]
        pc = 100 * sel.sum() / falhas_m if falhas_m else 0
        tab_d += (f"<tr><td>{nome}</td><td>{len(sel)}</td>"
                  f"<td>{fmt(sel.sum())} m</td><td>{fmt(pc)}%</td></tr>")

    corpo2 = f"""
<div class="frase">Cada fileira ganhou uma <b>nota individual</b>: a cor
mostra a % de falha daquela fileira específica. Verde = saudável;
vermelho = candidata a replantio.</div>
<div class="grade">
  <div class="painel"><div class="bloco-tit">FILEIRAS COLORIDAS PELA % DE FALHA</div>
    <img class="mapa" src="data:image/png;base64,{mapa2}">
    <div class="legenda-mapa">{" &nbsp; ".join(leg_itens)} fileiras</div></div>
  <div>
    <h2 class="sec" style="margin-top:0">Distribuição por classe</h2>
    <div class="painel"><table><tr><th>Classe</th><th>% falha</th>
      <th>Nº fileiras</th><th>Plantio (km)</th><th>Falhas (km)</th></tr>
      {tab2}</table></div>
    <h2 class="sec">Tamanho das falhas</h2>
    <div class="painel"><table><tr><th>Comprimento</th><th>Qtd.</th>
      <th>Metros</th><th>%</th></tr>{tab_d}</table></div>
    <div class="caixa" style="margin-top:12px"><b>Como usar:</b> falhas
      curtas espalhadas costumam ser da brotação; falhas longas concentradas
      nas mesmas fileiras indicam problema de plantio, solo ou máquina —
      e são as mais baratas de corrigir por replantio dirigido.</div>
  </div>
</div>"""

    # ------------------------------------------------ PÁGINA 3: MAPA DE CALOR
    hexes = grade_hexagonal(uniao, 1.0)
    sidx_l = gdf_linhas.sindex
    sidx_f = gdf_falhas.sindex if len(gdf_falhas) else None
    hex_info = []
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for h in hexes:
        il = list(sidx_l.query(h, predicate="intersects"))
        pm = float(gdf_linhas.iloc[il].intersection(h).length.sum()) if il else 0.0
        fm = 0.0
        if sidx_f is not None:
            i_f = list(sidx_f.query(h, predicate="intersects"))
            fm = float(gdf_falhas.iloc[i_f].intersection(h).length.sum()) if i_f else 0.0
        pc = 100 * fm / pm if pm else 0.0
        nome, cor = classe_de(pc)
        hex_info.append({"pct": pc, "classe": nome, "area": h.area / 1e4,
                         "plantio": pm, "falhas": fm})
        xs, ys = (h.exterior.xy if h.geom_type == "Polygon"
                  else max(h.geoms, key=lambda g: g.area).exterior.xy)
        ax.fill(xs, ys, color=cor, alpha=0.85, linewidth=0.4,
                edgecolor="#5d7263")
    if len(gdf_falhas):
        gdf_falhas.plot(ax=ax, color="#4a0d0d", linewidth=0.8)
    _limites_talhoes(ax, gdf_talhoes)
    _eixo_limpo(ax)
    mapa3 = fig_para_b64(fig)

    tab3 = ""
    for lo, hi, nome, cor in CLASSES_PCT:
        sel = [d for d in hex_info if d["classe"] == nome]
        ar = sum(d["area"] for d in sel)
        rot = ("&lt;" + format(hi, "g") + "%") if lo == 0 else \
            (("&gt;" + format(lo, "g") + "%") if hi > 100 else f"{lo:g}–{hi:g}%")
        tab3 += (f'<tr><td><span class="cor" style="background:{cor}"></span>'
                 f"{nome} ({INTERPRETACAO[nome]})</td><td>{rot}</td>"
                 f"<td>{len(sel)}</td><td>{fmt(ar, 2)}</td>"
                 f"<td>{fmt(100*ar/area_ha if area_ha else 0)}%</td></tr>")

    glos = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in GLOSSARIO)
    corpo3 = f"""
<div class="frase">Visão de <b>mapa de calor</b>: a área foi dividida em
hexágonos de 1 hectare — bata o olho e veja as zonas críticas sem precisar
olhar fileira por fileira.</div>
<div class="grade">
  <div class="painel"><div class="bloco-tit">HEXÁGONOS DE 1 ha ({len(hexes)}
    hexágonos | {fmt(area_ha)} ha)</div>
    <img class="mapa" src="data:image/png;base64,{mapa3}">
    <div class="legenda-mapa">Traços escuros = falhas reais sobrepostas.
    Verde = zona saudável; vermelho = zona crítica.</div></div>
  <div>
    <h2 class="sec" style="margin-top:0">Classificação dos hexágonos</h2>
    <div class="painel"><table><tr><th>Classe</th><th>% falha</th>
      <th>Nº</th><th>Área (ha)</th><th>% área</th></tr>{tab3}</table></div>
    <h2 class="sec">Glossário rápido</h2>
    <div class="painel" style="padding:10px 12px"><dl class="glos">{glos}</dl>
    </div>
    <div class="caixa" style="margin-top:12px"><b>Metodologia:</b> vão sem
      plantas ≥ {fmt(falha_min)} m na fileira (critério Stolf), medido de
      colmo a colmo com compensação de copa. Todo o processamento é local —
      as imagens não saem do seu computador.</div>
  </div>
</div>"""

    html = ("<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Relatório de Falhas — CanaScan</title>{CSS}</head><body>"
            + pagina("RELATÓRIO DE FALHAS NO PLANTIO", "Visão geral",
                     corpo1, 1, 3, agora)
            + pagina("RELATÓRIO DE FALHAS NO PLANTIO", "Fileira por fileira",
                     corpo2, 2, 3, agora)
            + pagina("RELATÓRIO DE FALHAS NO PLANTIO", "Mapa de calor (1 ha)",
                     corpo3, 3, 3, agora)
            + "</body></html>")
    caminho = os.path.join(pasta, arquivo)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html)
    return caminho
