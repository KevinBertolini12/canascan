"""Análise de replantio com curva de otimização (estilo NeloreTech).

Lógica: só falhas >= falha mínima de replantio interessam (ex.: 2,0 m).
As linhas são ranqueadas por metros de falha replantável; percorrendo as
mais falhadas primeiro, calcula-se quanto % das falhas é coberto (x) contra
o % de linhas percorridas (custo). O FATOR DE OTIMIZAÇÃO = x − custo mede a
vantagem sobre atuar uniformemente; o cenário ÓTIMO é o pico dessa curva.
"""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .relatorio_html import CSS, card, fig_para_b64, fmt, pagina

CENARIOS = [("CENÁRIO ECONÔMICO", "#ef6c00", "s", 0.70),
            ("CENÁRIO RECOMENDADO", "#2e7d32", "o", 1.00),
            ("CENÁRIO COMPLETO", "#1565c0", "D", 1.30)]
SUBTITULOS = {"CENÁRIO ECONÔMICO": "abaixo do ótimo — esforço mínimo",
              "CENÁRIO RECOMENDADO": "ponto ótimo da curva",
              "CENÁRIO COMPLETO": "acima do ótimo — cobertura máxima"}


def analisar(gdf_linhas, gdf_falhas, falha_min_rep=2.0):
    fr = gdf_falhas[gdf_falhas["comprimento_m"] >= falha_min_rep].copy()
    if not len(fr):
        return None
    total_rep_m = float(fr["comprimento_m"].sum())

    agg = fr.groupby("linha_id").agg(
        rep_m=("comprimento_m", "sum"), n_falhas=("comprimento_m", "size"))
    comp = gdf_linhas.set_index("linha_id")["comprimento_m"]
    agg["comp_m"] = comp.reindex(agg.index).fillna(0)
    agg = agg.sort_values("rep_m", ascending=False)
    K = len(agg)

    x = 100 * np.cumsum(agg["rep_m"].values) / total_rep_m       # % falhas cobertas
    custo = 100 * np.arange(1, K + 1) / K                        # % linhas percorridas
    fator = x - custo
    k_opt = int(np.argmax(fator))

    cenarios = []
    for titulo, cor, marca, mult in CENARIOS:
        alvo = min(x[k_opt] * mult, x[-1])
        k = int(np.argmin(np.abs(x - alvo)))
        sel_ids = agg.index[:k + 1]
        sel_f = fr[fr["linha_id"].isin(sel_ids)]
        cenarios.append({
            "titulo": titulo, "cor": cor, "marca": marca, "k": k,
            "linhas_ids": list(sel_ids),
            "comp_linhas_m": float(agg["comp_m"].values[:k + 1].sum()),
            "n_falhas": int(len(sel_f)),
            "falhas_m": float(sel_f["comprimento_m"].sum()),
            "pct": float(x[k]), "fator": float(fator[k]),
        })
    return {"fr": fr, "agg": agg, "x": x, "fator": fator, "k_opt": k_opt,
            "total_rep_m": total_rep_m, "cenarios": cenarios,
            "falha_min_rep": falha_min_rep}


def _curva(res, cen):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = np.concatenate([[0], res["x"]])
    ys = np.concatenate([[0], res["fator"]])
    ax.plot(xs, ys, color="#2e7d32", lw=2.0, label="Curva de Otimização")
    ax.plot(cen["pct"], cen["fator"], cen["marca"], color=cen["cor"],
            markersize=9, label=f"{cen['pct']:.1f}% das falhas")
    ax.annotate(f"{fmt(cen['pct'])}%", (cen["pct"], cen["fator"]),
                textcoords="offset points", xytext=(8, 8), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff", ec=cen["cor"]))
    ax.axvline(cen["pct"], color=cen["cor"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("% das Falhas Replantáveis (acumulado)")
    ax.set_ylabel("Fator de Otimização (%)")
    ax.set_title("Curva de Otimização de Replantio", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return fig_para_b64(fig)


def gerar(gdf_linhas, gdf_falhas, pasta, falha_min_rep=2.0,
          arquivo="relatorio_replantio.html"):
    """Gera o relatório HTML (3 cenários) e exporta as camadas do ÓTIMO.

    Retorna (caminho_html, gdf_linhas_rep, gdf_falhas_rep) ou None se não
    houver falhas replantáveis.
    """
    res = analisar(gdf_linhas, gdf_falhas, falha_min_rep)
    if res is None:
        return None
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    n_linhas_tot = len(gdf_linhas)
    n_falhas_tot = len(gdf_falhas)
    plantio_km = gdf_linhas["comprimento_m"].sum() / 1000

    paginas = []
    for i, cen in enumerate(res["cenarios"], start=1):
        curva = _curva(res, cen)
        frase = (f"Percorrendo apenas <b>{cen['k'] + 1} fileiras</b> "
                 f"({fmt(cen['comp_linhas_m']/1000, 2)} km), você elimina "
                 f"<b>{fmt(cen['pct'])}%</b> das falhas replantáveis — "
                 f"vantagem de <b>{fmt(cen['fator'])} pontos</b> sobre sair "
                 f"replantando sem priorização.")
        cab = f"""
<div class="frase" style="border-left:8px solid {cen['cor']}">{frase}</div>"""
        corpo = cab + f"""
<div class="grade">
  <div class="painel"><div class="bloco-tit" style="background:{cen['cor']}">CURVA DE OTIMIZAÇÃO</div>
    <img class="mapa" src="data:image/png;base64,{curva}"></div>
  <div>
    <div class="painel"><div class="cards">
      {card("Total linhas", n_linhas_tot, "Todas as fileiras mapeadas na área.")}{card("Total falhas", n_falhas_tot, "Todas as falhas, de qualquer tamanho.")}
      {card("Comp. linhas", fmt(plantio_km, 2) + " km", "Extensão total do plantio.")}
      {card("Comp. falhas replant.", fmt(res['total_rep_m']/1000, 3) + " km", "Só as falhas grandes o bastante p/ valer replantio (≥ falha mínima).")}
    </div>
    <div class="faixa"><span>FALHA MÍNIMA P/ REPLANTIO</span><span>{fmt(res['falha_min_rep'])} m</span></div></div>
    <div class="painel" style="margin-top:12px"><div class="bloco-tit">DADOS DO CENÁRIO</div>
      <div class="cards">
        {card("Linhas atuação", cen['k'] + 1, "Quantas fileiras a equipe percorre neste cenário.")}{card("Comp. linhas", fmt(cen['comp_linhas_m']/1000, 3) + " km", "Distância total percorrida sobre as fileiras escolhidas.")}
        {card("Nº falhas", cen['n_falhas'], "Falhas replantáveis cobertas neste cenário.")}{card("Falhas (m)", fmt(cen['falhas_m'], 0) + " m", "Metros de falha corrigidos neste cenário.")}
        {card("Falhas (km)", fmt(cen['falhas_m']/1000, 3) + " km", "O mesmo valor, em quilômetros.")}{card("% falhas", fmt(cen['pct']) + "%", "Fatia do problema total resolvida por este cenário.")}
      </div>
      <div class="fator" data-dica="Vantagem, em pontos percentuais, sobre sair replantando as fileiras sem nenhuma priorização."><span>FATOR DE OTIMIZAÇÃO</span>
        <span class="v">+{fmt(cen['fator'])}%</span><span style="font-size:11px;
        background:#cfeed3;padding:3px 8px;border-radius:3px">Vantagem</span></div>
    </div>
  </div>
</div>
<div style="font-size:10px;color:#78909c;margin-top:8px">Falha mínima p/
 replantio: {fmt(res['falha_min_rep'])} m | {SUBTITULOS[cen['titulo']]} |
 fator = % de falhas cobertas − % de fileiras percorridas. As fileiras do
 cenário recomendado estão em resultado_replantio.gpkg (abre no QGIS).</div>"""
        paginas.append(pagina("PLANO DE REPLANTIO — " + cen["titulo"],
                              "Onde agir primeiro para render mais",
                              corpo, i, 3, agora))

    html = ("<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Relatório de Replantio — CanaScan</title>{CSS}</head><body>"
            + "".join(paginas) + "</body></html>")
    caminho = os.path.join(pasta, arquivo)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html)

    otimo = res["cenarios"][1]
    gdf_lr = gdf_linhas[gdf_linhas["linha_id"].isin(otimo["linhas_ids"])].copy()
    gdf_fr = res["fr"][res["fr"]["linha_id"].isin(otimo["linhas_ids"])].copy()
    return caminho, gdf_lr, gdf_fr
