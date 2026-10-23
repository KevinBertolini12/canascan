"""CanaScan — linha de comando.

Uso:
  python -m canascan.cli sintetico --saida dados
  python -m canascan.cli processar --tif orto.tif --saida resultados \
      [--talhoes talhoes.gpkg] [--linhas linhas.gpkg] [--metodo unet] \
      [--espacamento 1.5] [--falha-min 0.5] [--replantio 2.0] [--verdade v.gpkg]
"""
import argparse
import os
import time

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

LIMITE_GB_IMAGEM_UNICA = 0.5  # acima disso, ativa a grade automática

from . import __version__, geo, pipeline, relatorio
from . import talhoes as mod_talhoes


def cmd_sintetico(args):
    from . import sintetico
    resumo, tif, gpkg = sintetico.gerar(
        args.saida, largura_m=args.largura, altura_m=args.altura,
        gsd=args.gsd, espacamento=args.espacamento,
        angulo_graus=args.angulo, seed=args.seed)
    print("Campo sintético gerado:")
    for k, v in resumo.items():
        print(f"  {k}: {v}")
    print(f"  -> {tif}")
    print(f"  -> {gpkg} (gabarito)")
    print(f"  -> {os.path.join(args.saida, 'talhoes.gpkg')} (3 talhões de teste)")


def _carregar_linhas_usuario(caminho, crs_destino):
    try:
        gdf = gpd.read_file(caminho, layer="linhas_plantio")
    except Exception:
        gdf = gpd.read_file(caminho)
    if gdf.crs is not None and crs_destino is not None and gdf.crs != crs_destino:
        gdf = gdf.to_crs(crs_destino)
    return gdf


def _grade_automatica(info, tam_m):
    """Cobre o raster com blocos quadrados de tam_m x tam_m (pseudo-talhões)."""
    tr, W, H = info["transform"], info["largura"], info["altura"]
    xs = [tr.c, tr.c + tr.a * W + tr.b * H]
    ys = [tr.f, tr.f + tr.d * W + tr.e * H]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    polys, n = [], 1
    for yy in np.arange(miny, maxy, tam_m):
        for xx in np.arange(minx, maxx, tam_m):
            polys.append({"talhao": f"B{n}", "geometry": box(
                xx, yy, min(xx + tam_m, maxx), min(yy + tam_m, maxy))})
            n += 1
    return gpd.GeoDataFrame(polys, crs=info["crs"])


def _explodir_linhas(geoms):
    saida = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            saida.append(g)
        elif g.geom_type == "MultiLineString":
            saida.extend(list(g.geoms))
    return saida


def cmd_processar(args):
    from . import edicao
    print(f"{edicao.NOME} v{__version__}")
    if args.daninhas and not edicao.tem("daninhas"):
        print("[edição] módulo Daninhas não incluído nesta edição — ignorando")
        args.daninhas = False
    if args.replantio and not edicao.tem("falhas"):
        args.replantio = None
    t0 = time.time()
    os.makedirs(args.saida, exist_ok=True)
    info = mod_talhoes.info_raster(args.tif)
    crs = info["crs"]
    print(f"[raster] {info['largura']}x{info['altura']} px | GSD {info['gsd']*100:.1f} cm/px "
          f"| CRS {crs} | ~{info['gb_abertos']:.2f} GB abertos")

    linhas_usuario = None
    if args.linhas:
        linhas_usuario = _carregar_linhas_usuario(args.linhas, crs)
        print(f"[linhas] {len(linhas_usuario)} linhas de plantio fornecidas — detecção pulada")

    params = dict(espacamento=args.espacamento, metodo=args.metodo,
                  limiar=args.limiar, falha_min=args.falha_min,
                  comp_copa=args.comp_copa, banda_m=args.banda,
                  min_linha=args.min_linha, sensibilidade=args.sensibilidade,
                  analisar_daninhas=args.daninhas,
                  adaptativo=args.adaptativo, tile_m=args.tile)

    resultados_globais, por_talhao, gdf_t = [], [], None
    overlay_pendente = None
    dan_polys, dan_area = [], 0.0
    max_cob = 0.0

    if args.talhoes:
        _gdf = mod_talhoes.carregar_talhoes(args.talhoes, crs)
        gdf_t = _gdf
        print(f"[talhões] {len(gdf_t)} talhões — processando um por vez "
              f"(memória de 1 talhão, área total ilimitada)")
    elif not args.sem_grade and (args.grade is not None
                                 or info["gb_abertos"] > LIMITE_GB_IMAGEM_UNICA):
        tam = args.grade if args.grade is not None else 250.0
        gdf_t = _grade_automatica(info, tam)
        print(f"[grade] imagem grande ({info['gb_abertos']:.1f} GB abertos): "
              f"grade automática de {tam:.0f}x{tam:.0f} m — {len(gdf_t)} blocos, "
              f"um por vez na memória")

    if gdf_t is not None:
        for _, row in gdf_t.iterrows():
            nome, geom = row["talhao"], row.geometry
            t1 = time.time()
            jan = mod_talhoes.ler_janela(args.tif, geom, pad_m=args.pad)
            if jan is None:
                print(f"  [talhão {nome}] fora do raster — pulado")
                continue
            rgb, valido, tr, _, gsd = jan
            lg = None
            if linhas_usuario is not None:
                sel = linhas_usuario[linhas_usuario.intersects(geom)]
                lg = _explodir_linhas(sel.intersection(geom).values)
            res, lin_px, ang, cob, dan = pipeline.processar_area(
                rgb, valido, tr, gsd, linhas_geo=lg, **params)
            for r in res:
                r["talhao"] = nome
            resultados_globais.extend(res)
            if dan:
                dan_polys.extend(dan["poligonos"]); dan_area += dan["area_m2"]
            max_cob = max(max_cob, cob or 0.0)
            pm = sum(r["comprimento_m"] for r in res)
            fm = sum(r["falha_m"] for r in res)
            if res:
                por_talhao.append({"talhao": nome, "n_linhas": len(res), "plantio": pm,
                                   "falha": fm, "pct": 100 * fm / pm if pm else 0})
            if args.overlays:
                relatorio.salvar_overlay(rgb, res, lin_px,
                    os.path.join(args.saida, f"overlay_talhao_{nome}.png"))
            if not res:
                print(f"  [{nome}] sem vegetação/linhas | {time.time()-t1:.1f} s")
            else:
                print(f"  [{nome}] {len(res)} linhas | {pm:,.0f} m | "
                      f"falha {fm:,.1f} m ({100*fm/pm if pm else 0:.2f}%) | "
                      f"verde {100*cob:.0f}% | {time.time()-t1:.1f} s")
            del rgb, valido
    else:
        print("[modo] imagem única (sem talhões) — carregando raster inteiro")
        rgb, valido, tr, crs, gsd = geo.carregar_rgb(args.tif)
        lg = _explodir_linhas(linhas_usuario.geometry.values) if linhas_usuario is not None else None
        res, lin_px, ang, cob, dan = pipeline.processar_area(
            rgb, valido, tr, gsd, linhas_geo=lg, **params)
        for r in res:
            r["talhao"] = "1"
        resultados_globais = res
        if dan:
            dan_polys.extend(dan["poligonos"]); dan_area += dan["area_m2"]
        max_cob = max(max_cob, cob or 0.0)
        pm = sum(r["comprimento_m"] for r in res)
        fm = sum(r["falha_m"] for r in res)
        por_talhao.append({"talhao": "1", "n_linhas": len(res), "plantio": pm,
                           "falha": fm, "pct": 100 * fm / pm if pm else 0})
        if ang is not None:
            print(f"  orientação {ang:.2f}° | cobertura verde {100*cob:.1f}%")
        overlay_pendente = (rgb, res, lin_px)  # desenhado só DEPOIS de salvar os dados

    if not resultados_globais:
        print("Nenhuma linha detectada — verifique o método/limiar de vegetação.")
        return

    # ids globais (por talhão os ids locais se repetem)
    for i, r in enumerate(resultados_globais, start=1):
        r["id"] = i

    gdf_linhas, gdf_falhas = relatorio.montar_camadas(resultados_globais, crs)
    gpkg = relatorio.salvar_gpkg(gdf_linhas, gdf_falhas, args.saida, gdf_t)

    res_glob = relatorio.resumo_global(resultados_globais)
    relatorio.salvar_txt(res_glob, resultados_globais, args.saida,
                         por_talhao=por_talhao)
    relatorio.salvar_histograma(res_glob, args.saida)

    if dan_polys:
        import geopandas as _g
        gdf_dan = _g.GeoDataFrame(geometry=dan_polys, crs=crs)
        gdf_dan.to_file(os.path.join(args.saida, "resultado.gpkg"),
                        layer="daninhas", driver="GPKG")
        print(f"[daninhas] {len(dan_polys)} manchas | {dan_area:,.1f} m² "
              f"-> camada 'daninhas' no resultado.gpkg")

    try:
        c1, c2 = relatorio.salvar_csv_usina(gdf_linhas, gdf_falhas,
                                            args.saida, gdf_t)
        print(f"[csv] {os.path.basename(c1)} e {os.path.basename(c2)} "
              f"(formato de relatório de usina)")
    except Exception as e:
        print(f"[csv] não gerado: {type(e).__name__}: {e}")

    from . import relatorio_html
    html = relatorio_html.gerar(gdf_linhas, gdf_falhas, gdf_t, args.saida,
                                falha_min=args.falha_min,
                                daninhas_m2=dan_area if dan_polys else None,
                                daninhas_n=len(dan_polys) if dan_polys else None)
    print(f"[relatório] {html}")

    if overlay_pendente is not None:
        try:
            relatorio.salvar_overlay(*overlay_pendente,
                                     os.path.join(args.saida, "overlay.png"))
            print(f"[overlay] {os.path.join(args.saida, 'overlay.png')}")
        except Exception as e:
            print(f"[overlay] não gerado ({type(e).__name__}) — dados salvos normalmente")

    if args.replantio:
        from . import replantio as mod_rep
        rep = mod_rep.gerar(gdf_linhas, gdf_falhas, args.saida,
                            falha_min_rep=args.replantio)
        if rep is None:
            print(f"[replantio] nenhuma falha >= {args.replantio} m — relatório não gerado")
        else:
            caminho, gdf_lr, gdf_fr = rep
            gpkg_r = os.path.join(args.saida, "resultado_replantio.gpkg")
            gdf_lr.to_file(gpkg_r, layer="linhas_replantio", driver="GPKG")
            if len(gdf_fr):
                gdf_fr.to_file(gpkg_r, layer="falhas_replantio", driver="GPKG")
            print(f"[replantio] {caminho}")
            print(f"[replantio] camadas do cenário ótimo -> {gpkg_r}")

    dt = time.time() - t0
    print("\n" + "=" * 48)
    print(f"  Linhas: {res_glob['n_linhas']}  |  Comprimento: {res_glob['comprimento_total_m']:,.0f} m")
    print(f"  Falhas: {res_glob['n_falhas']}  |  {res_glob['falha_total_m']:,.1f} m")
    print(f"  ÍNDICE DE FALHAS: {res_glob['pct_falha']:.2f} %")
    print(f"  Tempo total: {dt:.1f} s")
    print(f"  Saída: {args.saida}/")
    print("=" * 48)

    if args.verdade:
        avaliar(args.verdade, gpkg, res_glob)


def cmd_dataset(args):
    print(f"CanaScan v{__version__} — fábrica de dataset")
    from . import dataset as mod_ds
    existe = os.path.isfile(os.path.join(args.saida, "dataset.json"))
    fn = mod_ds.acrescentar if (existe and args.acrescentar) else mod_ds.gerar_dataset
    meta = fn(args.tif, args.gpkg, args.saida, tile=args.tile,
              sobrepos=args.sobreposicao, larg_linha=args.larg_linha,
              larg_falha=args.larg_falha, val_frac=args.val)
    print("\n" + "=" * 48)
    print(f"  Tiles gerados ....... {meta['n_tiles']}")
    print(f"  Treino / validação .. {meta['n_treino']} / {meta['n_val']}")
    print(f"  Classes ............. fundo / linha / falha")
    print(f"  Pasta ............... {args.saida}/")
    print(f"    images/  masks/  dataset.json")
    print("=" * 48)
    print("  Confira algumas máscaras (masks/*.png): amarelo=linha, "
          "vermelho=falha.")
    print("  Cada novo ortomosaico revisado: rode de novo com --acrescentar")


def cmd_comparar(args):
    from . import edicao
    if not edicao.tem("comparar"):
        print(f"{edicao.NOME}: o módulo Comparação não está incluído "
              f"nesta edição.")
        return
    print(f"{edicao.NOME} v{__version__} — comparação")
    from . import comparar as mod_cmp
    nosso_l = mod_cmp._ler_camada(args.nosso + "::linhas_plantio")
    try:
        nosso_f = mod_cmp._ler_camada(args.nosso + "::falhas")
    except Exception:
        import geopandas as _g
        nosso_f = _g.GeoDataFrame(geometry=[], crs=nosso_l.crs)
    crs = nosso_l.crs
    ref_l = mod_cmp._ler_camada(args.ref_linhas, crs) if args.ref_linhas else None
    ref_f = mod_cmp._ler_camada(args.ref_falhas, crs) if args.ref_falhas else None
    if ref_f is not None and not mod_cmp._so_linhas(ref_f):
        print("[aviso] falhas de referência não são linhas — metragens podem divergir")
    caminho, m, (titulo, texto) = mod_cmp.gerar(
        nosso_l, nosso_f, ref_l, ref_f, args.saida,
        tol_falhas=args.tolerancia, tol_linhas=args.tolerancia,
        nome_ref=args.nome_ref)
    print("\n" + "=" * 48)
    print(f"  {titulo}")
    print(f"  {texto}")
    f = m.get("falhas")
    if f:
        print(f"  Falhas: nosso {f['m_nosso']:,.1f} m | ref {f['m_ref']:,.1f} m "
              f"| dif {f['dif_pct']:+.1f}%")
        print(f"  Recall {f['recall']:.1f}% | Precisão {f['precisao']:.1f}%")
    li = m.get("linhas")
    if li and li.get("desvio"):
        d = li["desvio"]
        if li and li.get("m_ref"):
            dif = 100.0 * (li["m_nosso"] - li["m_ref"]) / li["m_ref"]
            print(f"  Plantio: nosso {li['m_nosso']/1000:,.1f} km | "
                  f"ref {li['m_ref']/1000:,.1f} km | dif {dif:+.1f}%")
        print(f"  Linhas: desvio lateral mediano {100*d['mediana']:.0f} cm "
              f"| p90 {100*d['p90']:.0f} cm | ≤25cm: {d['dentro_25cm']:.0f}%")
    print(f"  Relatório: {caminho}")
    print("=" * 48)


def avaliar(gpkg_verdade, gpkg_detec, res):
    ver = gpd.read_file(gpkg_verdade, layer="falhas")
    det = gpd.read_file(gpkg_detec, layer="falhas")
    ver_u, det_u = unary_union(ver.geometry), unary_union(det.geometry)
    len_ver, len_det = ver_u.length, det_u.length
    tol = 0.5
    recall = ver_u.intersection(det_u.buffer(tol)).length / len_ver if len_ver else 0
    prec = det_u.intersection(ver_u.buffer(tol)).length / len_det if len_det else 0
    print("\n--- AVALIAÇÃO CONTRA O GABARITO ---")
    print(f"  Falha real (gabarito): {len_ver:,.1f} m")
    print(f"  Falha detectada:       {len_det:,.1f} m")
    print(f"  Recall  (quanto do real foi achado): {100*recall:.1f}%")
    print(f"  Precisão (quanto do detectado é real): {100*prec:.1f}%")


def main():
    try:  # log em tempo real e acentos corretos mesmo empacotado (.exe)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
    except Exception:
        pass
    import builtins as _b            # flush garantido em TODO print,
    _orig_print = _b.print           # inclusive dos módulos internos —
    def _print_flush(*a, **k):       # indispensável no .exe (buffer de pipe)
        k.setdefault("flush", True)
        _orig_print(*a, **k)
    _b.print = _print_flush
    p = argparse.ArgumentParser(prog="canascan")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sintetico", help="gera campo de teste com gabarito e talhões")
    s.add_argument("--saida", default="dados_sinteticos")
    s.add_argument("--largura", type=float, default=120.0)
    s.add_argument("--altura", type=float, default=80.0)
    s.add_argument("--gsd", type=float, default=0.05)
    s.add_argument("--espacamento", type=float, default=1.5)
    s.add_argument("--angulo", type=float, default=18.0)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sintetico)

    r = sub.add_parser("processar", help="roda o pipeline completo num GeoTIFF")
    r.add_argument("--tif", required=True)
    r.add_argument("--saida", default="resultados")
    r.add_argument("--talhoes", default=None,
                   help="polígonos dos talhões (GPKG/SHP) — ativa o modo sem limite de área")
    r.add_argument("--linhas", default=None,
                   help="linhas de plantio prontas (GPKG/SHP) — pula a detecção")
    r.add_argument("--espacamento", type=float, default=1.5,
                   help="espaçamento entre linhas (m)")
    r.add_argument("--metodo", choices=["unet", "razao", "gli", "exg_otsu"], default="unet",
                   help="índice de vegetação; 'gli' equivale ao Limiar GLI do NeloreTech, 'unet' usa a IA")
    r.add_argument("--limiar", type=float, default=None,
                   help="limiar do índice (padrão: 1.5 p/ razao, 0.02 p/ gli)")
    r.add_argument("--falha-min", type=float, default=0.5,
                   help="comprimento mínimo p/ contar como falha (m) — Stolf")
    r.add_argument("--comp-copa", type=float, default=0.25, dest="comp_copa",
                   help="compensação de copa por extremidade da falha (m)")
    r.add_argument("--daninhas", action="store_true",
                   help="analisa daninhas na entrelinha (vegetação fora das linhas)")
    r.add_argument("--sensibilidade", type=float, default=1.0,
                   help="0,5=conservador … 2,0=agressivo (voo alto pede >1)")
    r.add_argument("--min-linha", type=float, default=5.0, dest="min_linha",
                   help="descarta linhas detectadas menores que isto (m)")
    r.add_argument("--banda", type=float, default=4.0,
                   help="passo do rastreador de linhas (m)")
    r.add_argument("--adaptativo", action="store_true",
                   help="detecção com ângulo/espaçamento LOCAIS por tile — "
                        "segue plantio em curva de nível (mais lento, mais preciso)")
    r.add_argument("--tile", type=float, default=35.0,
                   help="tamanho do tile local no modo adaptativo (m); "
                        "menor = segue curva mais fechada, mais lento")
    r.add_argument("--grade", type=float, nargs="?", const=250.0, default=None,
                   help="força grade automática de blocos (m); sem valor = 250")
    r.add_argument("--sem-grade", action="store_true", dest="sem_grade",
                   help="desativa a grade automática (força imagem única)")
    r.add_argument("--pad", type=float, default=6.0,
                   help="folga da janela ao redor de cada talhão (m)")
    r.add_argument("--replantio", type=float, default=None,
                   help="falha mínima p/ replantio (m) — gera relatório de otimização")
    r.add_argument("--overlays", action="store_true",
                   help="gera overlay PNG por talhão (mais lento)")
    r.add_argument("--verdade", default=None,
                   help="GPKG de gabarito p/ avaliação (opcional)")
    r.set_defaults(func=cmd_processar)

    c = sub.add_parser("comparar",
                       help="compara o resultado do CanaScan com a referência (X-Farming etc.)")
    c.add_argument("--nosso", required=True,
                   help="resultado.gpkg gerado pelo CanaScan")
    c.add_argument("--ref-linhas", default=None, dest="ref_linhas",
                   help="linhas da referência (shp/gpkg; use arq.gpkg::camada)")
    c.add_argument("--ref-falhas", default=None, dest="ref_falhas",
                   help="falhas da referência (shp/gpkg; use arq.gpkg::camada)")
    c.add_argument("--saida", default="comparacao")
    c.add_argument("--tolerancia", type=float, default=0.5,
                   help="tolerância espacial em metros (padrão 0,5)")
    c.add_argument("--nome-ref", default="Referência", dest="nome_ref",
                   help="nome da referência p/ o relatório (ex.: X-Farming)")
    c.set_defaults(func=cmd_comparar)

    d = sub.add_parser("dataset",
                       help="gera pares imagem+máscara p/ treino da rede neural")
    d.add_argument("--tif", required=True, help="ortomosaico (GeoTIFF)")
    d.add_argument("--gpkg", required=True,
                   help="resultado.gpkg REVISADO no QGIS (linhas/falhas corrigidas)")
    d.add_argument("--saida", default="dataset")
    d.add_argument("--tile", type=int, default=512, help="lado do recorte (px)")
    d.add_argument("--sobreposicao", type=int, default=64)
    d.add_argument("--larg-linha", type=float, default=0.40, dest="larg_linha",
                   help="largura da faixa de linha na máscara (m)")
    d.add_argument("--larg-falha", type=float, default=0.40, dest="larg_falha")
    d.add_argument("--val", type=float, default=0.2,
                   help="fração para validação (0.2 = 20 por cento)")
    d.add_argument("--acrescentar", action="store_true",
                   help="adiciona a um dataset existente em vez de recriar")
    d.set_defaults(func=cmd_dataset)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()