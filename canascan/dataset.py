"""Fábrica de dataset para treino da rede neural (segmentação).

Fluxo do usuário:
1. Processa o ortomosaico no CanaScan (gera resultado.gpkg).
2. Abre resultado.gpkg no QGIS, CORRIGE as linhas/falhas na camada.
3. Roda este módulo apontando imagem + o gpkg revisado.
   -> gera recortes (tiles) da imagem + máscaras alinhadas, prontos p/ treino.

A máscara tem 3 classes por pixel:
  0 = fundo (solo, mato, carreador)
  1 = linha de plantio (faixa ao longo da fileira)
  2 = falha (trecho de fileira sem planta)

Cada tile vira um par (imagem.png, mascara.png). O conjunto é dividido em
treino/validação e descrito num dataset.json — formato padrão que qualquer
framework de segmentação (PyTorch/TensorFlow) consome.
"""
import json
import os
import random
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import LineString
import geopandas as gpd


def _carregar_camada(gpkg, nomes):
    for n in nomes:
        try:
            g = gpd.read_file(gpkg, layer=n)
            if len(g):
                return g
        except Exception:
            continue
    return None


def _linhas_para_faixas(gdf, largura_m):
    """Transforma linhas em faixas (buffer) da largura da touceira."""
    if gdf is None or not len(gdf):
        return None
    return gdf.geometry.buffer(largura_m / 2.0, cap_style=2)


def gerar_dataset(tif, gpkg, saida, tile=512, sobrepos=64,
                  larg_linha=0.40, larg_falha=0.40, val_frac=0.2,
                  min_conteudo=0.01, seed=42):
    """Gera tiles de imagem + máscara a partir do ortomosaico e do gpkg revisado.

    tile: lado do recorte em pixels. sobrepos: sobreposição entre tiles.
    larg_linha/larg_falha: largura da faixa desenhada na máscara (m).
    min_conteudo: descarta tile com menos que esta fração de linha+falha.
    """
    os.makedirs(saida, exist_ok=True)
    dir_img = os.path.join(saida, "images")
    dir_msc = os.path.join(saida, "masks")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_msc, exist_ok=True)
    rng = random.Random(seed)

    linhas = _carregar_camada(gpkg, ["linhas_plantio", "linhas"])
    falhas = _carregar_camada(gpkg, ["falhas"])

    with rasterio.open(tif) as src:
        crs = src.crs
        if linhas is not None and linhas.crs != crs:
            linhas = linhas.to_crs(crs)
        if falhas is not None and falhas.crs != crs:
            falhas = falhas.to_crs(crs)
        faixa_linha = _linhas_para_faixas(linhas, larg_linha)
        faixa_falha = _linhas_para_faixas(falhas, larg_falha)

        W, H = src.width, src.height
        passo = tile - sobrepos
        registros = []
        idx = 0
        for y0 in range(0, H, passo):
            for x0 in range(0, W, passo):
                w = min(tile, W - x0)
                h = min(tile, H - y0)
                if w < tile // 2 or h < tile // 2:
                    continue
                jan = Window(x0, y0, w, h)
                arr = src.read(indexes=[1, 2, 3], window=jan)
                # pula tile quase vazio (sem imagem)
                if arr.max() == 0:
                    continue
                tr = src.window_transform(jan)

                mascara = np.zeros((h, w), dtype=np.uint8)
                formas = []
                if faixa_linha is not None:
                    formas += [(g, 1) for g in faixa_linha.values
                               if g is not None and not g.is_empty]
                if faixa_falha is not None:
                    formas += [(g, 2) for g in faixa_falha.values
                               if g is not None and not g.is_empty]
                if formas:
                    mascara = rasterize(
                        formas, out_shape=(h, w), transform=tr,
                        fill=0, dtype=np.uint8, all_touched=True)

                cobertura = float((mascara > 0).mean())
                if cobertura < min_conteudo:
                    continue

                img = np.transpose(arr, (1, 2, 0))
                nome = f"tile_{idx:05d}"
                _salvar_png(os.path.join(dir_img, nome + ".png"), img)
                _salvar_png(os.path.join(dir_msc, nome + ".png"),
                            mascara, paleta=True)
                registros.append({
                    "id": nome, "x": x0, "y": y0, "w": w, "h": h,
                    "cobertura": round(cobertura, 4),
                    "pct_linha": round(float((mascara == 1).mean()), 4),
                    "pct_falha": round(float((mascara == 2).mean()), 4)})
                idx += 1

    rng.shuffle(registros)
    n_val = int(len(registros) * val_frac)
    for i, r in enumerate(registros):
        r["split"] = "val" if i < n_val else "train"

    meta = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "fonte_imagem": os.path.basename(tif),
        "fonte_rotulos": os.path.basename(gpkg),
        "crs": str(crs), "tile": tile, "sobreposicao": sobrepos,
        "classes": {"0": "fundo", "1": "linha", "2": "falha"},
        "larg_linha_m": larg_linha, "larg_falha_m": larg_falha,
        "n_tiles": len(registros),
        "n_treino": len(registros) - n_val, "n_val": n_val,
        "tiles": registros}
    with open(os.path.join(saida, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _salvar_png(caminho, arr, paleta=False):
    from PIL import Image
    if paleta:  # máscara com cores visíveis p/ conferência
        im = Image.fromarray(arr, mode="P")
        pal = [60, 60, 60] + [255, 235, 0] + [220, 30, 30]
        pal += [0] * (768 - len(pal))
        im.putpalette(pal)
    else:
        im = Image.fromarray(arr.astype(np.uint8))
    im.save(caminho)


def acrescentar(tif, gpkg, dataset_dir, **kw):
    """Adiciona os tiles de um NOVO ortomosaico a um dataset existente,
    reindexando para não sobrescrever e atualizando o dataset.json."""
    principal = os.path.join(dataset_dir, "dataset.json")
    if not os.path.isfile(principal):
        return gerar_dataset(tif, gpkg, dataset_dir, **kw)

    with open(principal, encoding="utf-8") as f:
        antigo = json.load(f)
    base = antigo["n_tiles"]

    tmp = os.path.join(dataset_dir, "_tmp_add")
    novo = gerar_dataset(tif, gpkg, tmp, **kw)

    import shutil
    for r in novo["tiles"]:
        antigo_id = r["id"]
        novo_id = f"tile_{base:05d}"
        base += 1
        for sub in ("images", "masks"):
            shutil.move(os.path.join(tmp, sub, antigo_id + ".png"),
                        os.path.join(dataset_dir, sub, novo_id + ".png"))
        r["id"] = novo_id
        r["fonte"] = os.path.basename(tif)
        antigo["tiles"].append(r)
    shutil.rmtree(tmp, ignore_errors=True)

    antigo["n_tiles"] = len(antigo["tiles"])
    n_val = sum(1 for t in antigo["tiles"] if t["split"] == "val")
    antigo["n_val"] = n_val
    antigo["n_treino"] = antigo["n_tiles"] - n_val
    antigo["atualizado_em"] = datetime.now().isoformat(timespec="seconds")
    antigo.setdefault("fontes", [])
    antigo["fontes"].append(os.path.basename(tif))
    with open(principal, "w", encoding="utf-8") as f:
        json.dump(antigo, f, indent=2, ensure_ascii=False)
    return antigo


gerar_fn = gerar_dataset
acrescentar_fn = acrescentar

def gerar_de_shapefiles(tif, shp_linhas, shp_falhas, saida, acrescentar=False,
                        tile=512, sobrepos=64, larg_linha=0.40, larg_falha=0.40,
                        val_frac=0.2, min_conteudo=0.01, seed=42):
    """Gera tiles a partir de shapefiles externos de linhas e falhas.

    Ideal para projetos no formato Nelore/X-Farming onde cada talhão tem
    seus próprios .shp de linha e falha. Aceita string ou lista de caminhos.
    """
    import tempfile, shutil
    import geopandas as gpd

    def carregar_shps(caminhos):
        if not caminhos:
            return None
        if isinstance(caminhos, str):
            caminhos = [caminhos]
        gdfs = [gpd.read_file(c) for c in caminhos if c and os.path.isfile(c)]
        if not gdfs:
            return None
        merged = gpd.pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
        # mantém só geometry: evita conflito de FID/ID ao salvar no GPKG
        return merged[["geometry"]].reset_index(drop=True)

    gdf_l = carregar_shps(shp_linhas)
    gdf_f = carregar_shps(shp_falhas)
    if gdf_l is None or not len(gdf_l):
        print(f"  [aviso] sem linhas — pulando {os.path.basename(tif)}")
        return None

    tmp = tempfile.mkdtemp()
    gpkg_tmp = os.path.join(tmp, "tmp.gpkg")
    try:
        with rasterio.open(tif) as src:
            crs = src.crs

        def alinhar(g):
            """Corrige CRS rotulado como geográfico mas com coordenadas
            métricas (exportação comum de softwares de usina). Sem isso o
            to_crs desloca tudo e o projeto rende ZERO tiles."""
            if g is None or not len(g):
                return g
            if g.crs is not None and g.crs.is_geographic:
                b = g.total_bounds
                if np.isfinite(b).all() and np.abs(b).max() > 360:
                    return g.set_crs(crs, allow_override=True)
            if g.crs is not None and crs is not None and g.crs != crs:
                return g.to_crs(crs)
            if g.crs is None:
                return g.set_crs(crs, allow_override=True)
            return g

        gdf_l = alinhar(gdf_l)
        gdf_f = alinhar(gdf_f)
        gdf_l = gdf_l.reset_index(drop=True)
        gdf_l.to_file(gpkg_tmp, layer="linhas_plantio", driver="GPKG")
        if gdf_f is not None and len(gdf_f):
            gdf_f = gdf_f.reset_index(drop=True)
            gdf_f.to_file(gpkg_tmp, layer="falhas", driver="GPKG")
        existe = acrescentar and os.path.isfile(
            os.path.join(saida, "dataset.json"))
        fn = acrescentar_fn if existe else gerar_fn
        return fn(tif, gpkg_tmp, saida, tile=tile, sobrepos=sobrepos,
                  larg_linha=larg_linha, larg_falha=larg_falha,
                  val_frac=val_frac, min_conteudo=min_conteudo, seed=seed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
