"""Script de lote — converte projetos da usina em dataset de treino.

Estrutura esperada de cada projeto:
  PASTA_PROJETO/
    ORTOMOSAICO.tif                          <- imagem do drone
    cana/
      linhas_propagadas/
        linha_propagada_7.shp (+ .dbf .prj .shx)   <- FILEIRAS reais
        linha_propagada_8.shp ...
      falhas/
        falha_7.shp (+ .dbf .prj .shx)
        falha_8.shp ...

Uso:
  python processar_usina_lote.py --raiz "C:/Projetos/Ortomosaico" --dataset meu_dataset
  python processar_usina_lote.py --raiz "C:/Projetos/Ortomosaico" --dataset meu_dataset --tile 256

O script varre todas as subpastas automaticamente, encontra os pares
TIF + shapefiles, gera os tiles e acrescenta tudo no mesmo dataset.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canascan.dataset import gerar_de_shapefiles


def encontrar_projetos(raiz):
    """Varre raiz procurando pastas com ORTOMOSAICO.tif + estrutura cana/."""
    projetos = []
    for dirpath, dirs, files in os.walk(raiz):
        tifs = [f for f in files if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
        if not tifs:
            continue
        # prefere ORTOMOSAICO.tif, senão pega o primeiro .tif encontrado
        principal = next((f for f in tifs if "ortomosaico" in f.lower()), tifs[0])
        tif = os.path.join(dirpath, principal)
        pasta_cana = os.path.join(dirpath, "cana")
        if not os.path.isdir(pasta_cana):
            continue
        projetos.append((dirpath, tif, pasta_cana))
    return sorted(projetos)


def shapefiles_de_pasta(pasta, prefixo):
    """Lista todos os .shp com determinado prefixo dentro de uma pasta."""
    return sorted(glob.glob(os.path.join(pasta, prefixo + "*.shp")))


def main():
    ap = argparse.ArgumentParser(
        description="Processa projetos da usina em lote e monta dataset de treino")
    ap.add_argument("--raiz", required=True,
                    help="Pasta raiz com todos os projetos (subpastas 3007_ANALISE, 3015_ANALISE...)")
    ap.add_argument("--dataset", default="dataset_usina",
                    help="Pasta de destino do dataset (padrão: dataset_usina)")
    ap.add_argument("--tile", type=int, default=512,
                    help="Tamanho do recorte em pixels (padrão: 512)")
    ap.add_argument("--sobreposicao", type=int, default=64)
    ap.add_argument("--larg-linha", type=float, default=0.40, dest="larg_linha")
    ap.add_argument("--larg-falha", type=float, default=0.40, dest="larg_falha")
    ap.add_argument("--val", type=float, default=0.2,
                    help="Fracao para validacao (padrao 0.2 = 20 por cento)")
    ap.add_argument("--min-conteudo", type=float, default=0.01, dest="min_conteudo",
                    help="Descarta tile com menos que esta fracao de linha+falha")
    ap.add_argument("--recomecar", action="store_true",
                    help="Apaga o dataset existente e começa do zero")
    args = ap.parse_args()

    # SEGURANÇA: nunca deixar o dataset ser (ou estar dentro de) a pasta dos
    # projetos — --recomecar apagaria os ortomosaicos do usuário.
    raiz_abs = os.path.abspath(args.raiz)
    ds_abs = os.path.abspath(args.dataset)
    if ds_abs == raiz_abs or ds_abs.startswith(raiz_abs + os.sep) \
            or raiz_abs.startswith(ds_abs + os.sep):
        print("\n[ERRO] A pasta do dataset não pode ser a mesma (nem estar")
        print("       dentro) da pasta dos projetos — apagaria seus arquivos.")
        print(f"       projetos: {raiz_abs}")
        print(f"       dataset:  {ds_abs}")
        print("\n       Use pastas separadas, por exemplo:")
        print("         --raiz    .../projetos")
        print("         --dataset .../meu_dataset")
        return

    if args.recomecar and os.path.isfile(os.path.join(args.dataset, "dataset.json")):
        import shutil
        shutil.rmtree(args.dataset)
        print(f"[recomeçar] dataset anterior apagado")

    projetos = encontrar_projetos(args.raiz)
    if not projetos:
        print(f"Nenhum projeto encontrado em: {args.raiz}")
        print("Verifique se as pastas têm ORTOMOSAICO.tif e a subpasta cana/")
        return

    print(f"\n{'='*55}")
    print(f"  Projetos encontrados: {len(projetos)}")
    print(f"  Dataset de destino:   {args.dataset}/")
    print(f"{'='*55}\n")

    total_tiles = 0
    erros = []

    for i, (pasta_proj, tif, pasta_cana) in enumerate(projetos, 1):
        nome_proj = os.path.basename(pasta_proj)
        falhas_dir = os.path.join(pasta_cana, "falhas")
        # PROPAGADAS são as fileiras de verdade (centenas por talhão).
        # PRINCIPAIS têm só o eixo mestre (1 por talhão) — serve apenas
        # como reserva; treinar com elas produz máscara quase vazia.
        shp_linhas = shapefiles_de_pasta(
            os.path.join(pasta_cana, "linhas_propagadas"), "linha_propagada")
        origem = "propagadas"
        if not shp_linhas:
            shp_linhas = shapefiles_de_pasta(
                os.path.join(pasta_cana, "linhas_principais"), "linha_principal")
            origem = "principais (reserva)"
        shp_falhas = shapefiles_de_pasta(falhas_dir, "falha")

        print(f"[{i:02d}/{len(projetos)}] {nome_proj}")
        print(f"      linhas: {len(shp_linhas)} shp ({origem}) | "
              f"falhas: {len(shp_falhas)} shp")

        if not shp_linhas:
            msg = f"sem shapefiles de linhas — pulando"
            print(f"      [aviso] {msg}")
            erros.append((nome_proj, msg))
            continue

        try:
            acrescentar = os.path.isfile(os.path.join(args.dataset, "dataset.json"))
            meta = gerar_de_shapefiles(
                tif, shp_linhas, shp_falhas, args.dataset,
                acrescentar=acrescentar,
                tile=args.tile, sobrepos=args.sobreposicao,
                larg_linha=args.larg_linha, larg_falha=args.larg_falha,
                val_frac=args.val, min_conteudo=args.min_conteudo)
            if meta is None:
                erros.append((nome_proj, "sem tiles gerados"))
                continue
            n = meta["n_tiles"]
            total_tiles = n
            print(f"      tiles acumulados: {n}")
        except Exception as e:
            msg = str(e)[:80]
            print(f"      [ERRO] {msg}")
            erros.append((nome_proj, msg))
        print()

    print(f"\n{'='*55}")
    print(f"  CONCLUÍDO")
    print(f"  Total de tiles no dataset: {total_tiles}")
    print(f"  Projetos com erro: {len(erros)}")
    if erros:
        print(f"\n  Erros:")
        for nome, msg in erros:
            print(f"    {nome}: {msg}")
    print(f"\n  Pasta: {args.dataset}/")
    print(f"  Confira masks/*.png: amarelo=linha, vermelho=falha")
    if total_tiles >= 200:
        print(f"\n  Dica: com {total_tiles} tiles você já pode pensar em treino!")
        print(f"  Veja GUIA_REDE_NEURAL.md para os próximos passos.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
