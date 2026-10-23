"""Ponto de entrada do executável CanaScan.

Sem argumentos (duplo clique): abre a interface gráfica.
Com subcomando (processar/sintetico/comparar/dataset): roda a CLI —
é assim que a própria GUI se chama por baixo dos panos.
"""
import sys

SUBCOMANDOS = {"processar", "sintetico", "comparar", "dataset", "-h", "--help"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMANDOS:
        from canascan import cli
        cli.main()
    else:
        from canascan.gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
