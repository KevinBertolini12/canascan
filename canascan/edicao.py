"""Sistema de edições — produtos separados, núcleo ÚNICO.

O arquivo edicao.json (na pasta do programa) define qual produto este
executável é. Sem o arquivo, roda a edição completa (desenvolvimento).
Futuramente a licença assinada definirá a edição — este módulo já é o
ponto único de verdade que ela vai alimentar.

Edições:
  linha    -> LinhaScan: só restituição de linhas
  pro      -> CanaScan Pro: linhas + falhas + replantio + comparação
  completo -> CanaScan Completo: tudo, incluindo daninhas
"""
import json
import os
import sys

MODULOS = {
    "linhas":   "Restituição de Linhas",
    "falhas":   "Falhas + Replantio",
    "comparar": "Comparação com referência",
    "daninhas": "Detecção de Daninhas",
}

EDICOES = {
    "linha":    {"nome": "LinhaScan", "modulos": {"linhas"}},
    "pro":      {"nome": "CanaScan Pro",
                 "modulos": {"linhas", "falhas", "comparar"}},
    "completo": {"nome": "CanaScan Completo",
                 "modulos": {"linhas", "falhas", "comparar", "daninhas"}},
}


def _carregar():
    if getattr(sys, "frozen", False):
        raiz = os.path.dirname(sys.executable)
    else:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(raiz, "edicao.json")
    chave = "completo"
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                chave = json.load(f).get("edicao", "completo")
        except Exception:
            pass
    ed = EDICOES.get(chave, EDICOES["completo"])
    return chave, ed["nome"], set(ed["modulos"])


CHAVE, NOME, ATIVOS = _carregar()


def tem(modulo):
    """True se o módulo está liberado nesta edição."""
    return modulo in ATIVOS
