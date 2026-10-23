"""Restituição das linhas de plantio (o coração do sistema).

Estratégia:
1. Descobrir a orientação dominante das linhas (busca de ângulo que maximiza
   o contraste do perfil de projeção da máscara de vegetação).
2. Rotacionar a máscara para deixar as linhas verticais.
3. Dividir a imagem em bandas horizontais; em cada banda, o perfil de soma
   por coluna tem picos no centro de cada linha de cana.
4. Encadear os picos de banda em banda -> polilinhas (funciona com linhas
   curvas, tipo plantio em curva de nível, desde que a curvatura seja suave).
5. Converter as polilinhas de volta para o sistema de pixels original.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _rotacionar_com_borda(img, angulo, interp=cv2.INTER_NEAREST):
    """Rotaciona expandindo o canvas. Retorna (imagem, M, M_inversa)."""
    h, w = img.shape[:2]
    centro = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += nW / 2.0 - centro[0]
    M[1, 2] += nH / 2.0 - centro[1]
    rot = cv2.warpAffine(img, M, (nW, nH), flags=interp)
    Minv = cv2.invertAffineTransform(M)
    return rot, M, Minv


def orientacao_dominante(mask, gsd, espacamento, intens=None):
    """Ângulo de rotação (graus) que deixa as linhas de plantio verticais.

    O perfil é normalizado pela área válida de cada coluna (a rotação cria
    cantos vazios que, sem isso, distorcem o score) e filtrado passa-banda
    na escala do espaçamento, para medir só o contraste linha/entrelinha.
    Com `intens` (índice de vegetação contínuo), usa a TEXTURA do verde em
    vez do binário — essencial quando o dossel fecha e as ruas somem da
    máscara.
    """
    esp_px = espacamento / gsd
    fator = max(1, int(round(esp_px / 6.0)))  # reduz p/ espaçamento ~6 px
    fonte = intens if intens is not None else mask
    small = fonte[::fator, ::fator].astype(np.float32)
    uns = np.ones(small.shape, dtype=np.float32)
    esp_small = esp_px / fator

    def score(ang):
        rot, _, _ = _rotacionar_com_borda(small, ang)
        cnt, _, _ = _rotacionar_com_borda(uns, ang)
        veg = rot.sum(axis=0).astype(np.float64)
        val = cnt.sum(axis=0).astype(np.float64)
        ok = val > 0.3 * val.max()
        if ok.sum() < 10:
            return 0.0
        dens = veg[ok] / val[ok]
        alta = gaussian_filter1d(dens, sigma=max(1.0, esp_small / 6.0))
        base = gaussian_filter1d(dens, sigma=esp_small * 2.0)
        return float(np.std(alta - base))

    angulos = np.arange(-90, 90, 2.0)
    scores = [score(a) for a in angulos]
    melhor = angulos[int(np.argmax(scores))]

    finos = np.arange(melhor - 4, melhor + 4.01, 0.25)
    scores_f = [score(a) for a in finos]
    return float(finos[int(np.argmax(scores_f))])


def detectar_linhas(mask, gsd, espacamento, banda_m=8.0, min_comprimento_m=5.0,
                    angulo=None, completar=True, sensibilidade=1.0,
                    intens=None):
    """Detecta as linhas de plantio.

    Retorna (lista de polilinhas Nx2 em pixels da imagem ORIGINAL, angulo_usado).
    """
    if angulo is None:
        usar_int = intens if (intens is not None and mask.mean() > 0.55) \
            else None
        angulo = orientacao_dominante(mask, gsd, espacamento,
                                      intens=usar_int)

    m8 = mask.astype(np.uint8)
    rot, M, Minv = _rotacionar_com_borda(m8, angulo)
    cnt_rot, _, _ = _rotacionar_com_borda(np.ones_like(m8), angulo)
    H, W = rot.shape
    esp_px = espacamento / gsd
    banda_px = max(int(banda_m / gsd), int(2 * esp_px))

    # --- picos por banda (perfil = densidade de vegetação por coluna) ---
    bandas = []  # (y_centro, array de x dos picos)
    for y0 in range(0, H, banda_px):
        y1 = min(y0 + banda_px, H)
        altura = y1 - y0
        if altura < banda_px * 0.3:
            continue
        veg = rot[y0:y1].sum(axis=0).astype(np.float64)
        val = cnt_rot[y0:y1].sum(axis=0).astype(np.float64)
        ok = val > 0.5 * altura
        if ok.sum() < esp_px or veg[ok].max() < 1:
            continue
        dens = np.zeros(W)
        dens[ok] = veg[ok] / val[ok]
        dens_s = gaussian_filter1d(dens, sigma=esp_px / 5.0)
        s = float(np.clip(sensibilidade, 0.5, 2.0))
        picos, _ = find_peaks(
            dens_s,
            distance=max(2, 0.6 * esp_px),
            prominence=0.05 / s,
            height=0.06 / s,
        )
        picos = picos[ok[picos]]  # descarta picos fora da área válida
        finos = []
        for p in picos:  # refinamento subpixel (parábola de 3 pontos)
            if 1 <= p < W - 1:
                a, m, c = dens_s[p - 1], dens_s[p], dens_s[p + 1]
                den = a - 2 * m + c
                d = 0.5 * (a - c) / den if den != 0 else 0.0
                finos.append(p + float(np.clip(d, -0.6, 0.6)))
            else:
                finos.append(float(p))
        bandas.append(((y0 + y1) / 2.0, np.array(finos)))

    # --- encadeamento dos picos entre bandas (segue linhas curvas) ---
    # Quando o pico some (ex.: falha longa), a cadeia segue por extrapolação
    # linear por até 3 bandas, em vez de cortar caminho com uma corda reta
    # que desviaria do centro da rua em trechos curvos.
    tol = 0.45 * esp_px
    cadeias = []  # dict: xs, ys, reais (ponto real ou previsto), perdidas, ativa
    for bi, (yc, picos) in enumerate(bandas):
        usados = np.zeros(len(picos), dtype=bool)
        for c in cadeias:
            if not c["ativa"]:
                continue
            if len(c["xs"]) >= 2:
                passo = float(np.clip(c["xs"][-1] - c["xs"][-2],
                                      -0.45 * esp_px, 0.45 * esp_px))
                x_prev = c["xs"][-1] + passo
            else:
                x_prev = c["xs"][-1]
            dx = np.abs(picos - x_prev) if len(picos) else np.array([])
            if len(dx):
                dx = np.where(usados, np.inf, dx)
            if len(dx) and dx.min() <= tol and \
                    abs(picos[int(np.argmin(dx))] - c["xs"][-1]) <= 0.7 * esp_px:
                j = int(np.argmin(dx))
                usados[j] = True
                c["xs"].append(float(picos[j]))
                c["ys"].append(yc)
                c["reais"].append(True)
                c["perdidas"] = 0
            else:
                c["perdidas"] += 1
                if c["perdidas"] > 3:
                    c["ativa"] = False
                else:
                    c["xs"].append(float(x_prev))
                    c["ys"].append(yc)
                    c["reais"].append(False)
        for j, p in enumerate(picos):
            if not usados[j]:
                cadeias.append({"xs": [float(p)], "ys": [yc],
                                "reais": [True], "perdidas": 0, "ativa": True})

    # remove extrapolações penduradas no fim de cada cadeia
    limpas = []
    for c in cadeias:
        if not any(c["reais"]):
            continue
        fim = max(i for i, r in enumerate(c["reais"]) if r)
        if fim >= 1:
            limpas.append({"xs": c["xs"][:fim + 1], "ys": c["ys"][:fim + 1]})

    # emenda cadeias da mesma linha que quebraram (falhas maiores que a
    # janela de extrapolação): fim de uma casa com o início da outra
    limpas.sort(key=lambda c: c["ys"][0])
    fundidas = []
    for c in limpas:
        emendou = False
        for f in fundidas:
            gap_y = c["ys"][0] - f["ys"][-1]
            if 0 < gap_y <= 4.5 * banda_px:
                if len(f["xs"]) >= 2 and f["ys"][-1] != f["ys"][-2]:
                    incl = (f["xs"][-1] - f["xs"][-2]) / (f["ys"][-1] - f["ys"][-2])
                else:
                    incl = 0.0
                x_pred = f["xs"][-1] + incl * gap_y
                if abs(c["xs"][0] - x_pred) <= 1.2 * tol:
                    f["xs"].extend(c["xs"])
                    f["ys"].extend(c["ys"])
                    emendou = True
                    break
        if not emendou:
            fundidas.append(c)

    # corta trechos diagonais (saltos entre fileiras): um degrau brusco ou
    # uma deriva sustentada acima do que curva de nível real produz vira
    # ponto de corte — a cadeia é dividida e o "bigode" morre
    fundidas = _cortar_diagonais(fundidas, esp_px, banda_px)

    # suavização leve: tira a tremida sem alterar a trajetória
    for c in fundidas:
        if len(c["xs"]) >= 5:
            xs = np.asarray(c["xs"], dtype=float)
            xs[1:-1] = 0.25 * xs[:-2] + 0.5 * xs[1:-1] + 0.25 * xs[2:]
            c["xs"] = xs.tolist()

    # completamento por malha: se entre duas fileiras vizinhas cabe
    # exatamente 1..3 fileiras (múltiplo do espaçamento) E há vegetação no
    # meio, a fileira faltante é reconstruída — recupera justamente as
    # fileiras mais falhadas, que não formam cadeia sozinhas
    if completar and len(fundidas) >= 2:
        fundidas = fundidas + _completar_malha(fundidas, rot, esp_px, banda_px)

    fundidas = _mesclar_proximas(fundidas, esp_px, banda_px)
    _estender_pontas(fundidas, rot, esp_px, banda_px)

    # --- cadeias -> polilinhas no pixel original ---
    linhas_px = []
    min_pts = 3
    for c in fundidas:
        if len(c["xs"]) < min_pts:
            continue
        pts_rot = np.column_stack([c["xs"], c["ys"]]).astype(np.float32)
        pts = cv2.transform(pts_rot.reshape(-1, 1, 2), Minv).reshape(-1, 2)
        comp_m = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) * gsd
        if comp_m >= min_comprimento_m:
            linhas_px.append(pts.astype(np.float64))

    return linhas_px, angulo


def _presenca_media(rot, xs, ys, esp_px, banda_px):
    """Fração média de vegetação em janelinhas ao longo da linha candidata."""
    H, W = rot.shape
    w = max(2, int(0.25 * esp_px))
    h = max(2, int(banda_px / 2))
    valores = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        jan = rot[max(0, yi - h):min(H, yi + h),
                  max(0, xi - w):min(W, xi + w)]
        if jan.size:
            valores.append(float(jan.mean()))
    return float(np.mean(valores)) if valores else 0.0


def _completar_malha(cadeias, rot, esp_px, banda_px, tol_malha=0.40,
                     presenca_min=0.02):
    # a malha de vizinhança considera só cadeias SÓLIDAS: um toco curto em
    # cima de fileira morta "tapa" o buraco na contagem e impede a inserção
    solidas = [c for c in cadeias
               if (c["ys"][-1] - c["ys"][0]) >= 4 * banda_px]
    if len(solidas) < 2:
        solidas = cadeias
    ordenadas = sorted(solidas, key=lambda c: float(np.mean(c["xs"])))
    novas = []
    for a, b in zip(ordenadas, ordenadas[1:]):
        y0 = max(a["ys"][0], b["ys"][0])
        y1 = min(a["ys"][-1], b["ys"][-1])
        if y1 - y0 < 3 * banda_px:
            continue
        ys_comuns = [y for y in a["ys"] if y0 <= y <= y1]
        if len(ys_comuns) < 3:
            continue
        xa = np.interp(ys_comuns, a["ys"], a["xs"])
        xb = np.interp(ys_comuns, b["ys"], b["xs"])
        gap = float(np.median(xb - xa))
        k = int(round(gap / esp_px)) - 1
        if not 1 <= k <= 3:
            continue
        if abs(gap - (k + 1) * esp_px) > tol_malha * esp_px:
            continue  # o vão não casa com a malha do espaçamento
        for j in range(1, k + 1):
            frac = j / (k + 1)
            xs_novo = (xa + frac * (xb - xa)).tolist()
            # vão de UMA fileira, casando apertado com a malha e por dezenas
            # de bandas = fileira totalmente morta (replantio): reconstrói
            # mesmo sem vegetação; nos demais casos exige presença mínima
            aperto = (k == 1 and abs(gap - 2 * esp_px) <= 0.25 * esp_px
                      and (y1 - y0) >= 10 * banda_px)
            minimo = 0.004 if aperto else presenca_min
            if _presenca_media(rot, xs_novo, ys_comuns,
                               esp_px, banda_px) < minimo:
                continue  # sem vegetação: carreador/caminho, não fileira
            novas.append({"xs": xs_novo, "ys": list(ys_comuns)})
    return novas


def _estender_pontas(cadeias, rot, esp_px, banda_px,
                     presenca_min=0.08, max_passos=8):
    """Estende cada extremidade banda a banda enquanto houver vegetação —
    a linha passa a cobrir a fileira desde o começo de verdade."""
    H, W = rot.shape
    for c in cadeias:
        for direcao in (-1, 1):
            for _ in range(max_passos):
                xs, ys = c["xs"], c["ys"]
                if len(xs) < 2:
                    break
                k = min(5, len(xs) - 1)
                difs = np.diff(xs[:k + 1]) if direcao < 0 else \
                    np.diff(xs[-k - 1:])
                incl = float(np.clip(np.median(difs),
                                     -0.30 * esp_px, 0.30 * esp_px))
                if direcao < 0:
                    ny = ys[0] - banda_px
                    nx = xs[0] - incl
                else:
                    ny = ys[-1] + banda_px
                    nx = xs[-1] + incl
                if not (0 <= ny < H and 0 <= nx < W):
                    break
                if _presenca_media(rot, [nx], [ny], esp_px,
                                   banda_px) < presenca_min:
                    break
                if direcao < 0:
                    c["xs"].insert(0, float(nx))
                    c["ys"].insert(0, float(ny))
                else:
                    c["xs"].append(float(nx))
                    c["ys"].append(float(ny))


def _cortar_diagonais(cadeias, esp_px, banda_px=80.0,
                      res_medio=0.25, res_pico=0.50):
    """Tesoura por FLUXO: curvas de nível são compartilhadas — todas as
    fileiras vizinhas derivam juntas. O fluxo(y) é a mediana do passo
    lateral de todas as cadeias em cada banda; o que desvia dele (bigode,
    pulo de fileira, costura de mosaico) é cortado; a curva legítima, não."""
    if len(cadeias) < 4:
        return cadeias
    # fluxo comum por banda
    por_y = {}
    for c in cadeias:
        xs = np.asarray(c["xs"], dtype=float)
        ys = np.asarray(c["ys"], dtype=float)
        for i in range(len(xs) - 1):
            if ys[i + 1] - ys[i] > 1.5 * banda_px:
                continue  # emenda: não entra na estatística
            por_y.setdefault(round(ys[i] / banda_px), []).append(xs[i + 1] - xs[i])
    fluxo = {y: float(np.median(v)) for y, v in por_y.items() if len(v) >= 5}

    saida = []
    for c in cadeias:
        xs = np.asarray(c["xs"], dtype=float)
        ys = np.asarray(c["ys"], dtype=float)
        if len(xs) < 3:
            saida.append(c)
            continue
        res = np.zeros(len(xs) - 1)
        for i in range(len(xs) - 1):
            if ys[i + 1] - ys[i] > 1.5 * banda_px:
                continue  # passo de emenda: tolerado
            f = fluxo.get(round(ys[i] / banda_px), 0.0)
            res[i] = (xs[i + 1] - xs[i]) - f
        res_m = (np.convolve(res, np.ones(2) / 2.0, mode="same")
                 if len(res) >= 2 else res)
        cortes = np.where((np.abs(res) > res_pico * esp_px) |
                          (np.abs(res_m) > res_medio * esp_px))[0]
        ini = 0
        for fim in list(cortes + 1) + [len(xs)]:
            if fim - ini >= 3:
                saida.append({"xs": xs[ini:fim].tolist(),
                              "ys": ys[ini:fim].tolist()})
            ini = fim
    return saida


def _mesclar_proximas(cadeias, esp_px, banda_px, limiar=0.30):
    """Funde cadeias que correm quase na mesma linha (< limiar*esp de
    distância lateral onde coexistem) — elimina as linhas "grossas"
    formadas por duas detecções da mesma fileira."""
    if len(cadeias) < 2:
        return cadeias
    ordenadas = sorted(cadeias, key=lambda c: float(np.mean(c["xs"])))
    usada = [False] * len(ordenadas)
    saida = []
    for i, a in enumerate(ordenadas):
        if usada[i]:
            continue
        base = {"xs": list(a["xs"]), "ys": list(a["ys"])}
        for j in range(i + 1, len(ordenadas)):
            if usada[j]:
                continue
            b = ordenadas[j]
            if float(np.mean(b["xs"])) - float(np.mean(base["xs"])) > esp_px:
                break  # já passou; ordenadas por x
            y0 = max(base["ys"][0], b["ys"][0])
            y1 = min(base["ys"][-1], b["ys"][-1])
            if y1 - y0 < 2 * banda_px:
                continue  # mal se sobrepõem: fileiras distintas
            ys_c = [y for y in base["ys"] if y0 <= y <= y1]
            if len(ys_c) < 2:
                continue
            xa = np.interp(ys_c, base["ys"], base["xs"])
            xb = np.interp(ys_c, b["ys"], b["xs"])
            if float(np.median(np.abs(xa - xb))) < limiar * esp_px:
                # funde: une pontos e reordena por y, média onde coincide
                ys_all = sorted(set(base["ys"]) | set(b["ys"]))
                xa2 = np.interp(ys_all, base["ys"], base["xs"],
                                left=np.nan, right=np.nan)
                xb2 = np.interp(ys_all, b["ys"], b["xs"],
                                left=np.nan, right=np.nan)
                xm = np.nanmean(np.vstack([xa2, xb2]), axis=0)
                base = {"xs": xm.tolist(), "ys": list(ys_all)}
                usada[j] = True
        saida.append(base)
    return saida
