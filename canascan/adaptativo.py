"""Detecção de linhas ADAPTATIVA — ângulo e espaçamento LOCAIS por tile.

Motivação
---------
`linhas.detectar_linhas` descobre UMA orientação dominante e rotaciona a
área inteira por esse ângulo. Isso funciona muito bem num talhão homogêneo,
mas num talhão grande com plantio em CURVA DE NÍVEL (a orientação das
fileiras varia ao longo do talhão) um ângulo único acerta uma parte e ERRA
o resto — as linhas desgarram, cruzam, dobram e grudam.

Este módulo divide a área em tiles sobrepostos, mede o ângulo (e o
espaçamento) LOCAL de cada tile, suaviza o campo de ângulos com mediana
circular de vizinhança (mata o tile ralo que "vira" 90°), detecta as linhas
tile a tile e junta tudo com uma deduplicação global que remove paralelas
coladas. Mais lento que o global, porém segue as fileiras mesmo em curva.

Uso (mesma assinatura de retorno de detectar_linhas):
    linhas_px, ang_medio = detectar_linhas_adaptativo(mask, gsd, espacamento)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import linhas as ml


# --------------------------------------------------------------------------
# estimadores locais
# --------------------------------------------------------------------------
def _angulo_local(sub, gsd, espacamento):
    """Orientação dominante do tile (busca de ângulo em -90..90)."""
    esp_px = espacamento / gsd
    fator = max(1, int(round(esp_px / 6.0)))
    small = sub[::fator, ::fator].astype(np.float32)
    uns = np.ones_like(small)
    esp_s = esp_px / fator

    def score(a):
        rot, _, _ = ml._rotacionar_com_borda(small, a)
        cnt, _, _ = ml._rotacionar_com_borda(uns, a)
        veg = rot.sum(0).astype(np.float64)
        val = cnt.sum(0).astype(np.float64)
        ok = val > 0.3 * val.max()
        if ok.sum() < 10:
            return 0.0
        dens = veg[ok] / val[ok]
        alta = gaussian_filter1d(dens, sigma=max(1.0, esp_s / 6.0))
        base = gaussian_filter1d(dens, sigma=esp_s * 2.0)
        return float(np.std(alta - base))

    angs = np.arange(-90, 90, 3.0)
    sc = [score(a) for a in angs]
    a0 = angs[int(np.argmax(sc))]
    fin = np.arange(a0 - 3, a0 + 3.01, 0.5)
    scf = [score(a) for a in fin]
    imax = int(np.argmax(scf))
    return float(fin[imax]), float(scf[imax])


def _espacamento_local(sub, gsd, esp_prior, ang):
    """Espaçamento por autocorrelação do perfil projetado, LIMITADO a
    [0.7, 1.4]*prior — impede travar no harmônico de meio-espaçamento."""
    rot, _, _ = ml._rotacionar_com_borda(sub.astype(np.float32), ang)
    prof = rot.sum(0).astype(np.float64)
    prof = prof - prof.mean()
    if prof.std() < 1e-6:
        return esp_prior
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    if ac[0] <= 0:
        return esp_prior
    lo = int(0.7 * esp_prior / gsd)
    hi = int(1.4 * esp_prior / gsd)
    seg = ac[lo:hi]
    if len(seg) < 3:
        return esp_prior
    return (np.argmax(seg) + lo) * gsd


def _preencher_campo(A, valido, semente):
    """Preenche células válidas mas sem valor (NaN) com a mediana circular
    dos vizinhos já preenchidos, expandindo em anéis a partir das sementes.
    É o que faz a borda rala HERDAR o ângulo do miolo bom em vez de chutar."""
    R, C = A.shape
    out = A.copy()
    for _ in range(R + C):
        pend = valido & np.isnan(out)
        if not pend.any():
            break
        novos = out.copy()
        mudou = False
        for i in range(R):
            for j in range(C):
                if not pend[i, j]:
                    continue
                ii = slice(max(0, i - 1), i + 2)
                jj = slice(max(0, j - 1), j + 2)
                viz = out[ii, jj]
                viz = viz[~np.isnan(viz)]
                if len(viz) == 0:
                    continue
                v = np.deg2rad(viz * 2.0)
                ang = np.rad2deg(np.arctan2(np.median(np.sin(v)),
                                            np.median(np.cos(v)))) / 2.0
                ang = ang % 180
                if ang >= 90:
                    ang -= 180
                novos[i, j] = ang
                mudou = True
        out = novos
        if not mudou:
            break
    return out


def _mediana_circular(A, M):
    """Mediana circular 3x3 do campo de ângulos (graus, período 180),
    ignorando células inválidas. Remove tiles que 'viraram' sozinhos."""
    R, C = A.shape
    out = A.copy()
    for i in range(R):
        for j in range(C):
            if not M[i, j]:
                continue
            ii = slice(max(0, i - 1), i + 2)
            jj = slice(max(0, j - 1), j + 2)
            viz = A[ii, jj][M[ii, jj]]
            if len(viz) == 0:
                continue
            v = np.deg2rad(viz * 2.0)
            ang = np.rad2deg(np.arctan2(np.median(np.sin(v)),
                                        np.median(np.cos(v)))) / 2.0
            ang = ang % 180
            if ang >= 90:
                ang -= 180
            out[i, j] = ang
    return out


def _clip_segmento(p, q, x0, y0, x1, y1):
    """Liang-Barsky: recorta o segmento p->q à caixa [x0,x1]x[y0,y1].
    Retorna (a, b) dentro da caixa ou None."""
    px, py = p
    dx, dy = q[0] - px, q[1] - py
    t0, t1 = 0.0, 1.0
    for pcoef, qcoef in ((-dx, px - x0), (dx, x1 - px),
                         (-dy, py - y0), (dy, y1 - py)):
        if pcoef == 0:
            if qcoef < 0:
                return None
        else:
            t = qcoef / pcoef
            if pcoef < 0:
                if t > t1:
                    return None
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return None
                if t < t1:
                    t1 = t
    a = np.array([px + t0 * dx, py + t0 * dy])
    b = np.array([px + t1 * dx, py + t1 * dy])
    return a, b


def _clip_polilinha(pts, x0, y0, x1, y1):
    """Recorta uma polilinha à caixa; devolve lista de pedaços (>=2 pontos)
    inteiramente dentro do núcleo do tile."""
    pts = np.asarray(pts, float)
    pedacos, cur = [], []
    for i in range(len(pts) - 1):
        seg = _clip_segmento(pts[i], pts[i + 1], x0, y0, x1, y1)
        if seg is None:
            if len(cur) >= 2:
                pedacos.append(np.array(cur))
            cur = []
            continue
        a, b = seg
        if not cur:
            cur = [a, b]
        elif np.allclose(cur[-1], a, atol=1e-6):
            cur.append(b)
        else:
            if len(cur) >= 2:
                pedacos.append(np.array(cur))
            cur = [a, b]
    if len(cur) >= 2:
        pedacos.append(np.array(cur))
    return pedacos


def _dedup_global(linhas, esp_px):
    """Funde/remove linhas paralelas coladas (< 0.5*esp de distância lateral
    onde coexistem). Mantém a mais longa de cada par. Aceita polilinhas de
    orientação qualquer (necessário porque cada tile tem seu ângulo)."""
    if len(linhas) < 2:
        return linhas
    Ls = [np.asarray(L, float) for L in linhas]
    comp = [np.sum(np.linalg.norm(np.diff(L, axis=0), axis=1)) for L in Ls]
    keep = []
    for idx in np.argsort(comp)[::-1]:      # das maiores para as menores
        L = Ls[idx]
        mid = L.mean(0)
        dv = L[-1] - L[0]
        nrm = np.linalg.norm(dv)
        if nrm < 1e-6:
            continue
        dv /= nrm
        perp = np.array([-dv[1], dv[0]])
        dup = False
        for K in keep:
            dK = np.linalg.norm(K - mid, axis=1)
            j = int(np.argmin(dK))
            if dK[j] > 3 * esp_px:
                continue
            if abs(np.dot(K[j] - mid, perp)) < 0.5 * esp_px:
                dvk = K[min(j + 1, len(K) - 1)] - K[j]
                nk = np.linalg.norm(dvk)
                if nk < 1e-6 or abs(np.dot(dvk / nk, dv)) > 0.9:
                    dup = True
                    break
        if not dup:
            keep.append(L)
    return keep


# --------------------------------------------------------------------------
# motor principal
# --------------------------------------------------------------------------
def detectar_linhas_adaptativo(mask, gsd, espacamento, banda_m=4.0,
                               min_comprimento_m=5.0, sensibilidade=1.0,
                               tile_m=35.0, overlap_frac=0.35, intens=None):
    """Detecta linhas com ângulo/espaçamento LOCAIS por tile.

    Retorna (lista de polilinhas Nx2 em pixels da área, angulo_medio).
    Se a área for pequena demais para tiles, cai no detector global.
    """
    H, W = mask.shape
    esp_px = espacamento / gsd
    cell = int(tile_m / gsd)

    # só cai no global se a área inteira couber num único tile (aí não há
    # o que adaptar) ou se o tile for pequeno demais para conter fileiras
    if cell < 3 * esp_px or (H <= cell and W <= cell):
        return ml.detectar_linhas(
            mask, gsd, espacamento, banda_m=banda_m,
            min_comprimento_m=min_comprimento_m,
            sensibilidade=sensibilidade, intens=intens)

    nR = max(1, int(np.ceil(H / cell)))
    nC = max(1, int(np.ceil(W / cell)))

    # 1) campo de ângulo/espaçamento/confiança/verde por célula
    A = np.full((nR, nC), np.nan)
    E = np.full((nR, nC), float(espacamento))
    Cf = np.zeros((nR, nC))
    G = np.zeros((nR, nC))
    Mv = np.zeros((nR, nC), bool)
    for r in range(nR):
        for c in range(nC):
            sub = mask[r * cell:min((r + 1) * cell, H),
                       c * cell:min((c + 1) * cell, W)]
            g = float(sub.mean())
            if g < 0.01 or min(sub.shape) < 3 * esp_px:
                continue
            a, conf = _angulo_local(sub, gsd, espacamento)
            A[r, c] = a
            E[r, c] = _espacamento_local(sub, gsd, espacamento, a)
            Cf[r, c] = conf
            G[r, c] = g
            Mv[r, c] = True

    if not Mv.any():
        return [], None

    # 2) CONFIANÇA: só confia no ângulo de células com verde e contraste
    # suficientes. As de borda/ralas (pouco verde ou pouco contraste) NÃO
    # usam o próprio chute — herdam o ângulo dos vizinhos bons do miolo.
    # É isso que conserta as pontas de talhão desgarrando.
    if Mv.sum() >= 4:
        conf_ref = float(np.median(Cf[Mv]))
        confiavel = Mv & (Cf >= 0.40 * conf_ref) & (G >= 0.03)
        if confiavel.sum() >= max(2, int(0.15 * Mv.sum())):
            A = _preencher_campo(np.where(confiavel, A, np.nan), Mv, confiavel)
            E = np.where(confiavel, E, float(np.median(E[confiavel])))

    # suavização final (mata resíduo de flip)
    A = _mediana_circular(A, Mv)
    ang_medio = float(np.nanmedian(A[Mv]))

    # 3) detecção por tile sobreposto, usando o ângulo/esp LOCAL suavizado
    tile = cell
    ov = int(tile * overlap_frac)
    step = max(1, tile - ov)
    todas = []
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            sub = mask[y0:y1, x0:x1]
            if sub.mean() < 0.01 or min(sub.shape) < 3 * esp_px:
                continue
            rc = min(nR - 1, (y0 + tile // 2) // cell)
            cc = min(nC - 1, (x0 + tile // 2) // cell)
            ang = A[rc, cc]
            if np.isnan(ang):
                ang = ang_medio   # borda sem ângulo próprio: usa o do miolo
            esp_loc = float(E[rc, cc])
            lin, _ = ml.detectar_linhas(
                sub, gsd, esp_loc, banda_m=banda_m,
                min_comprimento_m=min_comprimento_m * 0.5,
                angulo=float(ang), sensibilidade=sensibilidade)
            # RECORTA cada linha ao NÚCLEO do tile (região sem sobreposição).
            # Os núcleos se encostam sem cobrir duas vezes -> cada fileira sai
            # coberta uma única vez, ponta a ponta, sem buraco e sem dobra.
            cx0 = x0 + (ov // 2 if x0 > 0 else 0)
            cx1 = x1 - (ov // 2 if x1 < W else 0)
            cy0 = y0 + (ov // 2 if y0 > 0 else 0)
            cy1 = y1 - (ov // 2 if y1 < H else 0)
            min_ped = max(1.2, min_comprimento_m * 0.3) / gsd  # px
            for L in lin:
                L = np.asarray(L) + [x0, y0]
                for ped in _clip_polilinha(L, cx0, cy0, cx1, cy1):
                    comp = np.sum(np.linalg.norm(np.diff(ped, axis=0), axis=1))
                    if comp >= min_ped:
                        todas.append(ped.astype(np.float64))

    return todas, ang_medio
