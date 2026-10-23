"""Painel de mapa da GUI — prévia eficiente de ortomosaicos gigantes.

- Prévia decimada (~3200 px) do raster inteiro; nunca carrega tudo.
- "Nitidez aqui" agora é um REMENDO em alta resolução desenhado POR CIMA da
  prévia geral: o mapa completo continua embaixo, "Ajustar" sempre volta à
  visão geral e "Remover nitidez" descarta o remendo.
- Vetores são simplificados SÓ para exibição (falhas viram segmentos entre
  as pontas; linhas são decimadas ao que o olho distingue): dezenas de
  milhares de feições continuam fluidas. O resultado.gpkg mantém a precisão
  total para o QGIS.
"""
import os

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)


class _CarregaPrevia(QThread):
    pronto = Signal(object)  # (img HxWx3 uint8, limites (l,b,r,t), erro)

    def __init__(self, caminho, max_px=3200, limites=None):
        super().__init__()
        self.caminho, self.max_px, self.limites = caminho, max_px, limites

    def run(self):
        try:
            import rasterio
            from rasterio.windows import Window, from_bounds, intersection
            with rasterio.open(self.caminho) as src:
                if self.limites:
                    l, b, r, t = self.limites
                    jan = from_bounds(l, b, r, t, src.transform)
                    jan = jan.round_offsets().round_lengths()
                    jan = intersection(jan, Window(0, 0, src.width, src.height))
                    W, H = int(jan.width), int(jan.height)
                    if W < 4 or H < 4:
                        raise ValueError("região fora do raster")
                    esc = max(1.0, max(W, H) / self.max_px)
                    arr = src.read(indexes=[1, 2, 3], window=jan,
                                   out_shape=(3, int(H / esc), int(W / esc)))
                    tr = src.window_transform(jan)
                    limites = (tr.c, tr.f + tr.e * H, tr.c + tr.a * W, tr.f)
                else:
                    W, H = src.width, src.height
                    esc = max(1.0, max(W, H) / self.max_px)
                    arr = src.read(indexes=[1, 2, 3],
                                   out_shape=(3, int(H / esc), int(W / esc)))
                    bb = src.bounds
                    limites = (bb.left, bb.bottom, bb.right, bb.top)
                crs = src.crs
            img = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
            self.pronto.emit((img, limites, crs, None))
        except Exception as e:  # noqa: BLE001
            self.pronto.emit((None, None, None, f"{type(e).__name__}: {e}"))


class _MedidorEspacamento(QThread):
    """Detecta o espaçamento entre fileiras na região visível.

    Como: máscara de vegetação -> orientação dominante -> perfil de
    densidade por coluna -> AUTOCORRELAÇÃO do perfil. O período do padrão
    repetitivo É o espaçamento — sem depender de chute inicial."""
    pronto = Signal(object)  # (esp_m, angulo, confianca, erro)

    def __init__(self, caminho, limites, max_px=1600):
        super().__init__()
        self.caminho, self.limites, self.max_px = caminho, limites, max_px

    def run(self):
        try:
            import rasterio
            from rasterio.windows import Window, from_bounds, intersection
            from scipy.ndimage import gaussian_filter1d
            from . import linhas as mod_linhas
            from . import vegetacao

            with rasterio.open(self.caminho) as src:
                l, b, r, t = self.limites
                jan = from_bounds(l, b, r, t, src.transform)
                jan = jan.round_offsets().round_lengths()
                jan = intersection(jan, Window(0, 0, src.width, src.height))
                W, H = int(jan.width), int(jan.height)
                if W < 50 or H < 50:
                    raise ValueError("região pequena demais")
                gsd_src = (abs(src.transform.a) + abs(src.transform.e)) / 2.0
                if max(W, H) * gsd_src > 220:
                    raise ValueError("região muito ampla — dê zoom até a "
                                     "tela mostrar ~150 m e tente de novo")
                esc = max(1.0, max(W, H) / self.max_px)
                arr = src.read(indexes=[1, 2, 3], window=jan,
                               out_shape=(3, int(H / esc), int(W / esc)))
                tr = src.window_transform(jan)
                gsd = (abs(tr.a) + abs(tr.e)) / 2.0 * esc
            img = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
            mask = vegetacao.mascara_vegetacao(img, metodo="gli", limiar=0.02)
            if mask.mean() < 0.01:
                mask = vegetacao.mascara_vegetacao(img, metodo="gli",
                                                   limiar=0.01)
            if mask.mean() < 0.005:
                raise ValueError("quase não há vegetação nesta região")

            # busca de orientação rápida e local (mask reduzida a ~640 px)
            m8 = mask.astype(np.uint8)
            f = max(1, int(round(max(m8.shape) / 640)))
            pequeno = m8[::f, ::f]
            uns_p = np.ones_like(pequeno)

            def score(a):
                rp, _, _ = mod_linhas._rotacionar_com_borda(pequeno, a)
                cp, _, _ = mod_linhas._rotacionar_com_borda(uns_p, a)
                v = rp.sum(axis=0).astype(np.float64)
                c = cp.sum(axis=0).astype(np.float64)
                okc = c > 0.3 * c.max()
                if okc.sum() < 10:
                    return 0.0
                d = v[okc] / c[okc]
                alta = gaussian_filter1d(d, 1.5)
                bx = gaussian_filter1d(d, 12.0)
                return float(np.std(alta - bx))

            grossos = np.arange(-90, 90, 3.0)
            ang = float(grossos[int(np.argmax([score(a) for a in grossos]))])
            finos = np.arange(ang - 3, ang + 3.01, 0.5)
            ang = float(finos[int(np.argmax([score(a) for a in finos]))])

            rot, _, _ = mod_linhas._rotacionar_com_borda(m8, ang)
            cnt, _, _ = mod_linhas._rotacionar_com_borda(np.ones_like(m8), ang)
            veg = rot.sum(axis=0).astype(np.float64)
            val = cnt.sum(axis=0).astype(np.float64)
            ok = val > 0.3 * val.max()
            dens = np.zeros_like(veg)
            dens[ok] = veg[ok] / val[ok]
            base = gaussian_filter1d(dens, sigma=max(3.0, 2.5 / gsd))
            sinal = (dens - base) * ok

            # suaviza 1 px: mata a autocorrelação de textura pixel-a-pixel
            sinal = gaussian_filter1d(sinal, 1.0)
            ac = np.correlate(sinal, sinal, "full")[len(sinal) - 1:]
            if ac[0] <= 0:
                raise ValueError("sem padrão de fileiras detectável")
            ac = ac / ac[0]
            # faixa agronômica: nenhuma cultura em linha usa < 0,35 m
            lo = max(4, int(0.35 / gsd))
            hi = min(len(ac) - 2, int(3.0 / gsd))
            if hi <= lo + 3:
                raise ValueError("resolução insuficiente p/ medir")

            from scipy.signal import find_peaks
            picos, props = find_peaks(ac[lo:hi], prominence=0.03)
            picos = picos + lo
            if not len(picos):
                raise ValueError("sem padrão de fileiras nesta região")
            ordem = picos[np.argsort(ac[picos])[::-1]]
            # validação de harmônico: padrão de fileira VERDADEIRO se repete
            # também no dobro do período; ruído de textura, não
            i = None
            for cand in ordem[:4]:
                dobro = 2 * cand
                if dobro < len(ac) - 1 and ac[dobro] >= 0.35 * ac[cand]:
                    i = int(cand)
                    break
            confiavel = i is not None
            if i is None:
                i = int(ordem[0])
            y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
            den = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            esp = max(lo * gsd, (i + float(np.clip(delta, -1, 1))) * gsd)
            conf = float(ac[i]) if confiavel else float(ac[i]) * 0.4
            self.pronto.emit((esp, ang, conf, None))
        except Exception as e:  # noqa: BLE001
            self.pronto.emit((None, None, None, f"{type(e).__name__}: {e}"))


class _CarregaRef(QThread):
    """Carrega shapefiles externos de referência (linhas e/ou falhas)."""
    pronto = Signal(object)

    def __init__(self, shp_linhas=None, shp_falhas=None, crs_destino=None):
        super().__init__()
        self.shp_linhas = shp_linhas
        self.shp_falhas = shp_falhas
        self.crs_destino = crs_destino

    def run(self):
        try:
            import geopandas as gpd
            from rasterio.crs import CRS

            def carregar(caminho):
                if not caminho or not os.path.isfile(caminho):
                    return []
                g = gpd.read_file(caminho)
                if self.crs_destino and g.crs != self.crs_destino:
                    try:
                        g = g.to_crs(self.crs_destino)
                    except Exception:
                        g = g.set_crs(self.crs_destino, allow_override=True)
                coords = []
                for geom in g.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    gs = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                    for p in gs:
                        coords.append(np.asarray(p.coords if hasattr(p, "coords")
                                                 else p.exterior.coords))
                return coords

            saida = {
                "ref_linhas": carregar(self.shp_linhas),
                "ref_falhas": carregar(self.shp_falhas),
            }
            self.pronto.emit((saida, None))
        except Exception as e:
            self.pronto.emit((None, f"{type(e).__name__}: {e}"))


class _CarregaVetores(QThread):
    """Carrega vetores do resultado.gpkg do CanaScan."""
    pronto = Signal(object)

    def __init__(self, gpkg):
        super().__init__()
        self.gpkg = gpkg

    def run(self):
        try:
            import geopandas as gpd

            def partes(gdf):
                out = []
                for g in gdf.geometry:
                    if g is None or g.is_empty:
                        continue
                    gs = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
                    for p in gs:
                        if p.geom_type == "Polygon":
                            out.append(np.asarray(p.exterior.coords))
                        else:
                            out.append(np.asarray(p.coords))
                return out

            saida = {}
            for camada, nome in [("linhas_plantio", "linhas"),
                                 ("falhas", "falhas"),
                                 ("talhoes", "talhoes"),
                                 ("daninhas", "daninhas")]:
                try:
                    saida[nome] = partes(gpd.read_file(self.gpkg, layer=camada))
                except Exception:
                    saida[nome] = []
            self.pronto.emit((saida, None))
        except Exception as e:  # noqa: BLE001
            self.pronto.emit((None, f"{type(e).__name__}: {e}"))


class _Vista(QGraphicsView):
    moveu = Signal(QPointF)
    clicou = Signal(QPointF)
    clicou_dir = Signal()

    def __init__(self, cena):
        super().__init__(cena)
        # sem antialiasing nos vetores: pan/zoom muito mais leves
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#101418"))

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicou.emit(self.mapToScene(ev.position().toPoint()))
        elif ev.button() == Qt.RightButton:
            self.clicou_dir.emit()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        self.moveu.emit(self.mapToScene(ev.position().toPoint()))
        super().mouseMoveEvent(ev)


ICONES = {"📂 Ref. externa": "📂", "▱ Talhões": "▱", "⛔ Excluir área": "⛔", "↩ Desfazer": "↩",
          "💾 Salvar talhões": "💾", "✕ talhões": "✕",
          "📐 Espaçamento auto": "📐", "📏 Medir": "📏",
          "🔍 Nitidez aqui": "🔍", "✕ nitidez": "🔍✕", "Ajustar": "⤢"}

ESTILOS = {  # camada: (cor, largura, ordem z)
    "talhoes":     ("#f0f0f0", 1.6, 2),
    "linhas":      ("#ffee00", 1.1, 3),
    "falhas":      ("#ff2020", 2.2, 4),
    "daninhas":    ("#ff8c00", 1.3, 4),
    "ref_linhas":  ("#00aaff", 1.1, 5),
    "ref_falhas":  ("#cc44ff", 2.2, 6),
}

NOMES_CAMADA = {
    "talhoes":    "Talhões",
    "linhas":     "Linhas",
    "falhas":     "Falhas",
    "daninhas":   "Daninhas",
    "ref_linhas": "Ref. Linhas",
    "ref_falhas": "Ref. Falhas",
}


def _decimar(sx, sy, tol_px=1.2):
    """Mantém só vértices que andam ≥ tol_px na tela — o olho não vê o resto."""
    n = len(sx)
    if n <= 2:
        return sx, sy
    dx = np.diff(sx)
    dy = np.diff(sy)
    passo = np.hypot(dx, dy)
    manter = [0]
    acumulado = 0.0
    for i in range(1, n - 1):
        acumulado += passo[i - 1]
        if acumulado >= tol_px:
            manter.append(i)
            acumulado = 0.0
    manter.append(n - 1)
    idx = np.array(manter)
    return sx[idx], sy[idx]


class PainelMapa(QWidget):
    espacamento_detectado = Signal(float)
    talhoes_salvos = Signal(str)

    def __init__(self):
        super().__init__()
        self.caminho_tif = None
        self.limites = None
        self.tam = (1, 1)
        self.vetores = {}
        self.itens = {}
        self.item_patch = None
        self.crs = None
        self._desenhando = False
        self._t_atual = []
        self._t_tmp = None
        self._t_polys = []      # talhões fechados (pontos de cena)
        self._t_excl = []       # áreas de exclusão (furos)
        self._t_items = []
        self._historico = []    # ("t"|"e", item, indice) p/ desfazer
        self._modo_excl = False
        self._medindo = False
        self._ponto_a = None
        self._medidas = []      # itens fixados (linhas e textos)
        self._tmp = None        # (linha, texto) da medição em andamento
        self._threads = []
        self._dados_base = None
        self._dados_patch = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        topo = QHBoxLayout()      # linha 1: camadas + talhões
        topo2 = QHBoxLayout()     # linha 2: ferramentas + status
        self.chk = {}
        for camada, rotulo in NOMES_CAMADA.items():
            c = QCheckBox(rotulo)
            c.setChecked(True)
            c.toggled.connect(self._alternar)
            self.chk[camada] = c
            topo.addWidget(c)
        bt_fit = QPushButton("Ajustar")
        bt_fit.setToolTip("Volta para a visão geral do mapa")
        bt_fit.clicked.connect(self.ajustar)
        self.bt_nitidez = QPushButton("🔍 Nitidez aqui")
        self.bt_nitidez.setToolTip("Recarrega em alta resolução só a região "
                                   "visível, por cima do mapa geral")
        self.bt_nitidez.clicked.connect(self.nitidez)
        self.bt_ref = QPushButton("📂 Ref. externa")
        self.bt_ref.setToolTip("Carrega shapefiles de referência (outro software) "
                               "para comparar as linhas e falhas no mapa")
        self.bt_ref.clicked.connect(self.carregar_ref)
        self.bt_talhao = QPushButton("▱ Talhões")
        self.bt_talhao.setCheckable(True)
        self.bt_talhao.setToolTip("Clique nos cantos do talhão; botão "
                                  "DIREITO fecha o polígono. Desenhe "
                                  "quantos quiser e clique em Salvar.")
        self.bt_talhao.toggled.connect(self._modo_talhao)
        self.bt_excluir = QPushButton("⛔ Excluir área")
        self.bt_excluir.setCheckable(True)
        self.bt_excluir.setToolTip("Desenhe áreas SEM plantio dentro do "
                                   "talhão (buracos, carreadores): elas "
                                   "serão recortadas na hora de salvar")
        self.bt_excluir.toggled.connect(self._modo_excluir)
        self.bt_desfazer = QPushButton("↩ Desfazer")
        self.bt_desfazer.setEnabled(False)
        self.bt_desfazer.clicked.connect(self.desfazer_poligono)
        self.bt_salvar_t = QPushButton("💾 Salvar talhões")
        self.bt_salvar_t.setEnabled(False)
        self.bt_salvar_t.clicked.connect(lambda: self.salvar_talhoes())
        self.bt_limpar_t = QPushButton("✕ talhões")
        self.bt_limpar_t.setEnabled(False)
        self.bt_limpar_t.clicked.connect(self.limpar_talhoes)
        self.bt_esp = QPushButton("📐 Espaçamento auto")
        self.bt_esp.setToolTip("Mede sozinho o espaçamento entre fileiras "
                               "na região visível — sem régua, sem ângulo")
        self.bt_esp.clicked.connect(self.medir_espacamento)
        self.bt_medir = QPushButton("📏 Medir")
        self.bt_medir.setCheckable(True)
        self.bt_medir.setToolTip("Clique em dois pontos para medir a "
                                 "distância em metros (ex.: atravesse 10 "
                                 "fileiras e divida por 10 = espaçamento). "
                                 "Desmarque para limpar as medidas.")
        self.bt_medir.toggled.connect(self._modo_medir)
        self.bt_rm_nitidez = QPushButton("✕ nitidez")
        self.bt_rm_nitidez.setToolTip("Remove o remendo de alta resolução")
        self.bt_rm_nitidez.clicked.connect(self.remover_nitidez)
        self.bt_rm_nitidez.setEnabled(False)
        topo.addWidget(bt_fit)
        topo.addWidget(self.bt_ref)
        topo.addWidget(self.bt_talhao)
        topo.addWidget(self.bt_excluir)
        topo.addWidget(self.bt_desfazer)
        topo.addWidget(self.bt_salvar_t)
        topo.addWidget(self.bt_limpar_t)
        topo.addStretch(1)
        topo2.addWidget(self.bt_esp)
        topo2.addWidget(self.bt_medir)
        topo2.addWidget(self.bt_nitidez)
        topo2.addWidget(self.bt_rm_nitidez)
        topo2.addStretch(1)
        self.lb_coord = QLabel("")
        self.lb_status = QLabel("Escolha um GeoTIFF para ver a prévia.")
        self.lb_status.setStyleSheet("color:#8fd19a;")
        topo2.addWidget(self.lb_coord)
        topo2.addWidget(QLabel("|"))
        topo2.addWidget(self.lb_status)
        self.bt_ref.setProperty("compacto", True)
        self._botoes_barra = []
        for lay in (topo, topo2):
            for i in range(lay.count()):
                wdg = lay.itemAt(i).widget()
                if wdg is not None and wdg.__class__.__name__ == "QPushButton":
                    wdg.setProperty("compacto", True)
                    texto = wdg.text()
                    if wdg.toolTip() == "":
                        wdg.setToolTip(texto)
                    self._botoes_barra.append((wdg, texto, ICONES.get(texto, texto)))
        v.addLayout(topo)
        v.addLayout(topo2)

        self.cena = QGraphicsScene()
        self.vista = _Vista(self.cena)
        self.vista.setMinimumSize(240, 200)
        self.vista.moveu.connect(self._mostrar_coord)
        self.vista.moveu.connect(self._mover_medida)
        self.vista.clicou.connect(self._clique_medida)
        self.vista.clicou.connect(self._clique_talhao)
        self.vista.clicou_dir.connect(self._fechar_talhao)
        v.addWidget(self.vista, 1)

    # ---------- carregamento ----------
    def carregar_tif(self, caminho, limites=None):
        if not caminho or not os.path.isfile(caminho):
            return
        eh_patch = limites is not None
        if not eh_patch and caminho != self.caminho_tif:
            self.vetores = {}  # trocou de arquivo: vetores antigos não valem
        self.caminho_tif = caminho
        self.lb_status.setText("Carregando região…" if eh_patch
                               else "Carregando prévia…")
        th = _CarregaPrevia(caminho, limites=limites)
        th.pronto.connect(self._patch_pronto if eh_patch else self._base_pronta)
        th.start()
        self._threads.append(th)

    def _fazer_qimage(self, img):
        H, W = img.shape[:2]
        return QImage(img.data, W, H, 3 * W, QImage.Format_RGB888)

    def _base_pronta(self, pacote):
        img, limites, crs, erro = pacote
        self.crs = crs
        if erro:
            self.lb_status.setText(f"Erro na prévia: {erro}")
            return
        self._dados_base = img
        self.cena.clear()
        self.itens = {}
        self.item_patch = None
        self._medidas = []
        self._tmp = None
        self._ponto_a = None
        self._t_items, self._t_polys, self._t_excl = [], [], []
        self._historico = []
        self._t_tmp, self._t_atual = None, []
        self.bt_salvar_t.setEnabled(False)
        self.bt_limpar_t.setEnabled(False)
        self.bt_desfazer.setEnabled(False)
        self.bt_rm_nitidez.setEnabled(False)
        H, W = img.shape[:2]
        self.cena.addPixmap(QPixmap.fromImage(self._fazer_qimage(img)))
        self.cena.setSceneRect(0, 0, W, H)
        self.limites, self.tam = limites, (W, H)
        self._redesenhar()
        self.ajustar()
        gsd = (limites[2] - limites[0]) / W * 100
        self.lb_status.setText(f"Prévia {W}x{H} px (~{gsd:.0f} cm/px)")

    def _patch_pronto(self, pacote):
        img, lim, _crs, erro = pacote
        if erro or self.limites is None:
            self.lb_status.setText(f"Erro na região: {erro}")
            return
        self._dados_patch = img
        H2, W2 = img.shape[:2]
        l, b, r, t = self.limites
        W, Hb = self.tam
        item = QGraphicsPixmapItem(QPixmap.fromImage(self._fazer_qimage(img)))
        x = (lim[0] - l) / (r - l) * W
        y = (t - lim[3]) / (t - b) * Hb
        larg_cena = (lim[2] - lim[0]) / (r - l) * W
        item.setPos(x, y)
        item.setScale(larg_cena / W2)
        item.setZValue(1)  # acima da base, abaixo dos vetores
        item.setTransformationMode(Qt.SmoothTransformation)
        if self.item_patch is not None:
            self.cena.removeItem(self.item_patch)
        self.cena.addItem(item)
        self.item_patch = item
        self.bt_rm_nitidez.setEnabled(True)
        gsd = (lim[2] - lim[0]) / W2 * 100
        self.lb_status.setText(f"Nitidez: {W2}x{H2} px (~{gsd:.0f} cm/px) — "
                               f"\"Ajustar\" volta à visão geral")

    def remover_nitidez(self):
        if self.item_patch is not None:
            self.cena.removeItem(self.item_patch)
            self.item_patch = None
        self.bt_rm_nitidez.setEnabled(False)

    def carregar_vetores(self, gpkg):
        if not gpkg or not os.path.isfile(gpkg):
            return
        self.lb_status.setText("Carregando resultado…")
        th = _CarregaVetores(gpkg)
        th.pronto.connect(self._vetores_prontos)
        th.start()
        self._threads.append(th)

    def _vetores_prontos(self, pacote):
        dados, erro = pacote
        if erro:
            self.lb_status.setText(f"Erro nos vetores: {erro}")
            return
        self.vetores = dados
        self._redesenhar()
        self.lb_status.setText(
            f"{len(dados.get('linhas', []))} linhas | "
            f"{len(dados.get('falhas', []))} falhas no mapa")

    # ---------- desenho ----------
    def _geo_para_cena(self, pts):
        l, b, r, t = self.limites
        W, H = self.tam
        sx = (pts[:, 0] - l) / (r - l) * W
        sy = (t - pts[:, 1]) / (t - b) * H
        return sx, sy

    def _redesenhar(self):
        """Rasteriza cada camada vetorial numa imagem transparente (UMA vez).

        Pan/zoom então custam o mesmo que o raster de fundo — fluido mesmo
        com centenas de milhares de segmentos. O preço: em zoom profundo os
        traços suavizam junto com a imagem (pro pixel exato, QGIS)."""
        for item in self.itens.values():
            self.cena.removeItem(item)
        self.itens = {}
        if self.limites is None:
            return
        W, H = self.tam
        fator = max(1.0, min(1.3, 4200.0 / max(W, H)))
        Wo, Ho = int(W * fator), int(H * fator)
        for camada, (cor, larg, z) in ESTILOS.items():
            arrays = self.vetores.get(camada, [])
            if not arrays:
                continue
            segs = []
            for pts in arrays:
                if len(pts) < 2:
                    continue
                sx, sy = self._geo_para_cena(pts)
                if camada == "falhas":  # exibição: segmento ponta a ponta
                    segs.append(QLineF(sx[0] * fator, sy[0] * fator,
                                       sx[-1] * fator, sy[-1] * fator))
                    continue
                sx, sy = _decimar(sx, sy)
                xs, ys = sx * fator, sy * fator
                segs.extend(QLineF(xs[i], ys[i], xs[i + 1], ys[i + 1])
                            for i in range(len(xs) - 1))
            img = QImage(Wo, Ho, QImage.Format_ARGB32_Premultiplied)
            img.fill(Qt.transparent)
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(cor), larg * fator))
            for i in range(0, len(segs), 50000):  # lotes p/ o motor C++
                p.drawLines(segs[i:i + 50000])
            p.end()
            item = self.cena.addPixmap(QPixmap.fromImage(img))
            item.setScale(1.0 / fator)
            item.setZValue(z)
            item.setTransformationMode(Qt.SmoothTransformation)
            item.setVisible(self.chk[camada].isChecked())
            self.itens[camada] = item

    def _alternar(self):
        for camada, item in self.itens.items():
            item.setVisible(self.chk[camada].isChecked())

    # ---------- interação ----------
    def ajustar(self):
        self.vista.fitInView(self.cena.sceneRect(), Qt.KeepAspectRatio)

    def nitidez(self):
        if not self.caminho_tif or self.limites is None:
            return
        rect = self.vista.mapToScene(
            self.vista.viewport().rect()).boundingRect()
        rect = rect.intersected(self.cena.sceneRect())
        if rect.width() < 2:
            return
        l, b, r, t = self.limites
        W, H = self.tam
        gl = l + rect.left() / W * (r - l)
        gr = l + rect.right() / W * (r - l)
        gt = t - rect.top() / H * (t - b)
        gb = t - rect.bottom() / H * (t - b)
        self.carregar_tif(self.caminho_tif, limites=(gl, gb, gr, gt))

    # ---------- régua ----------
    def _modo_medir(self, ligado):
        if ligado:
            self.bt_talhao.setChecked(False)
            self.bt_excluir.setChecked(False)
        self._medindo = ligado
        self._ponto_a = None
        self.vista.setDragMode(QGraphicsView.NoDrag if ligado
                               else QGraphicsView.ScrollHandDrag)
        self.vista.viewport().setCursor(Qt.CrossCursor if ligado
                                        else Qt.OpenHandCursor)
        if not ligado:
            for it in self._medidas:
                self.cena.removeItem(it)
            self._medidas = []
            self._remover_tmp()

    def _remover_tmp(self):
        if self._tmp:
            for it in self._tmp:
                self.cena.removeItem(it)
            self._tmp = None

    def _dist_m(self, a, b):
        l, bb, r, t = self.limites
        W, H = self.tam
        ax = l + a.x() / W * (r - l); ay = t - a.y() / H * (t - bb)
        bx = l + b.x() / W * (r - l); by = t - b.y() / H * (t - bb)
        return float(np.hypot(bx - ax, by - ay))

    def _novo_texto(self):
        from PySide6.QtWidgets import QGraphicsSimpleTextItem
        from PySide6.QtGui import QBrush, QFont
        tx = QGraphicsSimpleTextItem("")
        tx.setBrush(QBrush(QColor("#00e5ff")))
        tx.setPen(QPen(QColor("#00232a"), 0.6))
        f = QFont(); f.setBold(True); f.setPointSize(10)
        tx.setFont(f)
        tx.setFlag(tx.GraphicsItemFlag.ItemIgnoresTransformations)
        tx.setZValue(7)
        self.cena.addItem(tx)
        return tx

    def _clique_medida(self, p):
        if not self._medindo or self.limites is None:
            return
        if self._ponto_a is None:
            self._ponto_a = p
            pen = QPen(QColor("#00e5ff"), 1.8)
            pen.setCosmetic(True)
            linha = self.cena.addLine(QLineF(p, p), pen)
            linha.setZValue(7)
            self._tmp = (linha, self._novo_texto())
        else:
            self._atualizar_tmp(p)
            self._medidas.extend(self._tmp)
            self._tmp = None
            self._ponto_a = None

    def _atualizar_tmp(self, p):
        linha, tx = self._tmp
        linha.setLine(QLineF(self._ponto_a, p))
        d = self._dist_m(self._ponto_a, p)
        tx.setText(f"{d:.2f} m" if d < 100 else f"{d:,.1f} m")
        meio = (self._ponto_a + p) / 2
        tx.setPos(meio.x(), meio.y())

    def _mover_medida(self, p):
        if self._medindo and self._ponto_a is not None and self._tmp:
            self._atualizar_tmp(p)
        if self._desenhando and self._t_atual:
            self._atualizar_tmp_talhao(p)

    # ---------- desenho de talhões ----------
    def _modo_talhao(self, ligado):
        if ligado:
            self.bt_medir.setChecked(False)
            self.bt_excluir.setChecked(False)
            self._modo_excl = False
        self._desenhando = ligado or self.bt_excluir.isChecked()
        self.vista.setDragMode(QGraphicsView.NoDrag if ligado
                               else QGraphicsView.ScrollHandDrag)
        self.vista.viewport().setCursor(Qt.CrossCursor if ligado
                                        else Qt.OpenHandCursor)
        if not ligado:
            self._cancelar_talhao_atual()

    def _modo_excluir(self, ligado):
        if ligado:
            self.bt_medir.setChecked(False)
            self.bt_talhao.setChecked(False)
        self._modo_excl = ligado
        self._desenhando = ligado or self.bt_talhao.isChecked()
        self.vista.setDragMode(QGraphicsView.NoDrag if self._desenhando
                               else QGraphicsView.ScrollHandDrag)
        self.vista.viewport().setCursor(Qt.CrossCursor if self._desenhando
                                        else Qt.OpenHandCursor)
        if not self._desenhando:
            self._cancelar_talhao_atual()

    def _cancelar_talhao_atual(self):
        if self._t_tmp is not None:
            self.cena.removeItem(self._t_tmp)
            self._t_tmp = None
        self._t_atual = []

    def _clique_talhao(self, p):
        if not self._desenhando or self.limites is None:
            return
        self._t_atual.append(p)
        self._atualizar_tmp_talhao(p)

    def _atualizar_tmp_talhao(self, cursor):
        if not self._t_atual:
            return
        cam = QPainterPath()
        cam.moveTo(self._t_atual[0])
        for q in self._t_atual[1:]:
            cam.lineTo(q)
        cam.lineTo(cursor)
        if self._t_tmp is None:
            pen = QPen(QColor("#ffffff"), 1.6, Qt.DashLine)
            pen.setCosmetic(True)
            self._t_tmp = self.cena.addPath(cam, pen)
            self._t_tmp.setZValue(6)
        else:
            self._t_tmp.setPath(cam)

    def _fechar_talhao(self):
        if not self._desenhando:
            return
        if len(self._t_atual) < 3:
            self._cancelar_talhao_atual()
            return
        from PySide6.QtGui import QPolygonF
        poly = QPolygonF(self._t_atual)
        arr = np.array([[q.x(), q.y()] for q in self._t_atual])
        pen = QPen(QColor("#ffffff"), 1.8)
        pen.setCosmetic(True)
        if self._modo_excl:
            item = self.cena.addPolygon(poly, pen, QColor(198, 40, 40, 80))
            self._t_excl.append(arr)
            self._historico.append(("e", item, len(self._t_excl) - 1))
        else:
            item = self.cena.addPolygon(poly, pen, QColor(46, 125, 50, 70))
            self._t_polys.append(arr)
            self._historico.append(("t", item, len(self._t_polys) - 1))
        item.setZValue(5)
        self._t_items.append(item)
        self._cancelar_talhao_atual()
        self.bt_salvar_t.setEnabled(len(self._t_polys) > 0)
        self.bt_limpar_t.setEnabled(True)
        self.bt_desfazer.setEnabled(True)
        self.lb_status.setText(f"{len(self._t_polys)} talhão(ões) | "
                               f"{len(self._t_excl)} exclusão(ões) — Salvar "
                               f"recorta as exclusões dos talhões")

    def desfazer_poligono(self):
        if not self._historico:
            return
        tipo, item, _ = self._historico.pop()
        self.cena.removeItem(item)
        self._t_items.remove(item)
        if tipo == "t":
            self._t_polys.pop()
        else:
            self._t_excl.pop()
        self.bt_desfazer.setEnabled(bool(self._historico))
        self.bt_salvar_t.setEnabled(len(self._t_polys) > 0)
        self.lb_status.setText(f"{len(self._t_polys)} talhão(ões) | "
                               f"{len(self._t_excl)} exclusão(ões)")

    def limpar_talhoes(self):
        for it in self._t_items:
            self.cena.removeItem(it)
        self._t_items, self._t_polys, self._t_excl = [], [], []
        self._historico = []
        self._cancelar_talhao_atual()
        self.bt_salvar_t.setEnabled(False)
        self.bt_limpar_t.setEnabled(False)
        self.bt_desfazer.setEnabled(False)

    def _cena_para_geo(self, arr):
        l, b, r, t = self.limites
        W, H = self.tam
        x = l + arr[:, 0] / W * (r - l)
        y = t - arr[:, 1] / H * (t - b)
        return np.column_stack([x, y])

    def salvar_talhoes(self, caminho=None):
        if not self._t_polys or self.limites is None:
            return None
        if caminho is None:
            from PySide6.QtWidgets import QFileDialog
            base = os.path.dirname(self.caminho_tif or "")
            caminho, _ = QFileDialog.getSaveFileName(
                self, "Salvar talhões",
                os.path.join(base, "talhoes_desenhados.gpkg"),
                "GeoPackage (*.gpkg)")
            if not caminho:
                return None
        import geopandas as gpd
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        buraco = None
        if self._t_excl:
            buraco = unary_union([Polygon(self._cena_para_geo(a)).buffer(0)
                                  for a in self._t_excl])
        registros = []
        for i, a in enumerate(self._t_polys):
            g = Polygon(self._cena_para_geo(a)).buffer(0)
            if buraco is not None:
                g = g.difference(buraco)
            if g.is_empty:
                continue
            registros.append({"talhao": str(i + 1), "geometry": g})
        if not registros:
            self.lb_status.setText("Nada a salvar: exclusões cobriram tudo")
            return None
        gpd.GeoDataFrame(registros, crs=self.crs).to_file(
            caminho, layer="talhoes", driver="GPKG")
        self.lb_status.setText(f"{len(registros)} talhão(ões) salvos")
        self.talhoes_salvos.emit(caminho)
        return caminho

    def carregar_ref(self):
        """Abre diálogo para escolher shapefiles de referência externa."""
        from PySide6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QLabel
        shp_l, _ = QFileDialog.getOpenFileName(
            self, "Shapefile de LINHAS da referência",
            "", "Shapefile (*.shp);;GeoPackage (*.gpkg);;Todos (*.*)")
        shp_f, _ = QFileDialog.getOpenFileName(
            self, "Shapefile de FALHAS da referência (opcional — cancele se não tiver)",
            "", "Shapefile (*.shp);;GeoPackage (*.gpkg);;Todos (*.*)")
        if not shp_l and not shp_f:
            return
        self.lb_status.setText("Carregando referência externa…")
        crs = getattr(self, "crs", None)
        th = _CarregaRef(shp_l or None, shp_f or None, crs)
        th.pronto.connect(self._ref_pronta)
        th.start()
        self._threads.append(th)

    def _ref_pronta(self, pacote):
        dados, erro = pacote
        if erro:
            self.lb_status.setText(f"Ref. externa: {erro}")
            return
        for camada, arrays in dados.items():
            self.vetores[camada] = arrays
        self._redesenhar()
        nl = len(dados.get("ref_linhas", []))
        nf = len(dados.get("ref_falhas", []))
        self.lb_status.setText(
            f"Ref. carregada: {nl} linhas | {nf} falhas  "
            f"(🟡 nosso | 🔵 ref. linhas | 🟣 ref. falhas)")

    def medir_espacamento(self):
        if not self.caminho_tif or self.limites is None:
            return
        rect = self.vista.mapToScene(
            self.vista.viewport().rect()).boundingRect()
        rect = rect.intersected(self.cena.sceneRect())
        l, b, r, t = self.limites
        W, H = self.tam
        gl = l + rect.left() / W * (r - l)
        gr = l + rect.right() / W * (r - l)
        gt = t - rect.top() / H * (t - b)
        gb = t - rect.bottom() / H * (t - b)
        self.bt_esp.setEnabled(False)
        self.lb_status.setText("Medindo espaçamento…")
        th = _MedidorEspacamento(self.caminho_tif, (gl, gb, gr, gt))
        th.pronto.connect(self._espacamento_pronto)
        th.start()
        self._threads.append(th)

    def _espacamento_pronto(self, pacote):
        esp, ang, conf, erro = pacote
        self.bt_esp.setEnabled(True)
        if erro:
            self.lb_status.setText(f"Espaçamento: {erro}")
            return
        if conf >= 0.25:
            self.lb_status.setText(
                f"Espaçamento detectado: {esp:.2f} m | fileiras a {ang:.0f}°")
            self.espacamento_detectado.emit(float(esp))
        else:
            self.lb_status.setText(
                f"Padrão fraco aqui ({esp:.2f} m?) — aproxime numa área de "
                f"fileiras nítidas e tente de novo")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not hasattr(self, "_botoes_barra"):
            return
        so_icone = self.width() < 620
        for wdg, texto, icone in self._botoes_barra:
            novo = icone if so_icone else texto
            if wdg.text() != novo:
                wdg.setText(novo)

    def _mostrar_coord(self, p):
        if self.limites is None:
            return
        l, b, r, t = self.limites
        W, H = self.tam
        x = l + p.x() / W * (r - l)
        y = t - p.y() / H * (t - b)
        self.lb_coord.setText(f"{x:,.1f}  {y:,.1f}")
