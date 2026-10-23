"""Interface gráfica do CanaScan (PySide6).

Execução:  python -m canascan.gui

A GUI é uma casca sobre a CLI: monta o comando e executa num QProcess,
transmitindo o log em tempo real. O motor não muda em nada — e este é o
mesmo padrão que o futuro plugin QGIS usará (chamar a CLI por baixo).
"""
import os
import re
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QSplitter
from PySide6.QtGui import QIcon
from PySide6.QtGui import QAction, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QTabWidget,
    QVBoxLayout, QWidget)

from . import __version__, edicao
from .mapa import PainelMapa

if getattr(sys, "frozen", False):      # rodando como executável empacotado
    RAIZ = os.path.dirname(sys.executable)
else:
    RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QSS = """
QMainWindow, QWidget { background:#20262b; color:#dfe6ea; font-size:12px; }
QGroupBox { border:1px solid #3a444c; border-radius:3px; margin-top:12px;
            padding:10px 8px 8px 8px; font-weight:bold; }
QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px;
                   color:#7fd18a; }
QLabel { color:#c9d3d9; }
QLineEdit, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background:#14181c; border:1px solid #3a444c; border-radius:2px;
    padding:3px 5px; color:#e8eef1; selection-background-color:#1d8a34; }
QComboBox QAbstractItemView { background:#14181c; color:#e8eef1; }
QPushButton { background:#2e3940; border:1px solid #46525b; border-radius:3px;
              padding:6px 12px; }
QPushButton:hover:!disabled { background:#3a4750; }
QPushButton:disabled { color:#66717a; }
QPushButton#processar { background:#1d8a34; color:#fff; font-weight:bold;
                        font-size:13px; padding:9px; }
QPushButton#comparar  { background:#2456a8; color:#fff; font-weight:bold;
                        font-size:13px; padding:9px; }
QPushButton#cancelar  { background:#7a2a2a; color:#fff; }
QPushButton[compacto="true"] { padding:3px 8px; font-size:11px; }
QProgressBar { background:#14181c; border:1px solid #3a444c; height:16px;
               text-align:center; color:#dfe6ea; }
QProgressBar::chunk { background:#1d8a34; }
QTabWidget::pane { border:1px solid #3a444c; }
QTabBar::tab { background:#2e3940; padding:7px 18px; margin-right:2px; }
QTabBar::tab:selected { background:#1d8a34; color:#fff; font-weight:bold; }
QMenuBar { background:#20262b; } QMenuBar::item:selected { background:#3a4750; }
QMenu { background:#2e3940; } QMenu::item:selected { background:#1d8a34; }
"""

LIMIAR_PADRAO = {"gli": 0.02, "razao": 1.5}


class CampoArquivo(QWidget):
    """Campo de texto + botão Procurar (arquivo, pasta ou múltiplos)."""

    def __init__(self, filtro="Todos (*.*)", pasta=False, rotulo="…",
                 multiplo=False):
        super().__init__()
        self.filtro, self.pasta, self.multiplo = filtro, pasta, multiplo
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.ed = QLineEdit()
        bt = QPushButton(rotulo)
        bt.setFixedWidth(30)
        bt.clicked.connect(self._procurar)
        lay.addWidget(self.ed)
        lay.addWidget(bt)

    def _procurar(self):
        if self.pasta:
            c = QFileDialog.getExistingDirectory(self, "Escolher pasta")
        elif self.multiplo:
            arqs, _ = QFileDialog.getOpenFileNames(
                self, "Escolher arquivo(s) — pode selecionar vários",
                "", self.filtro)
            c = "; ".join(arqs) if arqs else ""
        else:
            c, _ = QFileDialog.getOpenFileName(self, "Escolher arquivo",
                                               "", self.filtro)
        if c:
            self.ed.setText(c)

    def caminho(self):
        return self.ed.text().strip()


def _spin(minimo, maximo, valor, passo=0.05, casas=2):
    s = QDoubleSpinBox()
    s.setRange(minimo, maximo)
    s.setValue(valor)
    s.setSingleStep(passo)
    s.setDecimals(casas)
    return s


class AbaProcessar(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)

        g1 = QGroupBox("Arquivos")
        f1 = QGridLayout(g1)
        self.tif = CampoArquivo("GeoTIFF (*.tif *.tiff);;Todos (*.*)")
        self.talhoes = CampoArquivo("Vetores (*.gpkg *.shp *.geojson);;Todos (*.*)")
        self.linhas = CampoArquivo("Vetores (*.gpkg *.shp *.geojson);;Todos (*.*)")
        self.saida = CampoArquivo(pasta=True)
        self.saida.ed.setText("resultados")
        f1.addWidget(QLabel("Imagem de drone (GeoTIFF)*"), 0, 0)
        f1.addWidget(self.tif, 0, 1)
        f1.addWidget(QLabel("Talhões (opcional)"), 1, 0)
        f1.addWidget(self.talhoes, 1, 1)
        f1.addWidget(QLabel("Linhas prontas (opcional)"), 2, 0)
        f1.addWidget(self.linhas, 2, 1)
        f1.addWidget(QLabel("Pasta de saída"), 3, 0)
        f1.addWidget(self.saida, 3, 1)
        v.addWidget(g1)

        g2 = QGroupBox("Detecção")
        f2 = QGridLayout(g2)
        self.metodo = QComboBox()
        self.metodo.addItems(["unet", "gli", "razao", "exg_otsu"])
        self.limiar = _spin(0.0, 10.0, 0.02, 0.005, 3)
        self.metodo.currentTextChanged.connect(self._metodo_mudou)
        self.espacamento = _spin(0.2, 5.0, 1.5)
        self.falha_min = _spin(0.1, 2.0, 0.5)
        self.comp_copa = _spin(-0.60, 1.0, 0.25)
        self.comp_copa.setToolTip(
            "Ajuste da largura da planta.\n"
            "POSITIVO: a máscara cobre mais que a planta (folhas sobre o vão)\n"
            "NEGATIVO: a máscara vê menos que a planta (cana jovem, touceira\n"
            "  com miolo verde pequeno) — desconta o raio da touceira")
        self.min_linha = _spin(3.0, 60.0, 5.0, 1.0, 0)
        self.sensibilidade = _spin(0.5, 2.0, 1.0, 0.1, 1)
        self.sensibilidade.setToolTip("Recuperação de fileira fraca. "
                                      "Voo alto ou cana rala: aumente "
                                      "(1,3–1,6). Imagem com mato: diminua.")
        f2.addWidget(QLabel("Índice de vegetação"), 0, 0)
        f2.addWidget(self.metodo, 0, 1)
        f2.addWidget(QLabel("Limiar"), 0, 2)
        f2.addWidget(self.limiar, 0, 3)
        f2.addWidget(QLabel("Espaçamento (m)"), 1, 0)
        f2.addWidget(self.espacamento, 1, 1)
        f2.addWidget(QLabel("Falha mínima (m)"), 1, 2)
        f2.addWidget(self.falha_min, 1, 3)
        f2.addWidget(QLabel("Comp. de copa (m)"), 2, 0)
        f2.addWidget(self.comp_copa, 2, 1)
        f2.addWidget(QLabel("Linha mínima (m)"), 2, 2)
        f2.addWidget(self.min_linha, 2, 3)
        f2.addWidget(QLabel("Sensibilidade"), 3, 0)
        f2.addWidget(self.sensibilidade, 3, 1)
        v.addWidget(g2)

        g3 = QGroupBox("Processamento")
        f3 = QGridLayout(g3)
        self.modo_blocos = QComboBox()
        self.modo_blocos.addItems(["Automático", "Forçar grade", "Imagem única"])
        self.tam_grade = _spin(50, 1000, 250, 50, 0)
        self.banda = _spin(1.0, 20.0, 4.0, 0.5, 1)
        self.pad = _spin(0.0, 30.0, 6.0, 1.0, 0)
        self.chk_replantio = QCheckBox("Gerar análise de replantio | falha mínima (m):")
        self.chk_replantio.setChecked(True)
        self.min_replantio = _spin(0.5, 10.0, 2.0)
        self.chk_daninhas = QCheckBox("Analisar daninhas na entrelinha")
        self.chk_daninhas.setToolTip("Mapeia vegetação fora das linhas de "
                                     "plantio (mato) — vira camada no mapa "
                                     "e no resultado.gpkg")
        self.chk_overlays = QCheckBox("Gerar overlay PNG por bloco (mais lento)")
        self.chk_adaptativo = QCheckBox(
            "Linhas adaptativas — plantio em curva de nível (mais lento, mais preciso)")
        self.chk_adaptativo.setToolTip(
            "Analisa o talhão em pedaços, medindo a orientação LOCAL de cada "
            "trecho. Use quando as fileiras mudam de direção ao longo do "
            "talhão (curva de nível) — as linhas deixam de cruzar/dobrar/grudar.")
        f3.addWidget(QLabel("Blocos"), 0, 0)
        f3.addWidget(self.modo_blocos, 0, 1)
        f3.addWidget(QLabel("Tamanho do bloco (m)"), 0, 2)
        f3.addWidget(self.tam_grade, 0, 3)
        f3.addWidget(QLabel("Banda do rastreador (m)"), 1, 0)
        f3.addWidget(self.banda, 1, 1)
        f3.addWidget(QLabel("Folga da janela (m)"), 1, 2)
        f3.addWidget(self.pad, 1, 3)
        f3.addWidget(self.chk_replantio, 2, 0, 1, 3)
        f3.addWidget(self.min_replantio, 2, 3)
        f3.addWidget(self.chk_daninhas, 3, 0, 1, 4)
        f3.addWidget(self.chk_overlays, 4, 0, 1, 4)
        f3.addWidget(self.chk_adaptativo, 5, 0, 1, 4)
        v.addWidget(g3)

        self.bt = QPushButton("▶  PROCESSAR")
        self.bt.setObjectName("processar")
        v.addWidget(self.bt)
        v.addStretch(1)

    def _metodo_mudou(self, metodo):
        self.limiar.setEnabled(metodo != "exg_otsu")
        if metodo in LIMIAR_PADRAO:
            self.limiar.setValue(LIMIAR_PADRAO[metodo])

    def montar_comando(self):
        if not self.tif.caminho():
            return None, "Escolha a imagem de drone (GeoTIFF)."
        if not os.path.isfile(self.tif.caminho()):
            return None, "Imagem de drone não encontrada."
        cmd = ["-m", "canascan.cli", "processar",
               "--tif", self.tif.caminho(),
               "--saida", self.saida.caminho() or "resultados",
               "--metodo", self.metodo.currentText(),
               "--espacamento", str(self.espacamento.value()),
               "--falha-min", str(self.falha_min.value()),
               "--comp-copa", str(self.comp_copa.value()),
               "--banda", str(self.banda.value()),
               "--min-linha", str(self.min_linha.value()),
               "--sensibilidade", str(self.sensibilidade.value()),
               "--pad", str(self.pad.value())]
        if self.metodo.currentText() != "exg_otsu":
            cmd += ["--limiar", str(self.limiar.value())]
        if self.talhoes.caminho():
            cmd += ["--talhoes", self.talhoes.caminho()]
        if self.linhas.caminho():
            cmd += ["--linhas", self.linhas.caminho()]
        modo = self.modo_blocos.currentText()
        if modo == "Forçar grade":
            cmd += ["--grade", str(self.tam_grade.value())]
        elif modo == "Imagem única":
            cmd += ["--sem-grade"]
        if self.chk_replantio.isChecked():
            cmd += ["--replantio", str(self.min_replantio.value())]
        if self.chk_daninhas.isChecked():
            cmd += ["--daninhas"]
        if self.chk_overlays.isChecked():
            cmd += ["--overlays"]
        if self.chk_adaptativo.isChecked():
            cmd += ["--adaptativo"]
        return cmd, None


class AbaComparar(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        g = QGroupBox("Comparação com o software de referência")
        f = QGridLayout(g)
        self.nosso = CampoArquivo("GeoPackage (*.gpkg)")
        self.ref_linhas = CampoArquivo(
            "Vetores (*.shp *.gpkg *.geojson);;Todos (*.*)", multiplo=True)
        self.ref_falhas = CampoArquivo(
            "Vetores (*.shp *.gpkg *.geojson);;Todos (*.*)", multiplo=True)
        self.saida = CampoArquivo(pasta=True)
        self.saida.ed.setText("comparacao")
        self.nome_ref = QLineEdit("X-Farming")
        self.tol = _spin(0.1, 3.0, 0.5)
        f.addWidget(QLabel("Nosso resultado (resultado.gpkg)*"), 0, 0)
        f.addWidget(self.nosso, 0, 1)
        f.addWidget(QLabel("Linhas da referência"), 1, 0)
        f.addWidget(self.ref_linhas, 1, 1)
        f.addWidget(QLabel("Falhas da referência"), 2, 0)
        f.addWidget(self.ref_falhas, 2, 1)
        f.addWidget(QLabel("Pasta de saída"), 3, 0)
        f.addWidget(self.saida, 3, 1)
        f.addWidget(QLabel("Nome da referência"), 4, 0)
        f.addWidget(self.nome_ref, 4, 1)
        f.addWidget(QLabel("Tolerância espacial (m)"), 5, 0)
        f.addWidget(self.tol, 5, 1)
        v.addWidget(g)
        self.bt = QPushButton("⇄  COMPARAR")
        self.bt.setObjectName("comparar")
        v.addWidget(self.bt)
        v.addStretch(1)

    def montar_comando(self):
        if not self.nosso.caminho():
            return None, "Escolha o resultado.gpkg do CanaScan."
        if not (self.ref_linhas.caminho() or self.ref_falhas.caminho()):
            return None, "Escolha ao menos uma camada da referência."
        cmd = ["-m", "canascan.cli", "comparar",
               "--nosso", self.nosso.caminho(),
               "--saida", self.saida.caminho() or "comparacao",
               "--tolerancia", str(self.tol.value()),
               "--nome-ref", self.nome_ref.text().strip() or "Referência"]
        if self.ref_linhas.caminho():
            cmd += ["--ref-linhas", self.ref_linhas.caminho()]
        if self.ref_falhas.caminho():
            cmd += ["--ref-falhas", self.ref_falhas.caminho()]
        return cmd, None


class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{edicao.NOME} v{__version__} — Detecção de "
                            f"Linhas e Falhas")
        _logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
        if os.path.isfile(_logo_path):
            self.setWindowIcon(QIcon(_logo_path))
        self.resize(1100, 700)
        self.setMinimumSize(960, 580)
        self.proc = None
        self.total_blocos = None
        self.blocos_feitos = 0
        self.pasta_resultado = None

        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)

        # coluna esquerda: abas de formulário
        self.abas = QTabWidget()
        self.aba_proc = AbaProcessar()
        self.aba_comp = AbaComparar()
        self.abas.addTab(self.aba_proc, "Processar")
        if edicao.tem("comparar"):
            self.abas.addTab(self.aba_comp, "Comparar")
        self.abas.setMaximumWidth(600)
        self.rolagem = QScrollArea()
        rolagem = self.rolagem
        rolagem.setWidget(self.abas)
        rolagem.setWidgetResizable(True)
        rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        rolagem.setFrameShape(QScrollArea.NoFrame)
        rolagem.setMinimumWidth(300)
        rolagem.setMaximumWidth(600)
        h.addWidget(rolagem)

        # coluna direita: mapa + log + progresso + resultados
        dir_ = QVBoxLayout()
        self.mapa = PainelMapa()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas, DejaVu Sans Mono", 9))
        self.log.setPlaceholderText("O log do processamento aparece aqui…")
        divisor = QSplitter(Qt.Vertical)
        divisor.addWidget(self.mapa)
        divisor.addWidget(self.log)
        divisor.setSizes([480, 170])
        dir_.addWidget(divisor, 1)
        self.barra = QProgressBar()
        self.barra.setValue(0)
        dir_.addWidget(self.barra)

        lin_bt = QHBoxLayout()
        self.bt_cancelar = QPushButton("Cancelar")
        self.bt_cancelar.setObjectName("cancelar")
        self.bt_cancelar.setEnabled(False)
        self.bt_rel_falhas = QPushButton("Relatório de falhas")
        self.bt_rel_rep = QPushButton("Relatório de replantio")
        self.bt_rel_comp = QPushButton("Comparação")
        self.bt_pasta = QPushButton("Abrir pasta")
        for b in (self.bt_rel_falhas, self.bt_rel_rep, self.bt_rel_comp,
                  self.bt_pasta):
            b.setEnabled(False)
        lin_bt.addWidget(self.bt_cancelar)
        lin_bt.addStretch(1)
        lin_bt.addWidget(self.bt_rel_falhas)
        lin_bt.addWidget(self.bt_rel_rep)
        lin_bt.addWidget(self.bt_rel_comp)
        lin_bt.addWidget(self.bt_pasta)
        dir_.addLayout(lin_bt)
        h.addLayout(dir_, 1)

        # menu
        self._menu_ferramentas = self.menuBar().addMenu("Ferramentas")
        m = self._menu_ferramentas
        acao = QAction("Gerar campo de teste (sintético)…", self)
        acao.triggered.connect(self.gerar_sintetico)
        m.addAction(acao)
        m.addSeparator()
        acao_ds = QAction("Gerar dataset de treino (rede neural)…", self)
        acao_ds.triggered.connect(self.gerar_dataset)
        m.addAction(acao_ds)
        self._menu_exibir = self.menuBar().addMenu("Exibir")
        self.acao_apres = QAction("Modo apresentação", self)
        self.acao_apres.setCheckable(True)
        self.acao_apres.setShortcut("F11")
        self.acao_apres.toggled.connect(self._modo_apresentacao)
        self._menu_exibir.addAction(self.acao_apres)

        # conexões
        self.aba_proc.bt.clicked.connect(self.rodar_processar)
        self.aba_comp.bt.clicked.connect(self.rodar_comparar)
        self.bt_cancelar.clicked.connect(self.cancelar)
        self.bt_rel_falhas.clicked.connect(
            lambda: self._abrir("relatorio_falhas.html"))
        self.bt_rel_rep.clicked.connect(
            lambda: self._abrir("relatorio_replantio.html"))
        self.bt_rel_comp.clicked.connect(lambda: self._abrir("comparacao.html"))
        self.bt_pasta.clicked.connect(lambda: self._abrir(""))

        # módulos por edição: esconde o que não está liberado
        if not edicao.tem("daninhas"):
            self.aba_proc.chk_daninhas.setVisible(False)
            self.aba_proc.chk_daninhas.setChecked(False)
        if not edicao.tem("falhas"):
            self.aba_proc.chk_replantio.setVisible(False)
            self.aba_proc.chk_replantio.setChecked(False)
            self.aba_proc.min_replantio.setVisible(False)

        # prévia automática ao escolher o GeoTIFF (com debounce)
        self._timer_tif = QTimer(self)
        self._timer_tif.setSingleShot(True)
        self._timer_tif.timeout.connect(self._tif_mudou)
        self.aba_proc.tif.ed.textChanged.connect(
            lambda: self._timer_tif.start(700))
        self.mapa.espacamento_detectado.connect(self._aplicar_espacamento)
        self.mapa.talhoes_salvos.connect(self.aba_proc.talhoes.ed.setText)

        self.statusBar().showMessage(f"CanaScan v{__version__} pronto.")

    def _aplicar_espacamento(self, esp):
        r = QMessageBox.question(
            self, "CanaScan",
            f"Detectei espaçamento de {esp:.2f} m entre as fileiras "
            f"desta região.\n\nAplicar no campo \"Espaçamento\"?")
        if r == QMessageBox.Yes:
            self.aba_proc.espacamento.setValue(round(esp, 2))

    def _tif_mudou(self):
        caminho = self.aba_proc.tif.caminho()
        if caminho and os.path.isfile(caminho):
            self.mapa.carregar_tif(caminho)

    # ---------- execução ----------
    def _rodar(self, cmd, pasta_saida):
        if self.proc is not None:
            return
        self.log.clear()
        self.total_blocos, self.blocos_feitos = None, 0
        self.barra.setRange(0, 0)  # indeterminado até saber o nº de blocos
        self.pasta_resultado = pasta_saida
        for b in (self.bt_rel_falhas, self.bt_rel_rep, self.bt_rel_comp,
                  self.bt_pasta):
            b.setEnabled(False)
        self.aba_proc.bt.setEnabled(False)
        self.aba_comp.bt.setEnabled(False)
        self.bt_cancelar.setEnabled(True)
        self.statusBar().showMessage("Executando…")

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(RAIZ)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._ler_saida)
        self.proc.finished.connect(self._terminou)
        if getattr(sys, "frozen", False):
            # empacotado: chama o executável CLI irmão (console) com o
            # subcomando direto, sem o prefixo "-m canascan.cli"
            if cmd[:2] == ["-m", "canascan.cli"]:
                cmd = cmd[2:]
            pasta_exe = os.path.dirname(sys.executable)
            nome_cli = "CanaScanCLI.exe" if os.name == "nt" else "CanaScanCLI"
            programa = os.path.join(pasta_exe, nome_cli)
            if not os.path.isfile(programa):
                programa = sys.executable  # reserva: o próprio exe despacha
            self.proc.start(programa, cmd)
        else:
            self.proc.start(sys.executable, cmd)

    def rodar_processar(self):
        cmd, erro = self.aba_proc.montar_comando()
        if erro:
            QMessageBox.warning(self, "CanaScan", erro)
            return
        self._rodar(cmd, self.aba_proc.saida.caminho() or "resultados")

    def rodar_comparar(self):
        cmd, erro = self.aba_comp.montar_comando()
        if erro:
            QMessageBox.warning(self, "CanaScan", erro)
            return
        self._rodar(cmd, self.aba_comp.saida.caminho() or "comparacao")

    def gerar_sintetico(self):
        pasta = QFileDialog.getExistingDirectory(self, "Pasta p/ dados de teste")
        if not pasta:
            return
        destino = os.path.join(pasta, "dados_teste")
        self._rodar(["-m", "canascan.cli", "sintetico", "--saida", destino,
                     "--largura", "300", "--altura", "200"], destino)
        self.aba_proc.tif.ed.setText(os.path.join(destino, "campo.tif"))
        self.aba_proc.talhoes.ed.setText(os.path.join(destino, "talhoes.gpkg"))
        self.aba_proc.metodo.setCurrentText("razao")

    def _modo_apresentacao(self, ligado):
        """F11: só o mapa em tela cheia — perfeito p/ demonstrar."""
        self.rolagem.setVisible(not ligado)
        self.log.setVisible(not ligado)
        self.menuBar().setVisible(not ligado)
        if ligado:
            self.showFullScreen()
            self.statusBar().showMessage(
                "Modo apresentação — F11 ou Esc para sair")
        else:
            self.showNormal()
            self.statusBar().showMessage("Pronto.")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape and self.acao_apres.isChecked():
            self.acao_apres.setChecked(False)
        else:
            super().keyPressEvent(ev)

    def gerar_dataset(self):
        tif = self.aba_proc.tif.caminho()
        if not tif or not os.path.isfile(tif):
            QMessageBox.warning(self, "CanaScan",
                                "Escolha primeiro a imagem de drone na aba Processar.")
            return
        gpkg, _ = QFileDialog.getOpenFileName(
            self, "Escolha o resultado.gpkg REVISADO (linhas corrigidas)",
            os.path.dirname(tif), "GeoPackage (*.gpkg)")
        if not gpkg:
            return
        pasta = QFileDialog.getExistingDirectory(
            self, "Pasta do dataset (nova ou existente p/ acrescentar)")
        if not pasta:
            return
        existe = os.path.isfile(os.path.join(pasta, "dataset.json"))
        cmd = ["-m", "canascan.cli", "dataset", "--tif", tif,
               "--gpkg", gpkg, "--saida", pasta]
        if existe:
            r = QMessageBox.question(
                self, "CanaScan",
                "Já existe um dataset nesta pasta.\n\nAcrescentar os novos "
                "tiles a ele? (Sim = soma | Não = recria do zero)")
            if r == QMessageBox.Yes:
                cmd.append("--acrescentar")
        self._rodar(cmd, pasta)

    def cancelar(self):
        if self.proc is not None:
            self.proc.kill()

    def _ler_saida(self):
        texto = self.proc.readAllStandardOutput().data().decode(
            "utf-8", errors="replace")
        for linha in texto.splitlines():
            self.log.appendPlainText(linha.rstrip())
            m = re.search(r"—\s*(\d+)\s*blocos", linha) or \
                re.search(r"\]\s*(\d+)\s*talhões", linha)
            if m:
                self.total_blocos = int(m.group(1))
                self.barra.setRange(0, self.total_blocos)
                self.barra.setValue(0)
            elif re.match(r"^\s{2}\[", linha) and self.total_blocos:
                self.blocos_feitos = min(self.blocos_feitos + 1,
                                         self.total_blocos)
                self.barra.setValue(self.blocos_feitos)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _terminou(self, codigo, _status):
        self.proc = None
        self.aba_proc.bt.setEnabled(True)
        self.aba_comp.bt.setEnabled(True)
        self.bt_cancelar.setEnabled(False)
        self.barra.setRange(0, 1)
        self.barra.setValue(1 if codigo == 0 else 0)
        p = self.pasta_resultado or ""
        existe = lambda nome: os.path.isfile(os.path.join(RAIZ, p, nome)) or \
            os.path.isfile(os.path.join(p, nome))
        self.bt_rel_falhas.setEnabled(existe("relatorio_falhas.html"))
        self.bt_rel_rep.setEnabled(existe("relatorio_replantio.html"))
        self.bt_rel_comp.setEnabled(existe("comparacao.html"))
        self.bt_pasta.setEnabled(bool(p))
        gpkg = os.path.join(p, "resultado.gpkg")
        gpkg_abs = gpkg if os.path.isfile(gpkg) else os.path.join(RAIZ, gpkg)
        if os.path.isfile(gpkg_abs):
            self.aba_comp.nosso.ed.setText(gpkg)
            self.mapa.carregar_vetores(gpkg_abs)  # desenha o resultado no mapa
        if codigo == 0:
            self.statusBar().showMessage("Concluído.")
            self.log.appendPlainText("\n✔ Concluído.")
        else:
            self.statusBar().showMessage(f"Terminou com erro (código {codigo}).")
            self.log.appendPlainText(f"\n✖ Terminou com erro (código {codigo}).")

    def closeEvent(self, ev):
        for t in getattr(self.mapa, "_threads", []):
            t.wait(2000)
        super().closeEvent(ev)

    def _abrir(self, nome):
        p = self.pasta_resultado or ""
        caminho = os.path.join(p, nome) if nome else p
        if not os.path.isabs(caminho):
            alt = os.path.join(RAIZ, caminho)
            caminho = alt if os.path.exists(alt) else os.path.abspath(caminho)
        QDesktopServices.openUrl(QUrl.fromLocalFile(caminho))


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = JanelaPrincipal()
    win.show()
    if "--screenshot" in sys.argv:  # uso interno p/ testes
        destino = sys.argv[sys.argv.index("--screenshot") + 1]
        QTimer.singleShot(400, lambda: (win.grab().save(destino), app.quit()))
    if "--autoteste" in sys.argv:   # smoke test do executável: processa um
        tif = sys.argv[sys.argv.index("--autoteste") + 1]  # tif e sai
        win.aba_proc.tif.ed.setText(tif)
        win.aba_proc.saida.ed.setText(os.path.join(
            os.path.dirname(tif), "autoteste_saida"))
        win.aba_proc.metodo.setCurrentText("razao")
        win.aba_proc.modo_blocos.setCurrentText("Imagem única")
        def verificar():
            if win.proc is not None:
                QTimer.singleShot(300, verificar)
                return
            ok = "ÍNDICE DE FALHAS" in win.log.toPlainText()
            with open(os.path.join(os.path.dirname(tif),
                                   "autoteste_resultado.txt"), "w") as f:
                f.write("PASSOU" if ok else "FALHOU")
            for t in getattr(win.mapa, "_threads", []):
                t.wait(3000)
            app.exit(0 if ok else 1)
        QTimer.singleShot(800, lambda: (win.rodar_processar(),
                                        QTimer.singleShot(1500, verificar)))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
