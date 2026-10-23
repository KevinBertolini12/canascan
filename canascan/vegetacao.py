"""Segmentação de vegetação (cana vs. solo) com integração da IA.

Métodos disponíveis:
- "unet": Utiliza a Rede Neural treinada com PyTorch.
- "razao": índice de verde 2G/(R+B) com limiar fixo.
- "gli": Green Leaf Index, (2G - R - B)/(2G + R + B).
- "exg_otsu": Excess Green (2G - R - B) com limiar automático de Otsu.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import os

LIMIARES_PADRAO = {"razao": 1.5, "gli": 0.02, "unet": 0.0}

def limiar_padrao(metodo):
    return LIMIARES_PADRAO.get(metodo, 0.0)

# =========================================================================
# INTEGRAÇÃO U-NET PYTORCH
# =========================================================================

def bloco(entra, sai):
    return nn.Sequential(
        nn.Conv2d(entra, sai, 3, padding=1, bias=False),
        nn.BatchNorm2d(sai), nn.ReLU(inplace=True),
        nn.Conv2d(sai, sai, 3, padding=1, bias=False),
        nn.BatchNorm2d(sai), nn.ReLU(inplace=True))

class UNet(nn.Module):
    """Arquitetura U-Net original para o carregamento correto dos pesos."""
    def __init__(self, classes=3, base=32):
        super().__init__()
        b = base
        self.d1 = bloco(3, b)
        self.d2 = bloco(b, b * 2)
        self.d3 = bloco(b * 2, b * 4)
        self.d4 = bloco(b * 4, b * 8)
        self.meio = bloco(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.u4 = bloco(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = bloco(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = bloco(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = bloco(b * 2, b)
        self.saida = nn.Conv2d(b, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.d1(x);            p1 = self.pool(c1)
        c2 = self.d2(p1);           p2 = self.pool(c2)
        c3 = self.d3(p2);           p3 = self.pool(c3)
        c4 = self.d4(p3);           p4 = self.pool(c4)
        m = self.meio(p4)
        x = self.u4(torch.cat([self.up4(m), c4], dim=1))
        x = self.u3(torch.cat([self.up3(x), c3], dim=1))
        x = self.u2(torch.cat([self.up2(x), c2], dim=1))
        x = self.u1(torch.cat([self.up1(x), c1], dim=1))
        return self.saida(x)

_modelo_unet_cache = None

def carregar_modelo_unet():
    """Carrega o modelo .pt apenas uma vez na memória."""
    global _modelo_unet_cache
    if _modelo_unet_cache is not None:
        return _modelo_unet_cache
    
    # Procura na raiz de execução ou na raiz do projeto (acima de canascan/)
    caminho_modelo = "modelo_cana.pt"
    if not os.path.isfile(caminho_modelo):
        caminho_modelo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelo_cana.pt")
        if not os.path.isfile(caminho_modelo):
            raise FileNotFoundError("O arquivo 'modelo_cana.pt' não foi encontrado na raiz do sistema.")
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(caminho_modelo, map_location=dev)
    
    base = ckpt.get("base", 32)
    classes = ckpt.get("classes", 3)
    modelo = UNet(classes=classes, base=base).to(dev)
    
    modelo.load_state_dict(ckpt["modelo"])
    modelo.eval()
    
    _modelo_unet_cache = (modelo, dev, ckpt.get("tamanho", 512))
    return _modelo_unet_cache

def mascara_unet(rgb, valido=None):
    """Gera a máscara binária utilizando a Rede Neural treinada."""
    modelo, dev, tile_size = carregar_modelo_unet()
    H, W, _ = rgb.shape
    
    # Adiciona borda refletida caso a imagem não seja múltipla de 512
    pad_y = (tile_size - (H % tile_size)) % tile_size
    pad_x = (tile_size - (W % tile_size)) % tile_size
    
    if pad_y > 0 or pad_x > 0:
        rgb_pad = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
    else:
        rgb_pad = rgb

    mask_pad = np.zeros((rgb_pad.shape[0], rgb_pad.shape[1]), dtype=bool)
    
    batch_imgs = []
    coords = []
    batch_max = 4 # Processa de 4 em 4 quadrados para poupar a memória do processador
    
    with torch.no_grad():
        for y in range(0, rgb_pad.shape[0], tile_size):
            for x in range(0, rgb_pad.shape[1], tile_size):
                tile = rgb_pad[y:y+tile_size, x:x+tile_size]
                tile_t = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 255.0
                
                batch_imgs.append(tile_t)
                coords.append((y, x))
                
                if len(batch_imgs) >= batch_max:
                    batch_t = torch.stack(batch_imgs).to(dev)
                    out = modelo(batch_t)
                    pred = out.argmax(dim=1).cpu().numpy()
                    
                    for i, (yy, xx) in enumerate(coords):
                        # Classe 1 é a linha de plantio real!
                        mask_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 1)
                        
                    batch_imgs = []
                    coords = []
                    
        # Roda o resíduo
        if len(batch_imgs) > 0:
            batch_t = torch.stack(batch_imgs).to(dev)
            out = modelo(batch_t)
            pred = out.argmax(dim=1).cpu().numpy()
            
            for i, (yy, xx) in enumerate(coords):
                mask_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 1)

    # Retorna o tamanho original cortando as bordas
    mask = mask_pad[:H, :W]
    
    if valido is not None:
        mask &= valido
        
    return mask


def mascara_unet_multiclasse(rgb, valido=None):
    """Retorna a máscara binária de linhas (Classe 1) e o mapa de falhas (Classe 2) previstos pela IA."""
    modelo, dev, tile_size = carregar_modelo_unet()
    H, W, _ = rgb.shape
    
    pad_y = (tile_size - (H % tile_size)) % tile_size
    pad_x = (tile_size - (W % tile_size)) % tile_size
    
    if pad_y > 0 or pad_x > 0:
        rgb_pad = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
    else:
        rgb_pad = rgb

    mask_pad = np.zeros((rgb_pad.shape[0], rgb_pad.shape[1]), dtype=bool)
    falhas_pad = np.zeros((rgb_pad.shape[0], rgb_pad.shape[1]), dtype=bool)
    
    batch_imgs = []
    coords = []
    batch_max = 4 
    
    with torch.no_grad():
        for y in range(0, rgb_pad.shape[0], tile_size):
            for x in range(0, rgb_pad.shape[1], tile_size):
                tile = rgb_pad[y:y+tile_size, x:x+tile_size]
                tile_t = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 255.0
                
                batch_imgs.append(tile_t)
                coords.append((y, x))
                
                if len(batch_imgs) >= batch_max:
                    batch_t = torch.stack(batch_imgs).to(dev)
                    out = modelo(batch_t)
                    pred = out.argmax(dim=1).cpu().numpy()
                    
                    for i, (yy, xx) in enumerate(coords):
                        mask_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 1)
                        falhas_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 2)
                        
                    batch_imgs = []
                    coords = []
                    
        if len(batch_imgs) > 0:
            batch_t = torch.stack(batch_imgs).to(dev)
            out = modelo(batch_t)
            pred = out.argmax(dim=1).cpu().numpy()
            
            for i, (yy, xx) in enumerate(coords):
                mask_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 1)
                falhas_pad[yy:yy+tile_size, xx:xx+tile_size] = (pred[i] == 2)

    mask = mask_pad[:H, :W]
    falhas = falhas_pad[:H, :W]
    
    if valido is not None:
        mask &= valido
        falhas &= valido
        
    return mask, falhas

# =========================================================================
# CÁLCULOS ANTIGOS DE MATEMÁTICA PURA (Mantidos por segurança)
# =========================================================================

def mascara_vegetacao(rgb, valido=None, metodo="razao", limiar=1.5,
                      limpar_px=3, bloco=2048):
    """Calcula a máscara em blocos de linhas: os temporários em float ficam
    do tamanho do bloco, não da imagem — essencial p/ ortomosaicos grandes."""
    H, W = rgb.shape[:2]
    if metodo == "exg_otsu":
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        exg = 2.0 * g - r - b
        exg_n = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, m = cv2.threshold(exg_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = m > 0
    elif metodo in ("razao", "gli"):
        mask = np.zeros((H, W), dtype=bool)
        for y0 in range(0, H, bloco):
            y1 = min(y0 + bloco, H)
            r = rgb[y0:y1, :, 0].astype(np.float32)
            g = rgb[y0:y1, :, 1].astype(np.float32)
            b = rgb[y0:y1, :, 2].astype(np.float32)
            if metodo == "razao":
                mask[y0:y1] = (2.0 * g) / (r + b + 1e-6) > float(limiar)
            else:
                mask[y0:y1] = ((2.0 * g - r - b) /
                               (2.0 * g + r + b + 1e-6)) > float(limiar)
    else:
        raise ValueError(f"método desconhecido: {metodo}")

    if valido is not None:
        mask &= valido

    # limpeza morfológica: remove ruído pontual e fecha buraquinhos
    if limpar_px and limpar_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (limpar_px, limpar_px))
        m8 = (mask.astype(np.uint8)) * 255
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
        mask = m8 > 0
    return mask


def indice_continuo(rgb, valido=None, metodo="gli"):
    """Índice de vegetação contínuo (float 0..1), sem binarizar.

    Com dossel fechado a máscara binária vira um tapete sem padrão, mas a
    INTENSIDADE do verde ainda ondula no ritmo das ruas — este índice
    preserva essa textura para a busca de orientação."""
    import numpy as np
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    
    # Como a U-Net não usa limiar, se ela for escolhida forçamos o cálculo
    # de continuidade baseado no método GLI que é o mais seguro matematicamente.
    if metodo == "razao":
        den = r + b + 1e-6
        idx = 2.0 * g / den
    else:  
        den = 2.0 * g + r + b + 1e-6
        idx = (2.0 * g - r - b) / den
        
    if valido is not None:
        idx = np.where(valido, idx, np.nan)
    lo, hi = np.nanpercentile(idx, [2, 98])
    idx = np.clip((idx - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(idx, nan=0.0).astype(np.float32)