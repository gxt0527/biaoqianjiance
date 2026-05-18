# -*- coding: utf-8 -*-
"""
checker_core.py - 印前检查核心模块
提供统一的图像读写接口，自动处理 CMYK/RGB 色彩空间、DPI 和 ICC Profile。
"""

import cv2, numpy as np, os
from PIL import Image as PILImage

# ── 模块级缓存 ─────────────────────────────────────────────
_original_image_info = {}  # path → {mode, icc, dpi, pil}


def imread(path: str, preserve: bool = False):
    """
    读取图像，自动缓存原始 CMYK/ICC/DPI 信息。

    Args:
        path: 图像路径（支持中文路径）
        preserve: True=返回BGR并缓存，False=直接返回BGR

    Returns:
        numpy.ndarray (BGR格式)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 用 PIL 打开，获取色彩空间和 ICC Profile
    pil_img = PILImage.open(path)

    # 缓存原始信息
    _original_image_info[path] = {
        'mode': pil_img.mode,          # 'CMYK' or 'RGB'
        'icc':  pil_img.info.get('icc_profile'),
        'dpi':  pil_img.info.get('dpi'),  # (300, 300)
        'pil':  pil_img.copy(),
    }

    # 转换为 RGB（PIL 标准格式）
    if pil_img.mode == 'CMYK':
        pil_rgb = pil_img.convert('RGB')
    elif pil_img.mode != 'RGB':
        pil_rgb = pil_img.convert('RGB')
    else:
        pil_rgb = pil_img.copy()

    # 转换为 BGR（OpenCV 标准格式）并返回
    return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)


def imwrite(path: str, bgr_img, original_path: str = None, dpi: tuple = None):
    """
    保存图像，自动还原原始色彩空间和 DPI。

    Args:
        path: 输出路径
        bgr_img: BGR格式图像
        original_path: 参考图像路径（用于获取 CMYK/ICC/DPI）
        dpi: 输出 DPI，如 (300, 300)
    """
    # 反转 BGR → RGB
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb_img)

    # 确定保存模式
    orig_mode = None
    orig_icc  = None
    save_dpi  = dpi

    if original_path and original_path in _original_image_info:
        info = _original_image_info[original_path]
        orig_mode = info['mode']
        orig_icc  = info['icc']
        if save_dpi is None:
            save_dpi = info['dpi']
    elif path in _original_image_info:
        info = _original_image_info[path]
        orig_mode = info['mode']
        orig_icc  = info['icc']
        if save_dpi is None:
            save_dpi = info['dpi']

    # 统一默认 DPI
    if save_dpi is None:
        save_dpi = (300, 300)

    # PNG 不支持 CMYK，自动转 RGB
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        if orig_mode == 'CMYK' and pil_img.mode == 'RGB':
            pil_img = pil_img  # PNG强制RGB
    else:
        # JPG/TIFF 等支持 CMYK
        if orig_mode == 'CMYK' and pil_img.mode == 'RGB':
            pil_img = pil_img.convert('CMYK')

    # 保存参数
    save_kwargs = {}
    if orig_icc:
        save_kwargs['icc_profile'] = orig_icc

    # 保存
    if ext in ('.jpg', '.jpeg'):
        pil_img.save(path, dpi=save_dpi, quality=95, subsampling=0, **save_kwargs)
    elif ext == '.png':
        pil_img.save(path, dpi=save_dpi, **save_kwargs)
    else:
        pil_img.save(path, dpi=save_dpi, **save_kwargs)


def get_image_info(path: str):
    """获取图像原始信息"""
    if path in _original_image_info:
        return _original_image_info[path]
    return None
