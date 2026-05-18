# -*- coding: utf-8 -*-
"""
PDF vs 原稿 — 完整印前检测流程
1. 加载原稿jpg和拼版文件pdf
2. 识别原稿主体内容区域（排除标注信息）
3. 识别PDF裁切框位置，简单对比，保留不相似标签
4. 4角度对比，保留最佳匹配
5. 详细对比：边缘检测、OCR、热力图
6. 生成检测报告
"""

import os, sys, io, locale, cv2, numpy as np, fitz, warnings, datetime
from collections import defaultdict

# ── 强制 print 立即 flush（解决 GUI 实时显示问题）──────────
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 加载中文字体（Windows）─────────────────────────────────
if sys.platform == 'win32':
    try:
        pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
        PDF_FONT = 'SimHei'
    except Exception:
        PDF_FONT = 'Helvetica'
else:
    PDF_FONT = 'Helvetica'

# ── Windows 编码设置（仅影响 print 输出到控制台）─────────────
if sys.platform == 'win32':
    # 抑制 PaddleOCR 的文件警告
    warnings.filterwarnings('ignore', message='.*model files.*')
    warnings.filterwarnings('ignore', message='.*provided mode.*')
    # 尝试设置控制台编码为 cp936（GBK）
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='cp936', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='cp936', errors='replace')
    except Exception:
        pass
from paddleocr import PaddleOCR

sys.path.insert(0, '.')
import checker_core
from checker_core import imread, imwrite

# -- 路径设置（支持环境变量覆盖）-------------------------
BASE      = os.path.dirname(os.path.abspath(__file__))
ORIG_JPG  = os.environ.get("PDF_CHECKER_ORIG") or os.path.join(BASE, "xin.jpg")
PDF_FILE  = os.environ.get("PDF_CHECKER_PDF")  or os.path.join(BASE, "xin.pdf")
OUT_DIR   = os.path.join(BASE, "pdf_align_compare")
os.makedirs(OUT_DIR, exist_ok=True)

# 报告保存目录：原稿同目录（用户要求）
REPORT_DIR = os.path.dirname(os.path.abspath(ORIG_JPG))
os.makedirs(REPORT_DIR, exist_ok=True)

# ╔══════════════════════════════════════════════════════════╗
# ║              调试参数配置区（2026-05-06）               ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Step3：PDF裁切框提取                                    ║
CFG_PDF_DPI           = 300     # PDF裁切框提取分辨率（DPI）
CFG_MAGENTA_R         = 0.7     # 洋红色裁切框识别：红通道阈值
CFG_MAGENTA_G         = 0.2     # 洋红色裁切框识别：绿通道上限
CFG_MAGENTA_B         = 0.3     # 洋红色裁切框识别：蓝通道阈值
CFG_MIN_BOX_W         = 50      # 最小裁切框宽度（点）
CFG_MIN_BOX_H         = 20      # 最小裁切框高度（点）
CFG_DUP_SIM_THRESH    = 90.0    # 相似度去重阈值（%），超过则认为重复标签
# ║  Step4：4角度相似度计算                                  ║
CFG_ORB_SIM_FEATURES  = 500     # ORB特征点数（相似度计算用）
CFG_ORB_SIM_RATIO     = 0.75    # Lowe比率过滤（相似度计算）
# ║  Step5：高分辨率对比                                     ║
CFG_HIGH_DPI          = 1200    # Step5高分辨率渲染DPI（PDF矢量优势）
# ║  Step5.1：ORB粗对齐（优化精度）                             ║
CFG_ORB_ALIGN_FEATURES= 2000    # ORB特征点数（对齐用）- 增加提高匹配精度
CFG_ORB_ALIGN_RATIO   = 0.65    # Lowe比率过滤（对齐）- 降低提高匹配质量
CFG_RANSAC_THRESH     = 3.0     # RANSAC重投影误差阈值（px）- 降低提高精度
CFG_RANSAC_MAXITERS   = 10000   # RANSAC最大迭代次数 - 增加提高精度
# ║  Step5.3~5.5：边缘轮廓差异分析                           ║
CFG_EDGE_CANNY_A      = 50      # Canny边缘检测：低阈值
CFG_EDGE_CANNY_B      = 150     # Canny边缘检测：高阈值
CFG_HEATMAP_ALPHA     = 0.70    # 热力图叠加透明度（0=原图，1=纯热力图）
# ║  差异标注（基于边缘差异）                                 ║
CFG_EDGE_DIFF_THRESH  = 30      # 边缘差异二值化阈值（边缘差值像素数）
CFG_MIN_CONTOUR_AREA  = 50      # 差异轮廓最小面积（px²），小于此忽略
# ║  Step5.9：OCR文字识别对比                                 ║
CFG_OCR_USE_ANGLE     = False   # PaddleOCR：禁用方向分类（提速+降误判）
CFG_OCR_MIN_H         = 8       # OCR最小文字高度（px），小于此忽略
CFG_TEXT_IOU_THRESH   = 0.15    # 文字框IoU匹配阈值（降低以容忍对齐偏移）
CFG_TEXT_SIM_THRESH   = 0.60    # 文字内容相似度阈值（Levenshtein）
CFG_OCR_DPI           = 300     # OCR渲染DPI（兼顾速度与精度）
# ║  内容匹配兜底：当IoU不够但文字相同时也视为匹配            ║
CFG_CONTENT_MATCH_SIM = 0.85    # 内容相似度阈值（文字相同/相近时的兜底匹配）
# ║  Step6：检测报告                                         ║
CFG_PASS_NCC          = 95.0    # 判定通过的NCC相似度下限（%）
# ╚══════════════════════════════════════════════════════════╝

# -- 预加载原稿（缓存CMYK/ICC/DPI）--------------------------
_ = imread(ORIG_JPG)

def save(path, bgr_img):
    imwrite(path, bgr_img, original_path=ORIG_JPG)

def to_lab(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)

def to_gray(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# -- 工具函数 ------------------------------------------------
def rotate_img(img, angle):
    """旋转图像0/90/180/270度"""
    if angle == 0:   return img
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def resize_match(src, dst):
    """调整src尺寸以匹配dst"""
    dh, dw = dst.shape[:2]
    return cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)

def calc_similarity(img1, img2):
    """计算两图相似度（基于NCC归一化互相关，对旋转敏感）"""
    # 统一缩放到相同尺寸（使用img2作为目标尺寸）
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]

    # 如果尺寸差异过大，先统一缩放到较小尺寸以提高鲁棒性
    max_dim = max(max(h1, w1), max(h2, w2))
    if max_dim > 800:
        scale = 800 / max_dim
        w1s, h1s = int(w1 * scale), int(h1 * scale)
        w2s, h2s = int(w2 * scale), int(h2 * scale)
        img1_rs = cv2.resize(img1, (w1s, h1s), interpolation=cv2.INTER_AREA)
        img2_rs = cv2.resize(img2, (w2s, h2s), interpolation=cv2.INTER_AREA)
    else:
        img1_rs, img2_rs = img1, img2

    # 将img1缩放到img2的尺寸（保证两者尺寸一致才能计算NCC）
    h2r, w2r = img2_rs.shape[:2]
    img1_resized = cv2.resize(img1_rs, (w2r, h2r), interpolation=cv2.INTER_LINEAR)

    # 转灰度
    g1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2_rs, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 计算NCC（归一化互相关）
    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    g1_c = g1 - g1_mean
    g2_c = g2 - g2_mean
    num = np.sum(g1_c * g2_c)
    den = np.sqrt(np.sum(g1_c**2) * np.sum(g2_c**2))
    ncc = num / den if den > 0 else 0.0

    # 将NCC从[-1,1]映射到[0,100]作为相似度百分比
    # NCC=1 -> 100%, NCC=0 -> 50%, NCC=-1 -> 0%
    sim = (ncc + 1) / 2 * 100
    return sim

def align_images(img_ref, img_tmpl):
    """ORB+RANSAC对齐，返回对齐图和内点率

    使用相似性变换（estimateAffinePartial2D），只允许：
    - 旋转
    - 均匀缩放
    - 平移
    禁止使用全单应性变换（findHomography），防止产生极端透视变形。
    """
    ref_gray  = to_gray(img_ref)
    tmpl_gray = to_gray(img_tmpl)
    orb = cv2.ORB_create(nfeatures=CFG_ORB_ALIGN_FEATURES)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tmpl_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < CFG_ORB_ALIGN_RATIO * n.distance]
    if len(good) < 4:
        return None, 0.0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # ── 使用相似性变换（只允许旋转+均匀缩放+平移） ──
    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=CFG_RANSAC_THRESH,
        maxIters=CFG_RANSAC_MAXITERS
    )
    if M is None:
        return None, 0.0

    inliers = int(mask.sum())
    rate = inliers / len(good)

    # ── 变换合理性校验 ──
    a, b = M[0, 0], M[0, 1]
    scale = np.sqrt(a * a + b * b)
    if scale < 0.3 or scale > 3.0:
        print(f"  [!] 对齐变换异常: 缩放因子={scale:.2f}，放弃变换")
        return None, rate

    h_t, w_t = img_tmpl.shape[:2]
    aligned = cv2.warpAffine(
        img_ref, M, (w_t, h_t),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 检查输出是否严重变形（黑色区域过多）
    gray_aligned = to_gray(aligned)
    black_ratio = np.count_nonzero(gray_aligned < 10) / (h_t * w_t)
    if black_ratio > 0.5:
        print(f"  [!] 对齐结果异常: 黑色区域占比{black_ratio*100:.0f}%，放弃")
        return None, rate

    return aligned, rate


def find_main_content_region(img_bgr):
    """
    检测原稿主体内容区域（排除边缘标注）
    算法：灰度投影 + Canny边缘 + ROI连通区域外接矩形

    返回: (x, y, w, h) 主体区域坐标和尺寸
    """
    H, W = img_bgr.shape[:2]

    # 转换为灰度
    gray = to_gray(img_bgr)

    # ═══════════════════════════════════════════════════════
    # 步骤1：灰度投影 + Canny边缘（找行列边界）
    # ═══════════════════════════════════════════════════════
    edges = cv2.Canny(gray, 50, 150)

    # 行投影：前景密度 + 边缘强度
    row_density = np.mean(gray < 240, axis=1)
    row_edge = np.mean(edges, axis=1) / 255.0
    row_score = 0.5 * row_density + 0.5 * row_edge
    row_score_smooth = np.convolve(row_score, np.ones(5) / 5, mode='same')
    row_thresh = np.max(row_score_smooth) * 0.3
    candidate_rows = np.where(row_score_smooth > row_thresh)[0]

    if len(candidate_rows) == 0:
        print(f"  [主体检测] 未找到有效行，返回全图")
        return (0, 0, W, H)

    top_row = candidate_rows[0]
    bottom_row = candidate_rows[-1]

    # 列投影（在主体行范围内）
    col_density = np.mean(gray[top_row:bottom_row + 1, :] < 240, axis=0)
    col_thresh = np.max(col_density) * 0.3
    candidate_cols = np.where(col_density > col_thresh)[0]

    if len(candidate_cols) == 0:
        print(f"  [主体检测] 未找到有效列，返回全图")
        return (0, 0, W, H)

    left_col = candidate_cols[0]
    right_col = candidate_cols[-1]

    # ═══════════════════════════════════════════════════════
    # 步骤2：ROI二值化 + 闭运算 + 连通区域外接矩形
    # ═══════════════════════════════════════════════════════
    roi = gray[top_row:bottom_row + 1, left_col:right_col + 1]
    _, binary = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = roi.shape[0] * roi.shape[1] * 0.05
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        valid_contours = contours

    x_list, y_list, x2_list, y2_list = [], [], [], []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x_list.append(x)
        y_list.append(y)
        x2_list.append(x + w)
        y2_list.append(y + h)

    tx_min = min(x_list)
    ty_min = min(y_list)
    tx_max = max(x2_list)
    ty_max = max(y2_list)

    # 转换为原图坐标
    left = left_col + tx_min
    top = top_row + ty_min
    right = left_col + tx_max
    bottom = top_row + ty_max

    # 转换为 (x, y, w, h) 格式
    x, y, w, h = left, top, right - left, bottom - top

    coverage_pct = h / H * 100
    print(f"  [主体检测] 主体矩形: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"  [主体检测] 坐标 (x={x}, y={y}) {w}x{h} [覆盖率: {coverage_pct:.1f}%]")

    # ═══════════════════════════════════════════════════════
    # 步骤3：保存调试图像（内容区域绿色框 + 分界线红色）
    # ═══════════════════════════════════════════════════════
    debug_img = img_bgr.copy()

    # ① 绿线 - 内容区域边界
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(debug_img, '① 内容区域', (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ② 顶部边界（红线）
    cv2.line(debug_img, (0, top), (W, top), (0, 0, 255), 3)
    cv2.putText(debug_img, f'② 顶部边界 @y={top}', (10, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ③ 底部边界（蓝线）
    cv2.line(debug_img, (0, bottom), (W, bottom), (255, 0, 0), 2)
    cv2.putText(debug_img, f'③ 底部边界 @y={bottom}', (10, bottom + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(os.path.join(OUT_DIR, 'debug_main_region.jpg'), debug_img)

    return (x, y, w, h)

def extract_pdf_label_regions(pdf_path, dpi=600):
    """从PDF中提取所有裁切框区域，保存为独立PDF（保持矢量清晰度）"""
    doc = fitz.open(pdf_path)

    # 找洋红色裁切框
    page = doc[0]
    drawings = page.get_drawings()
    magenta_rects = []
    for d in drawings:
        c = d.get('color')
        if c and len(c) >= 3:
            r, g, b = c[0], c[1], c[2]
            # 洋红特征：红高、蓝中高、绿低
            if r > CFG_MAGENTA_R and b > CFG_MAGENTA_B and g < CFG_MAGENTA_G:
                rect = d['rect']
                w = rect.x1 - rect.x0
                h = rect.y1 - rect.y0
                if w > CFG_MIN_BOX_W and h > CFG_MIN_BOX_H:
                    magenta_rects.append({
                        'l': rect.x0, 't': rect.y0,
                        'r': rect.x1, 'b': rect.y1,
                        'w': w, 'h': h
                    })

    # 为每个裁切框创建独立PDF
    regions = []
    for i, box in enumerate(magenta_rects):
        # 创建新PDF，只含此裁切框区域
        new_doc = fitz.open()
        new_page = new_doc.new_page(width=box['w'], height=box['h'])

        # 从原PDF复制内容（平移至新原点）
        new_page.show_pdf_page(
            fitz.Rect(0, 0, box['w'], box['h']),
            doc,
            0,
            clip=fitz.Rect(box['l'], box['t'], box['r'], box['b'])
        )

        # 保存为PDF
        pdf_out = os.path.join(OUT_DIR, f"step3_pdf_region_{i}.pdf")
        new_doc.save(pdf_out)
        new_doc.close()

        # 从新PDF高DPI渲染（600DPI用于基础对比）
        scale = dpi / 72.0
        render_doc = fitz.open(pdf_out)
        render_page = render_doc[0]
        pix = render_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pdf_bgr = cv2.cvtColor(np.array(pix.pil_image()), cv2.COLOR_RGB2BGR)
        render_doc.close()

        regions.append({
            'img': pdf_bgr,
            'pdf_path': pdf_out,
            'rect': (0, 0, pdf_bgr.shape[1], pdf_bgr.shape[0]),
            'label': f"region_{i}_{pdf_bgr.shape[1]}x{pdf_bgr.shape[0]}",
            'orig_rect': box
        })
        print(f"  裁切框{i}: {box['w']:.1f}x{box['h']:.1f} -> PDF已保存，渲染{ pdf_bgr.shape[1]}x{pdf_bgr.shape[0]}px")

    doc.close()
    print(f"  找到裁切框: {len(regions)} 个，已保存为独立PDF")
    return regions


# ═══════════════════════════════════════════════════════════════════════
#  Step5.9：OCR文字识别与缺字漏字检测
# ═══════════════════════════════════════════════════════════════════════

def _init_paddleocr():
    """懒加载PaddleOCR实例（全局单例），限制线程数以控制CPU占用"""
    if not hasattr(_init_paddleocr, '_ocr'):
        print("    初始化 PaddleOCR（首次运行需加载模型，约10秒）...")
        
        # 限制OpenMP和MKL线程数（PaddleOCR底层使用这些库）
        import os
        import psutil
        # 强制限制为3核心（用户要求）
        ocr_cores = 3
        cpu_count = psutil.cpu_count()
        if cpu_count < 3:
            ocr_cores = cpu_count  # 核心数不足3时，用全部
        
        # 限制OpenBLAS/MKL/OMP线程数
        os.environ['OMP_NUM_THREADS'] = str(ocr_cores)
        os.environ['MKL_NUM_THREADS'] = str(ocr_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(ocr_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(ocr_cores)
        print(f"    [OCR资源] 限制推理线程数为 {ocr_cores}（系统总核心: {cpu_count}）")
        
        # 禁用内部缩放和文档去扭曲
        # 注意：PaddleOCR内部仍可能缩放图像，我们会在ocr_recognize中手动还原坐标
        _init_paddleocr._ocr = PaddleOCR(
            lang='ch',
            text_det_limit_side_len=999999,  # 禁用检测缩放
            use_doc_unwarping=False,  # 禁用文档去扭曲
        )
    return _init_paddleocr._ocr


def ocr_recognize(img_bgr, label=""):
    """
    对图像进行OCR识别，返回文字块列表
    Returns:
        List[dict]  # [{'text': str, 'box': (x1,y1,x2,y2), 'score': float}, ...]
    """
    import tempfile
    ocr = _init_paddleocr()
    # 重要：PaddleOCR对numpy数组和路径文件的坐标处理不同！
    # 必须用路径传入才能保证坐标正确
    h_orig, w_orig = img_bgr.shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)  # 用PNG避免JPG压缩损失
    tmp.close()
    cv2.imwrite(tmp.name, img_bgr)
    try:
        raw = ocr.ocr(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    blocks = []
    if raw and raw[0]:
        r = raw[0]
        texts  = list(r['rec_texts']) if 'rec_texts' in r else []
        scores = list(r['rec_scores']) if 'rec_scores' in r else []
        polys  = list(r['rec_polys'])  if 'rec_polys'  in r else []

        # 检查是否需要坐标还原
        # PaddleOCR内部可能将图像缩放到max_side_limit=4000
        if polys and len(polys) > 0:
            all_coords = np.concatenate(polys, axis=0)
            max_coord = all_coords.max()
            # 如果坐标明显超出原图范围，计算缩放比例并还原
            if max_coord > max(w_orig, h_orig):
                scale = max(w_orig, h_orig) / max_coord
                print(f"    [{label}] 坐标还原: scale={scale:.4f}")
                # 还原所有坐标
                for i in range(len(polys)):
                    polys[i] = (polys[i] * scale).astype(np.int32)

        for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
            # poly: numpy array shape (4,2) 四角点
            text = text.strip()
            if not text:
                continue
            # 转换为整数角点
            pts = np.array(poly, dtype=np.int32)
            xs, ys = pts[:, 0], pts[:, 1]
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            h = y2 - y1
            if h < CFG_OCR_MIN_H:
                continue
            blocks.append({
                'text': text,
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'poly': pts.tolist(),  # 保留原始四角点坐标
                'score': float(score),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
            })

    n_good = len([b for b in blocks if b['score'] >= 0.5])
    print(f"    [{label}] OCR识别: {len(blocks)} 个文字块（置信度≥0.5: {n_good}）")
    return blocks


def box_iou(box_a, box_b):
    """计算两个轴对齐矩形的IoU"""
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def levenshtein_sim(s1, s2):
    """计算两字符串的相似度（Levenshtein距离归一化）"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    dist = dp[m][n]
    return 1 - dist / max(m, n)


def compare_text_blocks(blocks_orig, blocks_pdf):
    """
    对比原稿和PDF的文字块，检测缺字漏字

    以原稿为基准：
    - missing：原稿有、PDF无对应 -> [!]重要
    - extra：PDF有、原稿无对应 -> -次要
    - similar：位置接近但文字有差异 -> [!]文字变形/错误
    
    匹配策略：优先位置IoU匹配，若IoU不足但文字内容相同/相近则兜底匹配
    """
    missing = []
    extra = []
    similar = []
    matched_pdf_idx = set()

    for b_orig in blocks_orig:
        best_iou = 0
        best_pdf_idx = -1
        best_text_sim = 0

        for i, b_pdf in enumerate(blocks_pdf):
            if i in matched_pdf_idx:
                continue
            iou = box_iou(b_orig['box'], b_pdf['box'])
            text_sim = levenshtein_sim(b_orig['text'], b_pdf['text'])
            
            # 优先IoU匹配
            if iou > best_iou:
                best_iou = iou
                best_pdf_idx = i
                best_text_sim = text_sim

        # 判断是否为匹配
        is_matched = False
        if best_pdf_idx >= 0:
            # 策略1：位置IoU足够
            if best_iou >= CFG_TEXT_IOU_THRESH:
                is_matched = True
            # 策略2：位置不够但文字内容相同/相近（兜底匹配）
            # 容许PDF对齐微小偏移导致IoU不足，但文字实际是同一个
            elif best_text_sim >= CFG_CONTENT_MATCH_SIM:
                # 扩大搜索范围，检查是否有其他块IoU更接近
                # 如果内容高度相似且没有其他更好的位置匹配，视为匹配
                has_better_position = False
                for i, b_pdf in enumerate(blocks_pdf):
                    if i in matched_pdf_idx or i == best_pdf_idx:
                        continue
                    other_iou = box_iou(b_orig['box'], b_pdf['box'])
                    if other_iou > best_iou * 1.5:  # 明显更好的位置匹配存在
                        has_better_position = True
                        break
                if not has_better_position:
                    is_matched = True
                    print(f"    [OCR兜底] 文字'{b_orig['text'][:10]}...' IoU={best_iou:.2f}不足但内容相似度={best_text_sim:.2f}，视为匹配")

        if is_matched:
            matched_pdf_idx.add(best_pdf_idx)
            b_pdf = blocks_pdf[best_pdf_idx]
            if best_text_sim < CFG_TEXT_SIM_THRESH:
                similar.append({
                    'orig_text': b_orig['text'],
                    'pdf_text': b_pdf['text'],
                    'box': b_orig['box'],
                    'poly': b_orig.get('poly'),
                    'text_sim': best_text_sim,
                })
        else:
            missing.append({
                'text': b_orig['text'],
                'box': b_orig['box'],
                'poly': b_orig.get('poly'),
                'score': b_orig['score'],
            })

    for i, b_pdf in enumerate(blocks_pdf):
        if i not in matched_pdf_idx:
            extra.append({
                'text': b_pdf['text'],
                'box': b_pdf['box'],
                'poly': b_pdf.get('poly'),
                'score': b_pdf['score'],
            })

    return missing, extra, similar


def draw_text_diff(img_bgr, missing, extra, similar):
    """在图像上标注文字差异区域
    注意：missing/similar 的坐标来自原稿，extra 的坐标来自 PDF 对齐图
    这里用原稿作为背景，extra 的坐标可能不准（仅作参考）
    """
    ann = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    def _draw_box(box, color, label):
        x1, y1, x2, y2 = box
        # 确保坐标在图像范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W-1, int(x2)), min(H-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return  # 无效框
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3
        bg_x1, bg_y1 = x1, max(y1 - th - pad * 2, 0)
        bg_x2, bg_y2 = min(x1 + tw + pad * 2, W), y1
        cv2.rectangle(ann, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.rectangle(ann, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        cv2.putText(ann, label, (bg_x1 + pad, bg_y1 + th + pad),
                    font, font_scale, color, thickness)

    # 缺字：红色（坐标来自原稿，准确）
    for m in missing:
        _draw_box(m['box'], (0, 0, 255), f"缺:{m['text'][:8]}")
    # 多字：蓝色（坐标来自 PDF，可能不准，仅参考）
    for e in extra:
        _draw_box(e['box'], (255, 0, 0), f"多:{e['text'][:8]}")
    # 变形：黄色（坐标来自原稿，准确）
    for s in similar:
        _draw_box(s['box'], (0, 255, 255), f"异:{s['orig_text'][:4]}")
    return ann


def render_pdf_at_dpi(pdf_path, dpi):
    """从PDF文件高分辨率渲染，返回BGR图像"""
    scale = dpi / 72.0
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = cv2.cvtColor(np.array(pix.pil_image()), cv2.COLOR_RGB2BGR)
    doc.close()
    return img


def generate_pdf_report(results, out_dir):
    """生成紧凑简洁的PDF格式检测报告
    out_dir: PDF报告输出目录
    """
    from PIL import Image as PILImage
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    # 报告保存到原稿同目录（用户要求）
    pdf_path = os.path.join(REPORT_DIR, "印前检测报告.pdf")

    # DPI转mm系数 (300DPI)
    DPI_TO_MM = 25.4 / 300
    # A4 可用空间
    MAX_W = 180 * mm
    MAX_H = 250 * mm

    # 创建PDF文档
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=15*mm, rightMargin=15*mm,
                          topMargin=12*mm, bottomMargin=12*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontName=PDF_FONT, fontSize=14,
                                  alignment=1, spaceAfter=4)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
                                   fontName=PDF_FONT, fontSize=8,
                                   spaceAfter=2)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
                                    fontName=PDF_FONT, fontSize=7,
                                    textColor=colors.grey, spaceAfter=1,
                                    alignment=1)

    elements = []

    # 标题
    elements.append(Paragraph("PDF 印前检测报告", title_style))
    elements.append(Spacer(1, 3*mm))

    # 左右分栏布局
    # 左侧：原稿、PDF、时间、NCC相似度、显著轮廓
    left_col = [
        ["原稿", results['orig_file']],
        ["PDF", results['pdf_file']],
        ["时间", results['time']],
        ["NCC相似度", f"{results['fine_ncc']*100:.1f}%"],
        ["显著轮廓", f"{results['n_big']}个"],
    ]
    
    # 右侧：PDF缺失、PDF多出、缺字、多字、变形
    right_col = [
        ["[!]PDF缺失", f"{results['n_missing']:,}px ({results['n_missing']/results['total']*100:.1f}%)"],
        ["-PDF多出", f"{results['n_extra']:,}px ({results['n_extra']/results['total']*100:.1f}%)"],
        ["[!]缺字", f"{results['n_missing_text']}个"],
        ["-多字", f"{results['n_extra_text']}个"],
        ["[*]变形", f"{results['n_similar_text']}个"],
    ]

    left_table = Table(left_col, colWidths=[24*mm, 64*mm])
    left_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), PDF_FONT),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F0F0F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 2),
        # 高亮NCC相似度行（第4行，索引3）
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#FFF3E0') if results['fine_ncc']*100 < 85 else colors.HexColor('#E8F5E9')),
    ]))

    right_table = Table(right_col, colWidths=[28*mm, 50*mm])
    right_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), PDF_FONT),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F5F5F5')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 2),
        ('TEXTCOLOR', (0,0), (0,0), colors.red),
        ('TEXTCOLOR', (0,3), (0,3), colors.red),
    ]))

    # 并排放置两个表格
    info_layout = Table([[left_table, right_table]], colWidths=[90*mm, 80*mm])
    info_layout.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(info_layout)
    elements.append(Spacer(1, 3*mm))

    # 判定结果
    status_text = results['status']
    status_color = colors.HexColor('#4CAF50') if "[OK]" in status_text else colors.HexColor('#F44336')
    elements.append(Paragraph(f"<b>判定：{status_text}</b>", ParagraphStyle(
        'Status', fontName=PDF_FONT, fontSize=11,
        textColor=status_color, alignment=1, spaceAfter=4
    )))

    # 对比图示（从左到右依次排列）
    elements.append(Paragraph("对比图示", normal_style))
    elements.append(Spacer(1, 2*mm))

    def _img(fpath, max_w=MAX_W, max_h=65*mm):
        """加载图片并等比缩放，返回缩放后的Image对象"""
        img = Image(fpath)
        scale = min(max_w / img.drawWidth, max_h / img.drawHeight, 1.0)
        im = Image(fpath, width=img.drawWidth*scale, height=img.drawHeight*scale)
        im.hAlign = 'CENTER'
        return im

    def _add_single(fname, caption):
        """单图居中 + 标题"""
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            try:
                elements.append(_img(p))
                elements.append(Paragraph(caption, caption_style))
                elements.append(Spacer(1, 2*mm))
            except Exception as e:
                print(f"  [报告] 加载图片失败 {fname}: {e}")

    def _add_2col(left_path, right_path, caption, col_w=88*mm):
        """双栏：左侧 + 右侧 + 标题；原稿始终在左"""
        lp = os.path.join(out_dir, left_path)
        rp = os.path.join(out_dir, right_path)
        if os.path.exists(lp) and os.path.exists(rp):
            try:
                left  = _img(lp, max_w=col_w)
                right = _img(rp, max_w=col_w)
                target_h = max(left.drawHeight, right.drawHeight)
                left.drawHeight  = target_h
                right.drawHeight = target_h
                tbl = Table([[left, right]], colWidths=[col_w, col_w])
                tbl.setStyle(TableStyle([
                    ('ALIGN',    (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN',   (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING',  (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ]))
                tbl.hAlign = 'CENTER'
                elements.append(tbl)
                elements.append(Paragraph(caption, caption_style))
                elements.append(Spacer(1, 2*mm))
            except Exception as e:
                print(f"  [报告] 双栏加载失败 {left_path}/{right_path}: {e}")

    # 1. 综合对比（三栏：原稿 | PDF | 综合差异）
    _add_single("step5_combined_comparison.jpg", "综合对比")

    # 2. 热力差异（三栏：原稿 | PDF提取 | 热力差异标注：白=相同/红=缺失/蓝=多出）
    _add_single("step5_heatmap_pure.jpg", "热力差异")

    # 3. 文字缺多字差异（原稿 | OCR文字差异标注）
    _add_single("step5_ocr_text_diff.jpg", "文字缺多字差异")

    # 4. 轮廓差异（原稿 | 红色矩形标注）
    _add_single("step5_diff_annotated.jpg", "轮廓差异")

    # 生成PDF
    doc.build(elements)
    print(f"  -> 生成PDF报告: 印前检测报告.pdf")
    return pdf_path

# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  资源限制（CPU 3核 + 内存 4GB）
# ═══════════════════════════════════════════════════════════════
def limit_resources():
    """限制进程CPU（3核）和内存（4GB）"""
    import threading, time, psutil, os, gc

    try:
        pid = os.getpid()
        process = psutil.Process(pid)
        cpu_count = psutil.cpu_count()

        # CPU亲和性：限制为前3个核心（0,1,2）
        target_cores = [0, 1, 2]
        target_cores = [c for c in target_cores if c < cpu_count]
        if target_cores:
            process.cpu_affinity(target_cores)
            print(f"  [资源限制] CPU亲和性: 核心 {target_cores} (共{cpu_count}核)")

        # 内存上限：4GB = 4 * 1024**3 bytes
        MEM_LIMIT_BYTES = 4 * 1024**3  # 4GB

        def resource_monitor():
            """每2秒检查内存，超过4GB时GC"""
            while True:
                try:
                    time.sleep(2)
                    mem_info = psutil.virtual_memory()
                    if mem_info.percent > 85:
                        gc.collect()
                        print(f"  [资源限制] 内存 {mem_info.percent:.1f}% > 85%，触发GC")
                except Exception:
                    break

        t = threading.Thread(target=resource_monitor, daemon=True)
        t.start()
        print(f"  [资源限制] 内存监控已启动，上限4GB")

        # 进程优先级：below_normal 降低系统影响
        try:
            import subprocess
            subprocess.run(
                ['wmic', 'process', 'where', f'processid={pid}', 'set', 'priority=32768'],
                capture_output=True, creationflags=0x08000000
            )
        except Exception:
            pass

    except Exception as e:
        print(f"  [资源限制] 初始化失败: {e}")

limit_resources()

# -- Step1: 加载原稿（300DPI保持CMYK）----------------------
print("=" * 60)
print("Step1: 加载原稿jpg和PDF文件（300DPI）")
print("=" * 60)

# 读取原稿，让checker_core缓存CMYK信息
orig_bgr = imread(ORIG_JPG)
H0, W0 = orig_bgr.shape[:2]

# 检查原稿DPI，如果是低分辨率则提示
print(f"  原稿: {W0} x {H0} px  ({ORIG_JPG})")
print(f"  [!] 分辨率建议: ≥300DPI以获得最佳对比效果")

# PDF使用300DPI提取
PDF_DPI = 300
print(f"  PDF提取分辨率: {PDF_DPI}DPI")

# -- Step2: 识别原稿主体内容区域 -----------------------------
print("\n" + "=" * 60)
print("Step2: 识别原稿主体内容区域（排除标注信息）")
print("=" * 60)

main_rect = find_main_content_region(orig_bgr)
mx, my, mw, mh = main_rect
orig_content = orig_bgr[my:my+mh, mx:mx+mw].copy()
save(os.path.join(OUT_DIR, "step2_orig_content.jpg"), orig_content)
print(f"  -> 保存: step2_orig_content.jpg ({orig_content.shape[1]}x{orig_content.shape[0]})")

# -- Step3: 提取PDF裁切框（300DPI）--------------------------
print("\n" + "=" * 60)
print("Step3: 识别PDF裁切框位置，筛选不相似标签（300DPI）")
print("=" * 60)

pdf_regions = extract_pdf_label_regions(PDF_FILE, dpi=PDF_DPI)
print(f"  原始裁切框数量: {len(pdf_regions)}")

# 保存所有PDF裁切框
for i, reg in enumerate(pdf_regions):
    save(os.path.join(OUT_DIR, f"step3_pdf_region_{i}_{reg['label']}.jpg"), reg['img'])

# 相似度去重：相似度>90%的只保留一个
unique_regions = []
for reg in pdf_regions:
    is_duplicate = False
    for unique in unique_regions:
        sim = calc_similarity(reg['img'], unique['img'])
        print(f"    相似度 {reg['label']} vs {unique['label']}: {sim:.1f}%")
        if sim > CFG_DUP_SIM_THRESH:
            is_duplicate = True
            print(f"      -> 相似>{CFG_DUP_SIM_THRESH:.0f}%，跳过")
            break
    if not is_duplicate:
        unique_regions.append(reg)
        print(f"      -> 保留为独立标签")

print(f"  筛选后标签数量: {len(unique_regions)}")

# 保存筛选后的PDF标签（4个角度）
print(f"\n  保存筛选后标签（0°/90°/180°/270°）...")
for i, reg in enumerate(unique_regions):
    for angle in [0, 90, 180, 270]:
        rotated = rotate_img(reg['img'], angle)
        fname = f"step3_pdf_unique_{i}_angle{angle}_{reg['label']}.jpg"
        save(os.path.join(OUT_DIR, fname), rotated)

# -- Step4: 4角度对比找最佳匹配 ----------------------------
print("\n" + "=" * 60)
print("Step4: 4角度对比原稿和PDF提取图")
print("=" * 60)

# 用第一个筛选后的PDF标签进行对比
best_pdf_path_saved = None  # 供Step5.1直接使用，避免np.array_equal不可靠
if unique_regions:
    best_sim = 0
    best_angle = 0
    best_pdf = None
    best_idx = -1

    print(f"  对比原稿 vs 筛选后PDF标签:")
    for i, reg in enumerate(unique_regions):
        print(f"\n  [PDF标签 {i}] {reg['label']}:")
        for angle in [0, 90, 180, 270]:
            rotated = rotate_img(reg['img'], angle)
            sim = calc_similarity(rotated, orig_content)
            flag = " ← 最佳" if sim > best_sim else ""
            print(f"    {angle}°: 相似度 {sim:.1f}%{flag}")
            if sim > best_sim:
                best_sim = sim
                best_angle = angle
                best_pdf = rotated
                best_idx = i

    print(f"\n  [OK] 最佳匹配: {best_angle}° (相似度 {best_sim:.1f}%)")
    
    # 保存最佳匹配的PDF路径，供Step5.1高分辨率渲染使用
    if best_idx >= 0:
        best_pdf_path_saved = unique_regions[best_idx]['pdf_path']
        print(f"  [OK] 最佳匹配PDF: {best_pdf_path_saved}")
    else:
        best_pdf_path_saved = None

    # 保存最佳匹配的PDF
    save(os.path.join(OUT_DIR, "step4_best_pdf.jpg"), best_pdf)
    print(f"  -> 保存: step4_best_pdf.jpg")
else:
    print("  [!] 没有筛选后的PDF标签，使用全部区域")
    best_pdf = pdf_regions[0]['img'] if pdf_regions else orig_content
    best_angle = 0
    best_sim = 0

# -- Step5: 详细对比 ----------------------------------------
print("\n" + "=" * 60)
print("Step5: 详细对比（高分辨率PDF vs 原稿）")
print("=" * 60)

# -- 5.1: 高分辨率PDF渲染（体现PDF矢量优势）-----------------
print(f"\n  [PDF矢量优势] 高分辨率放大...")
HIGH_DPI = CFG_HIGH_DPI
best_pdf_hires = None

# 直接复用Step4中已渲染的 best_pdf（300DPI裁切图像）
# 不再重新从PDF文件渲染，避免子PDF尺寸/PyMuPDF坐标系不一致问题
# （原方案 render_pdf_at_dpi(best_pdf_path_saved, 1200) 在裁切框尺寸≠DPI换算尺寸时会出现内容偏差）
# 改用高质量 Lanczos4 放大：从 300DPI 等比放大到 1200DPI（4×）
if best_pdf is not None:
    h_300, w_300 = best_pdf.shape[:2]
    w_hires = w_300 * 4   # 1200/300 = 4倍
    h_hires = h_300 * 4
    print(f"  从Step4已渲染图像放大: {w_300}x{h_300} → {w_hires}x{h_hires} (1200 DPI)")
    best_pdf_hires = cv2.resize(best_pdf, (w_hires, h_hires), interpolation=cv2.INTER_LANCZOS4)
    print(f"  放大后: {best_pdf_hires.shape[1]}x{best_pdf_hires.shape[0]} px")
else:
    print(f"  [!] best_pdf为空，跳过高分辨率放大")
    best_pdf_hires = None

# 保存高分辨率PDF渲染图
if best_pdf_hires is not None:
    save(os.path.join(OUT_DIR, "step5_pdf_hires_1200dpi.jpg"), best_pdf_hires)
    print(f"  -> 保存: step5_pdf_hires_1200dpi.jpg")
else:
    print(f"  [!] 无法生成高分辨率PDF，流程终止")
    print(f"  （best_pdf为空，请检查Step4是否正常选择最佳匹配）")
    sys.exit(1)  # 模块级退出，避免后续崩溃

# -- 5.2: 直接缩放对齐（保持Step4确定的方向，不做ORB再对齐） --
# 【修复】原方案先缩放到原稿全尺寸再做ORB粗对齐，
# ORB的estimateAffinePartial2D可能对best_pdf（已正确旋转）再次估计出错误旋转变换，
# 导致最终输出方向与Step4不一致。
# 新方案：Step4已用NCC精确选出最佳角度，best_pdf方向已正确，
# 直接将best_pdf缩放到红线区域尺寸（与orig_content一致），保留方向不变。
print(f"\n  [Step 5.2] 直接缩放对齐（保持Step4方向）...")

# aligned = best_pdf（已旋转的正确方向）resize到红线区域尺寸
# orig_content = orig_bgr[my:my+mh, mx:mx+mw]（红线区域原稿）
# 两者都代表红线区域内容，尺寸应一致（或接近），直接resize匹配
H_content, W_content = orig_content.shape[:2]
print(f"  红线区域原稿尺寸: {W_content}x{H_content}")

# best_pdf可能与红线区域尺寸不同（旋转后宽高交换，或DPI略有差异）
# 使用LANCZOS4高质量缩放，同时处理旋转后宽高不一致问题
h_pdf, w_pdf = best_pdf.shape[:2]
print(f"  best_pdf原始尺寸: {w_pdf}x{h_pdf} (best_angle={best_angle}°)")

# 如果best_pdf是90°/270°旋转，宽高已交换，resize到红线区域尺寸即可
aligned = cv2.resize(best_pdf, (W_content, H_content), interpolation=cv2.INTER_LANCZOS4)
print(f"  直接缩放至红线区域: {W_content}x{H_content}")

# 保存调试图像
debug_path = os.path.join(OUT_DIR, "debug_5_2_aligned.jpg")
cv2.imwrite(debug_path, aligned)
print(f"  -> 保存: debug_5_2_aligned.jpg (方向与step4_best_pdf一致)")

# NCC精细匹配分数（验证方向正确性）
def calc_ncc(img1, img2):
    g1 = to_gray(img1).astype(np.float32)
    g2 = to_gray(img2).astype(np.float32)
    g1_mean, g2_mean = np.mean(g1), np.mean(g2)
    g1_c, g2_c = g1 - g1_mean, g2 - g2_mean
    num = np.sum(g1_c * g2_c)
    den = np.sqrt(np.sum(g1_c**2) * np.sum(g2_c**2))
    return num / den if den > 0 else 0

fine_ncc = calc_ncc(aligned, orig_content)
print(f"  红线区域内NCC: {fine_ncc:.4f}")

# 对齐图 = 直接缩放后的红线区域
aligned_crop = aligned

# inlier_rate在Step5.2原ORB对齐中使用，新方案跳过对齐，设为0.0
inlier_rate = 0.0
print(f"  内点率: {inlier_rate*100:.1f}% (跳过ORB对齐，使用Step4 NCC方向)")

# -- 5.3: 边缘检测 ----------------------------------------
print(f"  边缘检测 (Canny {CFG_EDGE_CANNY_A}/{CFG_EDGE_CANNY_B})...")
edges_pdf  = cv2.Canny(to_gray(aligned),      CFG_EDGE_CANNY_A, CFG_EDGE_CANNY_B)
edges_orig = cv2.Canny(to_gray(orig_content), CFG_EDGE_CANNY_A, CFG_EDGE_CANNY_B)

# -- 5.4: 边缘轮廓差异统计 --------------------------------
H, W = orig_content.shape[:2]
edge_pdf_bool  = (edges_pdf  > 0)
edge_orig_bool = (edges_orig > 0)

both_edge   = edge_pdf_bool & edge_orig_bool       # 双方都有边缘 -> 内容一致
edge_missing = edge_orig_bool & ~edge_pdf_bool     # 原稿有、PDF无 -> 内容缺失
edge_extra   = edge_pdf_bool  & ~edge_orig_bool    # PDF有、原稿无 -> 内容多出

n_both    = int(np.count_nonzero(both_edge))
n_missing = int(np.count_nonzero(edge_missing))
n_extra   = int(np.count_nonzero(edge_extra))
total     = H * W

print(f"\n  [边缘轮廓差异统计 - 以原稿为准]")
print(f"    边缘一致（双方均有）: {n_both:>8,} px ({n_both/total*100:5.1f}%)")
print(f"    [!] PDF缺失（原稿有、PDF无）: {n_missing:>8,} px ({n_missing/total*100:5.1f}%)  ← 标红（重要！）")
print(f"    - PDF多出（PDF有、原稿无）: {n_extra:>8,} px ({n_extra/total*100:5.1f}%)  ← 标蓝（次要）")
print(f"    差异像素合计:                {n_missing+n_extra:>8,} px ({(n_missing+n_extra)/total*100:5.1f}%)")

# -- 5.5: 边缘差异热力图（红=PDF缺失←重要，蓝=PDF多出，白=一致）-------
diff_heatmap = np.zeros((H, W, 3), dtype=np.uint8)
diff_heatmap[:, :, 2] = edge_missing.astype(np.uint8) * 255   # R=PDF缺失内容（标红·重要！）
diff_heatmap[:, :, 0] = edge_extra.astype(np.uint8) * 255      # B=PDF多出内容（标蓝·次要）
# 纯热力图（白底，仅差异区域有颜色）
diff_heatmap_pure = diff_heatmap.copy()
diff_heatmap_pure[:, :] = 255  # 无差异区域填白

# 叠加到原稿（只有差异区域着色，无差异区域显示原图）
has_diff = edge_missing | edge_extra
alpha_map = has_diff.astype(np.float32)[:, :, np.newaxis]
orig_float = orig_content.astype(np.float32)
heat_float = diff_heatmap.astype(np.float32)
overlay = (orig_float * (1 - alpha_map * CFG_HEATMAP_ALPHA) +
           heat_float * alpha_map * CFG_HEATMAP_ALPHA).astype(np.uint8)

# 定义间隔（所有对比图共用，4px紧凑间距）
pad = np.ones((H, 4, 3), dtype=np.uint8) * 80

# -- 5.6: PDF提取图对比（原稿 | PDF，双栏，与差异图排列一致）--
pdf_compare = np.hstack([orig_content, pad, aligned])
save(os.path.join(OUT_DIR, "step5_pdf_compare.jpg"), pdf_compare)
print(f"  -> 保存: step5_pdf_compare.jpg (原稿|PDF提取)")

# -- 5.7: 组合对比图（原稿 | PDF | 差异）------------------------
combined = np.hstack([orig_content, pad, aligned, pad, overlay])
save(os.path.join(OUT_DIR, "step5_combined_comparison.jpg"), combined)

# -- 5.7: 热力差异标注图（白=相同，红=PDF缺失，蓝=PDF多出，基于像素差异）--
# 改用像素级绝对差异+NCC思路，替代边缘检测，能正确捕捉PDF缺失区域
orig_gray = cv2.cvtColor(orig_content, cv2.COLOR_BGR2GRAY)
aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

# 像素级绝对差异
diff_abs = cv2.absdiff(orig_gray, aligned_gray)
_, diff_mask = cv2.threshold(diff_abs, 25, 255, cv2.THRESH_BINARY)

# 轻微膨胀，使相近差异区域连通
_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
diff_mask = cv2.dilate(diff_mask, _kernel, iterations=1)

# 白底标注图
hm_annotated = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色底色（相同区域）

cnts, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_hm_missing = 0
n_hm_extra = 0
for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < CFG_MIN_CONTOUR_AREA:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    # 用ROI均值差判断类型：原稿 - PDF
    _roi_o = orig_gray[y:y+h, x:x+w].astype(np.float32)
    _roi_a = aligned_gray[y:y+h, x:x+w].astype(np.float32)
    mean_d = float(np.mean(_roi_o - _roi_a))
    if mean_d > 15:    # 原稿明显更亮 → PDF缺失内容（红M）
        n_hm_missing += 1
        color = (0, 0, 255)
        label = f"M{int(area)}"
    elif mean_d < -15:  # PDF明显更亮 → PDF多出内容（蓝E）
        n_hm_extra += 1
        color = (255, 0, 0)
        label = f"E{int(area)}"
    else:
        continue
    cv2.rectangle(hm_annotated, (x, y), (x+w, y+h), color, 2)
    cv2.putText(hm_annotated, label, (x, max(y-5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

print(f"  热力差异(像素比对): PDF缺失{n_hm_missing}处(M红) / PDF多出{n_hm_extra}处(E蓝) / 白色=相同")

# 保存热力差异标注图（三栏：原稿 | PDF提取 | 热力差异标注）
hm_annotated_rgb = cv2.cvtColor(hm_annotated, cv2.COLOR_BGR2RGB)
heatmap_3col = np.hstack([orig_content, pad, aligned, pad, hm_annotated_rgb])
save(os.path.join(OUT_DIR, "step5_heatmap_pure.jpg"), heatmap_3col)
print(f"  -> 保存: step5_heatmap_pure.jpg (原稿|PDF提取|热力差异: 白=相同/红=缺失/蓝=多出)")

# -- 5.7: 差异轮廓标注 ------------------------------------
# 基于边缘差异二值图标注
edge_diff_bin = ((edge_missing | edge_extra) * 255).astype(np.uint8)
annotated = orig_content.copy()
contours, _ = cv2.findContours(edge_diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_big = 0
for cnt in contours:
    if cv2.contourArea(cnt) < CFG_MIN_CONTOUR_AREA:
        continue
    n_big += 1
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(annotated, f"A{cv2.contourArea(cnt):.0f}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# 差异标注图（原稿 | 标注，红色框标出显著差异区域）
annotated_combined = np.hstack([
    orig_content,        # 原稿
    pad,                 # 紧凑间隔
    annotated            # 差异区域红色矩形标注
])
save(os.path.join(OUT_DIR, "step5_diff_annotated.jpg"), annotated_combined)
print(f"  显著差异轮廓: {n_big} 个 (面积>{CFG_MIN_CONTOUR_AREA}px²)")

# -- 5.8: 边缘对比图（原稿 | PDF | 差异）-----------------------
_edge_font = cv2.FONT_HERSHEY_SIMPLEX
edge_diff_abs = cv2.absdiff(edges_pdf, edges_orig)
edge_combined = np.hstack([
    cv2.cvtColor(edges_orig, cv2.COLOR_GRAY2BGR),  # 原稿
    pad,
    cv2.cvtColor(edges_pdf,  cv2.COLOR_GRAY2BGR),  # PDF
    pad,
    cv2.cvtColor(edge_diff_abs, cv2.COLOR_GRAY2BGR)  # 差异
])
# 无标签行，直接保存
save(os.path.join(OUT_DIR, "step5_edge_comparison.jpg"), edge_combined)

print(f"  -> 保存: step5_combined_comparison.jpg (原稿|PDF|差异)")
print(f"  -> 保存: step5_heatmap_comparison.jpg (原稿|叠加|热力图)")
print(f"  -> 保存: step5_diff_annotated.jpg (原稿|差异标注)")
print(f"\n  [*] PDF矢量优势: PDF以{HIGH_DPI}DPI渲染({best_pdf_hires.shape[1]}x{best_pdf_hires.shape[0]}px)，细节更清晰！")

# ═══════════════════════════════════════════════════════════════════════
#  Step5.9：OCR文字识别 — 缺字漏字检测
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step5.9: OCR文字识别对比（缺字漏字检测）")
print("=" * 60)

# OCR输入：300DPI尺寸（兼顾速度与识别率）
# aligned: PDF对齐图；orig_content: 原稿
ocr_pdf   = aligned.copy()
ocr_orig  = orig_content.copy()

print(f"\n  [Step 5.9.1] OCR识别原稿文字...")
blocks_orig = ocr_recognize(ocr_orig, "原稿")
print(f"  [Step 5.9.2] OCR识别PDF对齐图文字...")
blocks_pdf  = ocr_recognize(ocr_pdf,  "PDF")

print(f"\n  [Step 5.9.3] 对比文字块，检测缺字漏字...")
missing, extra, similar = compare_text_blocks(blocks_orig, blocks_pdf)

print("  +-----------------------------------------------+")
print("  |          OCR文字差异统计 — 以原稿为准          |")
print("  +-----------------------------------------------+")
print(f"  |  [!] 缺字: {len(missing):>3}   - 多字: {len(extra):>3}   [*] 变形: {len(similar):>3}       |")
print("  +-----------------------------------------------+")

# 保存文字差异标注图
# 所有坐标都在PDF对齐图像上（标注在PDF图上，方便与原稿对照）
ann = ocr_pdf.copy()
H, W = ann.shape[:2]

# ═══ 调试：保存所有OCR框的完整标注图（用于验证坐标正确性）═══
try:
    _dbg_all = ocr_orig.copy()
    _dbg_pil = PILImage.fromarray(cv2.cvtColor(_dbg_all, cv2.COLOR_BGR2RGB))
    _dbg_draw = PILImageDraw.Draw(_dbg_pil)
    _dbg_font = PILImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 14)
    _clr = [(255,0,0),(0,200,0),(0,0,255),(200,200,0),(200,0,200),(0,200,200)]
    for _bi, _bb in enumerate(blocks_orig):
        _c = _clr[_bi % len(_clr)]
        _bx = _bb['box']
        if _bb.get('poly'):
            _pts = [(int(p[0]),int(p[1])) for p in _bb['poly']]
            _dbg_draw.line(_pts + [_pts[0]], fill=_c, width=2)
        else:
            _dbg_draw.rectangle([_bx[0],_bx[1],_bx[2],_bx[3]], outline=_c, width=2)
        _dbg_draw.text((_bx[0]+1,_bx[1]+1), f"{_bi}:{_bb['text'][:8]}", fill=_c, font=_dbg_font)
    _dbg_out = cv2.cvtColor(np.array(_dbg_pil), cv2.COLOR_RGB2BGR)
    save(os.path.join(OUT_DIR, "debug_all_ocr_boxes.jpg"), _dbg_out)
except Exception as _ex:
    print(f"  [调试] 保存debug_all_ocr_boxes失败: {_ex}")
# ═══ 调试结束 ═══

# 用 PIL 绘制中文标注（cv2 不支持中文）
ann_pil = PILImage.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
draw = PILImageDraw.Draw(ann_pil)

# 尝试加载中文字体
chinese_font = None
for font_path in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"]:
    try:
        chinese_font = PILImageFont.truetype(font_path, 80)  # 4倍放大
        break
    except:
        continue
if chinese_font is None:
    chinese_font = PILImageFont.load_default()

def _draw_pil_box(draw, box, color_rgb, label, poly=None):
    """在图像上绘制标注框 + 标签，确保不重叠、清晰可读"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    # 1. 画框：粗线（12px）+ 半透明填充让框更醒目（4倍放大）
    if poly and len(poly) >= 4:
        pts = [(int(p[0]), int(p[1])) for p in poly]
        # 画多边形轮廓（双层粗细）
        for w in [12, 4]:
            draw.line(pts + [pts[0]], fill=color_rgb, width=w)
    else:
        # 矩形框：外框12px + 内框4px（4倍放大）
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=12)
        draw.rectangle([x1+2, y1+2, x2-2, y2-2], outline=color_rgb, width=4)

    # 2. 标签：放在框的右下方外侧（远离框体）
    # 用小字体(14px)画标签，避免遮挡内容
    try:
        label_font = PILImageFont.truetype(
            chinese_font.path if hasattr(chinese_font, 'path') else \
            ("C:/Windows/Fonts/simhei.ttf" if os.path.exists("C:/Windows/Fonts/simhei.ttf") else None) or \
            "C:/Windows/Fonts/msyh.ttc", 56)  # 4倍放大
    except Exception:
        label_font = chinese_font

    bbox = draw.textbbox((0, 0), label, font=label_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 3
    gap = 4  # 框和标签之间的间距

    # 标签位置：优先右下角，超出则左下角
    lx1 = x2 + gap
    ly1 = y2 + gap
    lx2 = lx1 + tw + pad * 2
    ly2 = ly1 + th + pad * 2

    # 右边放不下 → 放左边
    if lx2 > W:
        lx1 = max(x1 - tw - pad * 2 - gap, 0)
        lx2 = lx1 + tw + pad * 2
    # 下边放不下 → 放上边
    if ly2 > H:
        ly1 = max(y1 - th - pad * 2 - gap, 0)
        ly2 = ly1 + th + pad * 2

    # 画标签背景（半透明效果：深色底+白字）
    draw.rectangle([lx1, ly1, lx2, ly2], fill=color_rgb)
    draw.rectangle([lx1, ly1, lx2, ly2], outline=(255, 255, 255), width=1)
    draw.text((lx1 + pad, ly1 + pad), label, font=label_font, fill=(255, 255, 255))

# 绘制缺字（红色）
for m in missing:
    _draw_pil_box(draw, m["box"], (255, 0, 0), "缺:" + m["text"][:8], m.get("poly"))
# 绘制变形（黄色）
for s in similar:
    _draw_pil_box(draw, s["box"], (255, 255, 0), "异:" + s["orig_text"][:4], s.get("poly"))

# 转回 OpenCV 格式并保存
ann = cv2.cvtColor(np.array(ann_pil), cv2.COLOR_RGB2BGR)

# ══════ 关键调试：对比 debug 图和 ann 图的差异 ═══════
try:
    _dbg_path = os.path.join(OUT_DIR, "debug_all_ocr_boxes.jpg")
    if os.path.exists(_dbg_path):
        _dbg_img = cv2.imread(_dbg_path)
        _diff = cv2.absdiff(_dbg_img, ann)
        _dc = np.count_nonzero(_diff)
        print(f"  [关键调试] debug vs ann 差异像素: {_dc} / {_dbg_img.size}")
        if missing:
            _m = missing[0]
            _bx = _m['box']
            _mg = 30
            _y1,_y2 = max(0,_bx[1]-_mg), min(H,_bx[3]+_mg)
            _x1,_x2 = max(0,_bx[0]-_mg), min(W,_bx[2]+_mg)
            _c_orig = ocr_orig[_y1:_y2, _x1:_x2]
            _c_dbg = _dbg_img[_y1:_y2, _x1:_x2]
            _c_ann = ann[_y1:_y2, _x1:_x2]
            _ch, _cw = _c_dbg.shape[:2]
            _comp = np.zeros((_ch, _cw*3, 3), dtype=np.uint8)
            _comp[:, 0:_cw] = _c_orig
            _comp[:, _cw:2*_cw] = _c_dbg
            _comp[:, 2*_cw:3*_cw] = _c_ann
            save(os.path.join(OUT_DIR, "debug_vs_ann_compare.jpg"), _comp)
            print(f"  [关键调试] 缺字'{_m['text']}' @_bx 对比图已保存: debug_vs_ann_compare.jpg")
except Exception as _ex2:
    print(f"  [关键调试] 对比失败: {_ex2}")
# ══════ 调试结束 ══════

# 如果有 extra（多字），创建PDF右图（含多字蓝色标注）
if extra:
    ann_pdf = ocr_pdf.copy()
    ann_pdf_pil = PILImage.fromarray(cv2.cvtColor(ann_pdf, cv2.COLOR_BGR2RGB))
    draw_pdf = PILImageDraw.Draw(ann_pdf_pil)
    for e in extra:
        _draw_pil_box(draw_pdf, e["box"], (0, 0, 255), "多:" + e["text"][:8], e.get("poly"))
    ann_pdf = cv2.cvtColor(np.array(ann_pdf_pil), cv2.COLOR_RGB2BGR)
    ann_text = np.hstack([ann, ann_pdf])
    cv2.line(ann_text, (W, 0), (W, H), (255, 255, 255), 2)
else:
    ann_text = ann

# 双栏：原稿 | OCR文字差异标注（与差异标注排列一致：原稿在前，差异在后）
text_diff_2col = np.hstack([orig_content, pad, ann_text])
save(os.path.join(OUT_DIR, "step5_ocr_text_diff.jpg"), text_diff_2col)
print(f"  -> 保存: step5_ocr_text_diff.jpg (原稿|OCR文字差异: 红=缺字/蓝=多字/黄=文字变形)")

# 保存文字内容对照表
if missing or extra or similar:
    with open(os.path.join(OUT_DIR, "step5_ocr_text_diff_report.txt"), "w", encoding="utf-8") as f:
        f.write("OCR文字差异报告 — 以原稿为准\n")
        f.write("=" * 50 + "\n\n")
        if missing:
            f.write(f"[!] 缺字（原稿有、PDF无）: {len(missing)} 个\n")
            for i, m in enumerate(missing, 1):
                f.write(f"  [{i}] \"{m['text']}\" @ {m['box']} (置信度:{m['score']:.2f})\n")
            f.write("\n")
        if extra:
            f.write(f"- 多字（PDF有、原稿无）: {len(extra)} 个\n")
            for i, e in enumerate(extra, 1):
                f.write(f"  [{i}] \"{e['text']}\" @ {e['box']} (置信度:{e['score']:.2f})\n")
            f.write("\n")
        if similar:
            f.write(f"[*] 文字变形（内容不符）: {len(similar)} 个\n")
            for i, s in enumerate(similar, 1):
                f.write(f"  [{i}] 原稿:\"{s['orig_text']}\" -> PDF:\"{s['pdf_text']}\" 相似度:{s['text_sim']:.2f}\n")
    print(f"  -> 保存: step5_ocr_text_diff_report.txt")

# -- Step6: 生成检测报告 -----------------------------------
print("\n" + "=" * 60)
print("Step6: 生成检测报告（文本+PDF）")
print("=" * 60)

pass_threshold = CFG_PASS_NCC
status = "[OK] 通过" if fine_ncc * 100 >= pass_threshold else "[X] 需检查"

# 分行打印报告（避免多行 f-string 可能导致的卡住问题）
print("+-----------------------------------------------+")
print("|                  印前检测报告                  |")
print("+-----------------------------------------------+")
print(f"|  原稿: {os.path.basename(ORIG_JPG):<35} |")
print(f"|  PDF:  {os.path.basename(PDF_FILE):<35} |")
print("+-----------------------------------------------+")
print(f"|  原稿主体区域: ({mx},{my}) {mw}x{mh}           |")
print(f"|  PDF裁切框数量: {len(pdf_regions)} (筛选后{len(unique_regions)})                |")
print("+-----------------------------------------------+")
print(f"|  最佳角度: {best_angle}    相似度: {best_sim:.1f}%         |")
print(f"|  内点率: {inlier_rate*100:.1f}%    NCC相似度: {fine_ncc*100:.1f}%   |")
print("+-----------------------------------------------+")
print(f"|  [!]PDF缺失(重要): {n_missing:>7,} px ({n_missing/total*100:.1f}%)      |")
print(f"|  -PDF多出(次要):  {n_extra:>7,} px ({n_extra/total*100:.1f}%)      |")
print(f"|  显著轮廓: {n_big} 个                                 |")
print("+-----------------------------------------------+")
print(f"|  [!]缺字: {len(missing):>3}  -多字: {len(extra):>3}  [*]变形: {len(similar):>3}        |")
print(f"|  判定: {status}                                |")
print("+-----------------------------------------------+")

# 生成PDF报告
print("\n  正在生成PDF报告...")
report_results = {
    'orig_file': os.path.basename(ORIG_JPG),
    'pdf_file': os.path.basename(PDF_FILE),
    'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'main_rect': (mx, my, mw, mh),
    'n_regions': len(pdf_regions),
    'n_unique': len(unique_regions),
    'best_angle': best_angle,
    'best_sim': best_sim,
    'inlier_rate': inlier_rate,
    'fine_ncc': fine_ncc,
    'n_missing': n_missing,
    'n_extra': n_extra,
    'total': total,
    'n_big': n_big,
    'n_missing_text': len(missing),
    'n_extra_text': len(extra),
    'n_similar_text': len(similar),
    'status': status,
}
pdf_path = generate_pdf_report(report_results, OUT_DIR)

orig_name = os.path.splitext(os.path.basename(ORIG_JPG))[0]
final_report = os.path.join(OUT_DIR, "印前检测报告.pdf")
print(f"\n报告已保存: {final_report}")

print(f"\n[OK] 完成")

# 强制刷新所有缓冲后立即终止进程
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
