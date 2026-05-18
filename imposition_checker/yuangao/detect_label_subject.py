# -*- coding: utf-8 -*-
"""
标签主体检测 + CMYK 原稿绘制
- 检测阶段可以随意颜色
- 绘制阶段在 CMYK 副本绘制 100% 黄色，线宽3
- 保留原稿色彩模式、分辨率、位深
- 弹窗显示
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

# --------------------------
# 输入文件
# --------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.environ.get("LABEL_IMG") or os.path.join(BASE, "label.jpg")
OUTPUT_PATH = os.path.join(BASE, "label_debug.jpg")  # 保存 CMYK 原稿副本

# --------------------------
# 读取原稿 CMYK
# --------------------------
img_cmyk = Image.open(IMG_PATH)
print(f"原稿模式: {img_cmyk.mode}, 尺寸: {img_cmyk.size}")

# --------------------------
# 检测阶段：使用 RGB 副本
# --------------------------
if img_cmyk.mode == "CMYK":
    img_detect = img_cmyk.convert("RGB")
else:
    img_detect = img_cmyk.copy()

img_np = np.array(img_detect)
if len(img_np.shape) == 3:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
else:
    gray = img_np.copy()

H, W = gray.shape[:2]

# 灰度 + 边缘 + 行列投影
edges = cv2.Canny(gray, 50, 150)
row_density = np.mean(gray < 240, axis=1)
row_edge = np.mean(edges, axis=1)/255
row_score = 0.5*row_density + 0.5*row_edge
row_score_smooth = np.convolve(row_score, np.ones(5)/5, mode='same')
thresh = np.max(row_score_smooth)*0.3
candidate_rows = np.where(row_score_smooth > thresh)[0]
top_row = candidate_rows[0]
bottom_row = candidate_rows[-1]

col_density = np.mean(gray[top_row:bottom_row+1, :] < 240, axis=0)
col_thresh = np.max(col_density)*0.3
candidate_cols = np.where(col_density > col_thresh)[0]
left_col = candidate_cols[0]
right_col = candidate_cols[-1]

# ROI 二值化 + 闭运算 + 连通区域
roi = gray[top_row:bottom_row+1, left_col:right_col+1]
_, binary = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = roi.shape[0]*roi.shape[1]*0.05
valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
if not valid_contours:
    valid_contours = contours

x_list, y_list, x2_list, y2_list = [], [], [], []
for c in valid_contours:
    x, y, w, h = cv2.boundingRect(c)
    x_list.append(x)
    y_list.append(y)
    x2_list.append(x+w)
    y2_list.append(y+h)

tx_min = min(x_list)
ty_min = min(y_list)
tx_max = max(x2_list)
ty_max = max(y2_list)

# 转换为原图坐标
left = left_col + tx_min
top = top_row + ty_min
right = left_col + tx_max
bottom = top_row + ty_max

print(f"检测结果主体矩形: left={left}, top={top}, right={right}, bottom={bottom}")

# --------------------------
# 绘制阶段：在 CMYK 原稿副本绘制
# --------------------------
vis_cmyk = img_cmyk.copy()
draw = ImageDraw.Draw(vis_cmyk)

# CMYK黄色 (0,0,100,0) 对应 (0,0,255,0)，线宽3
draw.rectangle([left, top, right, bottom], outline=(0,0,255,0), width=3)

# 保存 CMYK 副本
vis_cmyk.save(OUTPUT_PATH)
print(f"标注图已保存: {OUTPUT_PATH}")

# --------------------------
# 弹窗显示：CMYK -> RGB
# --------------------------
vis_rgb = vis_cmyk.convert("RGB")
vis_cv = cv2.cvtColor(np.array(vis_rgb), cv2.COLOR_RGB2BGR)
cv2.imshow("Label Subject Detection", vis_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()