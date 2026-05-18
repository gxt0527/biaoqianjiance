# -*- coding: utf-8 -*-
"""
PDF印前检测工具 — GUI主界面 v2
现代化设计，拖拽支持，缩略图预览，实时统计
"""
import os, sys, io, threading, subprocess, datetime, re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image as PILImage, ImageTk, ImageDraw

# ── 强制 UTF-8 输出 ────────────────────────────────────────
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── 路径配置 ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
COMPARE_PY   = os.path.join(SCRIPT_DIR, "compare_pdf_align.py")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "pdf_align_compare")

# ── 配色方案 v2 ─────────────────────────────────────────
BG           = "#F5F6FA"   # 页面背景（淡紫灰）
CARD_BG      = "#FFFFFF"   # 卡片背景
CARD_HOVER   = "#FAFBFF"   # 卡片悬停
ACCENT       = "#5B8DEF"   # 主色调（科技蓝）
ACCENT_HOVER = "#4A7DD4"   # hover
ACCENT_LIGHT = "#EBF2FF"   # 浅蓝背景
SUCCESS      = "#4CAF7D"   # 成功绿
WARN         = "#E05D5D"   # 警告红
ORANGE       = "#F59E42"   # 橙色
TEXT_D       = "#2C3142"   # 主文字
TEXT_M       = "#6B7280"   # 中等文字
TEXT_L       = "#9CA3AF"   # 浅文字
BORDER       = "#E5E7EB"   # 边框
DROP_ACTIVE  = "#DBEAFE"   # 拖拽激活背景


class PdfCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF 印前检测")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # 窗口最小尺寸
        self.root.minsize(640, 520)

        # 变量
        self.orig_path = tk.StringVar(value="")
        self.pdf_path  = tk.StringVar(value="")
        self.running   = False
        self.proc      = None
        self.thumb_orig = None  # 原稿缩略图
        self.thumb_pdf  = None
        self.report_stats = {}  # 检测统计结果

        self._setup_style()
        self._build_ui()

    # ── 样式配置 ──────────────────────────────────────────
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        # 进度条
        style.configure("Accent.Horizontal.TProgressbar",
                       background=ACCENT, troughcolor=BORDER,
                       borderwidth=0, thickness=5)

        # 滚动条
        style.configure("Modern.Vertical.TScrollbar",
                       background=BORDER, troughcolor=BG,
                       arrowcolor=TEXT_M, thickness=6)

    # ── UI 构建 ────────────────────────────────────────────
    def _build_ui(self):
        font_main   = ("Microsoft YaHei UI", 9)
        font_title  = ("Microsoft YaHei UI", 13, "bold")
        font_sub    = ("Microsoft YaHei UI", 10, "bold")
        font_small  = ("Microsoft YaHei UI", 8)
        font_mono   = ("Consolas", 8)

        # ════════════ 标题栏 ════════════
        header = tk.Frame(self.root, bg=ACCENT, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)

        # 左侧标题
        title_frame = tk.Frame(header, bg=ACCENT)
        title_frame.pack(side="left", padx=20, pady=10)
        tk.Label(title_frame, text="⚡", font=("Segoe UI Emoji", 14), fg="white", bg=ACCENT).pack(side="left", padx=(0, 6))
        tk.Label(title_frame, text="PDF 印前检测", font=font_title, fg="white", bg=ACCENT).pack(side="left")

        # 右侧状态
        status_frame = tk.Frame(header, bg=ACCENT)
        status_frame.pack(side="right", padx=16, pady=8)
        self.status_dot = tk.Label(status_frame, text="●", font=("Segoe UI", 10), fg="#93C5FD", bg=ACCENT)
        self.status_dot.pack(side="left", padx=(0, 4))
        self.status_text = tk.Label(status_frame, text="就绪", font=font_small, fg="#BFDBFE", bg=ACCENT)
        self.status_text.pack(side="left")

        # ════════════ 主内容区 ════════════
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # ── 左侧面板：文件选择 ──
        left_panel = tk.Frame(body, bg=BG)
        left_panel.pack(side="left", fill="both", expand=True)

        # 文件选择卡片
        file_card = tk.Frame(left_panel, bg=CARD_BG, relief="flat", bd=0)
        file_card.pack(fill="x", pady=(0, 10))
        # 卡片阴影效果（用多层Frame模拟）
        for i in range(3, 0, -1):
            shadow = tk.Frame(left_panel, bg="#E8E9EF", height=1)
            shadow.place(x=i, y=i, relwidth=1, width=-2*i, in_=file_card)
        file_card.lift()

        inner_file = tk.Frame(file_card, bg=CARD_BG)
        inner_file.pack(fill="x", padx=16, pady=14)

        # 标题
        tk.Label(inner_file, text="文件选择", font=font_sub, fg=TEXT_D, bg=CARD_BG).pack(anchor="w")

        # 原稿选择区
        orig_frame = tk.Frame(inner_file, bg=CARD_BG)
        orig_frame.pack(fill="x", pady=(12, 6))

        # 原稿标签行
        orig_header = tk.Frame(orig_frame, bg=CARD_BG)
        orig_header.pack(fill="x", pady=(0, 6))
        tk.Label(orig_header, text="📷 原稿图片", font=font_main, fg=TEXT_D, bg=CARD_BG).pack(side="left")
        tk.Label(orig_header, text="JPG / PNG / TIF", font=font_small, fg=TEXT_L, bg=CARD_BG).pack(side="right")

        # 原稿输入+按钮
        orig_input = tk.Frame(orig_frame, bg=CARD_BG)
        orig_input.pack(fill="x")
        self.ent_orig = tk.Entry(orig_input, textvariable=self.orig_path, font=font_main,
                                 bg="white", fg=TEXT_D, bd=1, relief="solid", highlightthickness=1,
                                 highlightcolor=ACCENT, highlightbackground=BORDER)
        self.ent_orig.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        self.ent_orig.bind("<FocusIn>", lambda e: self._highlight_border(self.ent_orig, True))
        self.ent_orig.bind("<FocusOut>", lambda e: self._highlight_border(self.ent_orig, False))

        btn_orig = tk.Button(orig_input, text="浏览", command=self._select_orig,
                            font=font_main, fg="white", bg=ACCENT,
                            activebackground=ACCENT_HOVER, bd=0, padx=14, pady=4,
                            cursor="hand2", relief="flat")
        btn_orig.pack(side="right")

        # PDF选择区
        pdf_frame = tk.Frame(inner_file, bg=CARD_BG)
        pdf_frame.pack(fill="x", pady=(0, 0))

        pdf_header = tk.Frame(pdf_frame, bg=CARD_BG)
        pdf_header.pack(fill="x", pady=(0, 6))
        tk.Label(pdf_header, text="📄 PDF 拼版", font=font_main, fg=TEXT_D, bg=CARD_BG).pack(side="left")
        tk.Label(pdf_header, text="PDF 文件", font=font_small, fg=TEXT_L, bg=CARD_BG).pack(side="right")

        pdf_input = tk.Frame(pdf_frame, bg=CARD_BG)
        pdf_input.pack(fill="x")
        self.ent_pdf = tk.Entry(pdf_input, textvariable=self.pdf_path, font=font_main,
                                bg="white", fg=TEXT_D, bd=1, relief="solid", highlightthickness=1,
                                highlightcolor=ACCENT, highlightbackground=BORDER)
        self.ent_pdf.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        self.ent_pdf.bind("<FocusIn>", lambda e: self._highlight_border(self.ent_pdf, True))
        self.ent_pdf.bind("<FocusOut>", lambda e: self._highlight_border(self.ent_pdf, False))

        btn_pdf = tk.Button(pdf_input, text="浏览", command=self._select_pdf,
                           font=font_main, fg="white", bg=ACCENT,
                           activebackground=ACCENT_HOVER, bd=0, padx=14, pady=4,
                           cursor="hand2", relief="flat")
        btn_pdf.pack(side="right")

        # 拖拽提示
        drop_hint = tk.Label(inner_file, text="支持拖拽文件到输入框快速加载",
                            font=font_small, fg=TEXT_L, bg=CARD_BG)
        drop_hint.pack(anchor="w", pady=(8, 0))

        # ── 统计摘要卡片（检测完成后显示）─
        self.stats_card = tk.Frame(left_panel, bg=CARD_BG, relief="flat")
        self.stats_card.pack(fill="x", pady=(0, 10))
        self.stats_card.pack_forget()  # 初始隐藏

        inner_stats = tk.Frame(self.stats_card, bg=CARD_BG)
        inner_stats.pack(fill="x", padx=16, pady=12)

        tk.Label(inner_stats, text="📊 检测摘要", font=font_sub, fg=TEXT_D, bg=CARD_BG).pack(anchor="w")

        self.stats_labels = {}
        stats_items = [
            ("相似度", "ncc", "%"),
            ("PDF缺失", "missing", "px"),
            ("PDF多出", "extra", "px"),
            ("缺字", "missing_text", "个"),
            ("变形", "similar_text", "个"),
        ]
        for label, key, unit in stats_items:
            row = tk.Frame(inner_stats, bg=CARD_BG)
            row.pack(fill="x", pady=(8, 0))
            tk.Label(row, text=label, font=font_main, fg=TEXT_M, bg=CARD_BG, width=8, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="-", font=("Microsoft YaHei UI", 10, "bold"), fg=TEXT_D, bg=CARD_BG)
            val_lbl.pack(side="left")
            tk.Label(row, text=unit, font=font_small, fg=TEXT_L, bg=CARD_BG).pack(side="left", padx=(2, 0))
            self.stats_labels[key] = val_lbl

        # ── 操作按钮 ──
        btn_frame = tk.Frame(left_panel, bg=BG)
        btn_frame.pack(fill="x", pady=(0, 10))

        self.btn_report = tk.Button(btn_frame, text="📋 查看报告",
                                    command=self._open_report,
                                    font=font_main, fg="white", bg=TEXT_L,
                                    activebackground=TEXT_M, bd=0, padx=16, pady=8,
                                    cursor="hand2", relief="flat", state="disabled")
        self.btn_report.pack(side="left")

        self.btn_start = tk.Button(btn_frame, text="▶ 开始检测",
                                   command=self._start_check,
                                   font=font_main, fg="white", bg=ACCENT,
                                   activebackground=ACCENT_HOVER, bd=0, padx=20, pady=8,
                                   cursor="hand2", relief="flat")
        self.btn_start.pack(side="right")

        # ════════════ 右侧面板：日志 ════════════
        right_panel = tk.Frame(body, bg=BG)
        right_panel.pack(side="right", fill="both", expand=True, padx=(12, 0))

        log_card = tk.Frame(right_panel, bg=CARD_BG, relief="flat")
        log_card.pack(fill="both", expand=True)

        inner_log = tk.Frame(log_card, bg=CARD_BG)
        inner_log.pack(fill="both", expand=True, padx=14, pady=12)

        # 日志标题
        log_header = tk.Frame(inner_log, bg=CARD_BG)
        log_header.pack(fill="x", pady=(0, 10))
        tk.Label(log_header, text="📝 检测日志", font=font_sub, fg=TEXT_D, bg=CARD_BG).pack(side="left")
        self.step_label = tk.Label(log_header, text="", font=font_small, fg=ACCENT, bg=CARD_BG)
        self.step_label.pack(side="right")

        # 进度条
        self.pbar = ttk.Progressbar(inner_log, mode="indeterminate",
                                    style="Accent.Horizontal.TProgressbar")
        self.pbar.pack(fill="x", pady=(0, 10))

        # 日志文本框
        log_text_frame = tk.Frame(inner_log, bg="#F8F9FC",
                                   highlightbackground=BORDER, highlightthickness=1)
        log_text_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_text_frame, font=font_mono,
                                bg="#F8F9FC", fg=TEXT_D, bd=0,
                                padx=10, pady=8, highlightthickness=0,
                                insertbackground=ACCENT, wrap="word",
                                state="disabled", spacing1=2, spacing2=1)
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_text_frame, command=self.log_text.yview,
                                  style="Modern.Vertical.TScrollbar")
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # ════════════ 底部状态栏 ════════════
        footer = tk.Frame(self.root, bg=CARD_BG, height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        tk.Label(footer, text="OpenCV + PaddleOCR + PyMuPDF",
                 font=font_small, fg=TEXT_L, bg=CARD_BG).pack(side="right", padx=14, pady=4)
        tk.Label(footer, text="拖拽文件快速加载",
                 font=font_small, fg=TEXT_L, bg=CARD_BG).pack(side="left", padx=14, pady=4)

    def _highlight_border(self, widget, focused):
        """输入框焦点高亮"""
        if focused:
            widget.config(highlightbackground=ACCENT, highlightcolor=ACCENT)
        else:
            widget.config(highlightbackground=BORDER, highlightcolor=BORDER)

    # ── 文件选择 ───────────────────────────────────────────
    def _select_orig(self):
        path = filedialog.askopenfilename(
            title="选择原稿图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.tif *.tiff"),
                       ("所有文件", "*.*")])
        if path:
            self.orig_path.set(path)
            self._log(f"[选择] 原稿: {os.path.basename(path)}", ACCENT)

    def _select_pdf(self):
        path = filedialog.askopenfilename(
            title="选择拼版 PDF 文件",
            filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")])
        if path:
            self.pdf_path.set(path)
            self._log(f"[选择] PDF: {os.path.basename(path)}", ACCENT)

    # ── 日志输出 ───────────────────────────────────────────
    def _log(self, msg, color=None):
        self.log_text.config(state="normal")
        if color:
            tag = f"c{abs(hash(color)) % 10000}"
            self.log_text.tag_config(tag, foreground=color)
            self.log_text.insert("end", msg + "\n", tag)
        else:
            self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_step(self, text):
        self.step_label.config(text=text)

    def _set_status(self, text, dot_color="#93C5FD"):
        self.status_text.config(text=text)
        self.status_dot.config(fg=dot_color)

    def _set_running(self, is_running):
        self.running = is_running
        if is_running:
            self.btn_start.config(text="⏳ 检测中...", bg=TEXT_L,
                                  state="disabled", cursor="arrow")
            self.btn_report.config(state="disabled")
            self.stats_card.pack_forget()  # 隐藏统计卡片
            self.pbar.start(10)
        else:
            self.pbar.stop()
            self.btn_start.config(text="▶ 开始检测", bg=ACCENT,
                                state="normal", cursor="hand2")

    def _show_stats(self, stats):
        """显示检测统计摘要"""
        self.report_stats = stats

        # 更新统计值
        ncc = stats.get('fine_ncc', 0) * 100
        self.stats_labels['ncc'].config(
            text=f"{ncc:.1f}%",
            fg=SUCCESS if ncc >= 95 else (ORANGE if ncc >= 85 else WARN)
        )
        self.stats_labels['missing'].config(text=f"{stats.get('n_missing', 0):,}")
        self.stats_labels['extra'].config(text=f"{stats.get('n_extra', 0):,}")
        self.stats_labels['missing_text'].config(text=f"{stats.get('n_missing_text', 0)}")
        self.stats_labels['similar_text'].config(text=f"{stats.get('n_similar_text', 0)}")

        # 显示卡片
        self.stats_card.pack(fill="x", pady=(0, 10), before=self.btn_start.master)

    def _parse_stats_from_log(self, text):
        """从日志中解析检测统计"""
        stats = {}

        # NCC相似度
        ncc_match = re.search(r'NCC.*?(\d+\.?\d*)%', text)
        if ncc_match:
            stats['fine_ncc'] = float(ncc_match.group(1)) / 100

        # PDF缺失
        miss_match = re.search(r'PDF缺失.*?([\d,]+)\s*px', text)
        if miss_match:
            stats['n_missing'] = int(miss_match.group(1).replace(',', ''))

        # PDF多出
        extra_match = re.search(r'PDF多出.*?([\d,]+)\s*px', text)
        if extra_match:
            stats['n_extra'] = int(extra_match.group(1).replace(',', ''))

        # 缺字
        miss_txt_match = re.search(r'缺字.*?(\d+)', text)
        if miss_txt_match:
            stats['n_missing_text'] = int(miss_txt_match.group(1))

        # 变形
        sim_txt_match = re.search(r'变形.*?(\d+)', text)
        if sim_txt_match:
            stats['n_similar_text'] = int(sim_txt_match.group(1))

        return stats

    # ── 开始检测 ───────────────────────────────────────────
    def _start_check(self):
        orig = self.orig_path.get().strip()
        pdf  = self.pdf_path.get().strip()

        if not orig:
            messagebox.showwarning("提示", "请选择原稿图片文件")
            self.ent_orig.focus()
            return
        if not pdf:
            messagebox.showwarning("提示", "请选择拼版 PDF 文件")
            self.ent_pdf.focus()
            return
        if not os.path.exists(orig):
            messagebox.showerror("错误", f"原稿文件不存在:\n{orig}")
            return
        if not os.path.exists(pdf):
            messagebox.showerror("错误", f"PDF 文件不存在:\n{pdf}")
            return

        # 清空日志
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._set_step("")

        # 更新状态
        self._set_status("检测中", "#FCD34D")
        self._set_running(True)
        self._log("━" * 40)
        self._log(f"  开始检测  {datetime.datetime.now().strftime('%H:%M:%S')}")
        self._log("━" * 40)

        # 启动子线程
        t = threading.Thread(target=self._run_worker, args=(orig, pdf), daemon=True)
        t.start()

    def _run_worker(self, orig, pdf):
        """子线程：运行检测脚本"""
        import time
        env = os.environ.copy()
        env["PDF_CHECKER_ORIG"] = orig
        env["PDF_CHECKER_PDF"]  = pdf

        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", COMPARE_PY],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                cwd=SCRIPT_DIR,
            )

            log_output = []  # 收集日志用于解析统计
            line_count = 0
            completed = False
            start_time = time.time()
            TIMEOUT_SECONDS = 300

            while True:
                if time.time() - start_time > TIMEOUT_SECONDS:
                    self.root.after(0, lambda: self._log("⚠ 检测超时，强制结束", WARN))
                    if self.proc.poll() is None:
                        self.proc.kill()
                    break

                poll_ret = self.proc.poll()

                try:
                    raw_bytes = self.proc.stdout.readline()
                    if not raw_bytes:
                        if poll_ret is not None:
                            break
                        time.sleep(0.1)
                        continue
                except Exception:
                    if poll_ret is not None:
                        break
                    time.sleep(0.1)
                    continue

                # 解码
                try:
                    line = raw_bytes.decode('utf-8', errors='strict')
                except (UnicodeDecodeError, ValueError):
                    try:
                        line = raw_bytes.decode('gbk', errors='replace')
                    except Exception:
                        line = raw_bytes.decode('utf-8', errors='replace')

                line = line.rstrip('\n')
                log_output.append(line)
                line_count += 1

                # 检测完成
                if "[OK]" in line and "完成" in line:
                    completed = True
                    time.sleep(0.5)
                    if self.proc.poll() is None:
                        try:
                            self.proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.proc.kill()
                    break

                display = self._filter_line(line)
                if display is not None:
                    color = self._line_color(line)
                    self.root.after(0, lambda d=display, c=color: self._log(d, c))

                if "Step" in line and "==" not in line:
                    step = line.strip()
                    self.root.after(0, lambda s=step: self._set_step(s[:28]))

            ret = self.proc.poll()
            if ret is None:
                ret = self.proc.wait()
            self.proc = None

            # 解析统计并显示
            if completed:
                stats = self._parse_stats_from_log('\n'.join(log_output))
                self.root.after(0, lambda s=stats: self._show_stats(s))

            final_ret = 0 if completed else (ret if ret is not None else 0)
            self.root.after(0, lambda r=final_ret: self._on_done(r))
            self.root.after(100, lambda: self._set_step(""))

        except Exception as e:
            self.root.after(0, lambda: self._log(f"✗ 进程异常: {str(e)}", WARN))
            self.root.after(0, lambda: self._on_done(1))

    def _filter_line(self, line):
        """过滤日志行"""
        s = line.strip()
        if not s:
            return None

        skip_kw = ["Model files already", "Creating model:", "ccache", "download",
                   "warnings.warn", "UserWarning", "ResourceWarning", "DeprecationWarning",
                   "FutureWarning", "用提供的模式无法找到文件", "pattern"]
        if any(k in s for k in skip_kw):
            return None

        if s.startswith("=="):
            return None

        if re.match(r'^Step\d', s):
            return s

        if "[Step" in s:
            return s[:100]

        key_markers = ["[OK]", "[X]", "[!]", "[*]", "[PDF", "[自动", "[形态学",
                       "[资源限制]", "[OCR资源]", "NCC", "内点率", "相似度"]
        if any(s.startswith(k) or k in s for k in key_markers):
            return s[:100]

        progress_kw = ["完成", "保存", "识别", "差异", "裁切框", "主体区域",
                      "渲染", "对齐", "提升", "检测", "匹配", "角度", "文字"]
        if any(k in s for k in progress_kw):
            return s[:100]

        if re.search(r'[\u4e00-\u9fa5]+.*:\s*\d+', s):
            return s[:100]

        if s.startswith(" ") and len(s) < 30:
            return None

        return s[:80] if len(s) > 5 else None

    def _line_color(self, line):
        if "✗" in line or "失败" in line or "Error" in line:
            return WARN
        if "✓" in line or "完成" in line:
            return SUCCESS
        if "⚠" in line or "差异" in line:
            return ORANGE
        if "Step" in line:
            return ACCENT
        return TEXT_M

    def _on_done(self, retcode):
        self._set_running(False)
        if retcode == 0:
            self._set_status("检测完成", SUCCESS)
            self._log("", None)
            self._log("✓ 检测完成", SUCCESS)
            self.btn_report.config(bg=SUCCESS, state="normal")
        else:
            self._set_status("检测失败", WARN)
            self._log("", None)
            self._log("✗ 检测异常退出，请检查日志", WARN)
            self.btn_report.config(bg=WARN, state="normal")

    # ── 查看报告 ───────────────────────────────────────────
    def _open_report(self):
        """打开检测报告"""
        orig = self.orig_path.get().strip()
        if orig:
            report_dir = os.path.dirname(os.path.abspath(orig))
        else:
            report_dir = OUTPUT_DIR
        pdf_report = os.path.join(report_dir, "印前检测报告.pdf")
        if os.path.exists(pdf_report):
            os.startfile(pdf_report)
        elif os.path.exists(report_dir):
            os.startfile(report_dir)
        else:
            messagebox.showwarning("提示", "未找到报告文件，请先运行检测")


def main():
    import locale
    locale.setlocale(locale.LC_ALL, '')

    root = tk.Tk()
    default_font = ("Microsoft YaHei UI", 10)
    root.option_add("*Font", default_font)
    app = PdfCheckerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
