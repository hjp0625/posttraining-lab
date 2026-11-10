# -*- coding: utf-8 -*-
"""Build the SFT vs GRPO experiment report (DOCX)."""
import json
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "report")
os.makedirs(OUT, exist_ok=True)
DOCX = os.path.join(OUT, "SFT_vs_GRPO_小规模对比实验报告.docx")

with open(os.path.join(RES, "summary.json"), encoding="utf-8") as f:
    S = json.load(f)

NAVY = RGBColor(0x14, 0x21, 0x3D)
DARK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x47, 0x55, 0x69)
ACCENT = RGBColor(0x2F, 0x6D, 0xB3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)
sec.header_distance = Cm(1.0)
sec.footer_distance = Cm(0.9)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(10.0)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
style.paragraph_format.line_spacing = 1.2
style.paragraph_format.space_after = Pt(2.0)
style.paragraph_format.space_before = Pt(0)


def set_run(run, east="宋体", ascii_f="Times New Roman", size=10.0, bold=False, color=DARK):
    run.font.name = ascii_f
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), ascii_f)
    rFonts.set(qn("w:hAnsi"), ascii_f)
    rFonts.set(qn("w:eastAsia"), east)


def add_rich(p, text, size=10.0, color=DARK, bold=False):
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        set_run(r, size=size, bold=(bold or i % 2 == 1), color=color)


def body(text, indent=True, size=10.0, after=2.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(after)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    add_rich(p, text, size=size)
    return p


def h1(text, num=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{num}  {text}" if num else text)
    set_run(r, east="黑体", ascii_f="Arial", size=12.5, bold=True, color=NAVY)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "3")
    bottom.set(qn("w:color"), "2F6DB3")
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run(r, east="黑体", ascii_f="Arial", size=10.6, bold=True, color=ACCENT)
    return p


def formula(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run(r, ascii_f="Cambria Math", size=9.4, color=NAVY)


def add_header(section):
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("SFT 与 GRPO 的可验证任务对比实验")
    set_run(r, size=8, color=MUTED)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "D7DEE8")
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("— ")
    set_run(r, size=8.5, color=MUTED)
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "1"
    fld3 = OxmlElement("w:fldChar"); fld3.set(qn("w:fldCharType"), "end")
    r2 = p.add_run(); r2._r.append(fld1); r2._r.append(instr); r2._r.append(fld2); r2._r.append(t); r2._r.append(fld3)
    set_run(r2, size=8.5, color=MUTED)
    r3 = p.add_run(" —")
    set_run(r3, size=8.5, color=MUTED)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def table(headers, rows, widths, caption, note=None, font=8.6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(caption)
    set_run(r, east="黑体", ascii_f="Arial", size=9.0, bold=True, color=NAVY)

    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    t.style = "Table Grid"
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "4"); el.set(qn("w:space"), "0"); el.set(qn("w:color"), "CBD5E1")
        borders.append(el)
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)

    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.width = widths[i]
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        set_run(r, east="黑体", ascii_f="Arial", size=font, bold=True, color=WHITE)
        shade(c, "14213D")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            c = cells[ci]
            c.width = widths[ci]
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(val)
            set_run(r, size=font, bold=(ci == 0), color=NAVY if ci == 0 else DARK)
            if ci == 0:
                shade(c, "EEF3F9")
            elif ri % 2 == 1:
                shade(c, "F8FAFC")
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(note)
        set_run(r, size=7.6, color=MUTED)
    return t


add_header(sec)
add_footer(sec)

# ---------------- title ----------------
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("小规模训练实验 · 后训练方法对比")
set_run(r, east="黑体", ascii_f="Arial", size=9, bold=True, color=ACCENT)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(2)
p.paragraph_format.line_spacing = 1.08
r = p.add_run("SFT 与 GRPO 在可验证算术任务上的对比")
set_run(r, east="微软雅黑", ascii_f="Microsoft YaHei", size=18.5, bold=True, color=NAVY)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(4)
r = p.add_run("训练损失、任务准确率、输出长度与计算开销")
set_run(r, east="微软雅黑", ascii_f="Microsoft YaHei", size=12, color=MUTED)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(8)
r = p.add_run("2026 年 11 月 · 0.6M 参数字符级 Transformer · CPU 14 线程 · 单随机种子")
set_run(r, size=8.6, color=MUTED)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(6)
pPr = p._p.get_or_add_pPr()
pBdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "10"); bottom.set(qn("w:space"), "1"); bottom.set(qn("w:color"), "14213D")
pBdr.append(bottom); pPr.append(pBdr)

# ---------------- abstract ----------------
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(2)
r = p.add_run("摘要")
set_run(r, east="黑体", ascii_f="Arial", size=11, bold=True, color=NAVY)

abstract = ("本文在一个可验证的两位整数加法任务上，比较监督微调（SFT）与组相对策略优化（GRPO）。"
            "模型为 0.6M 参数的字符级因果 Transformer，训练集 5,000 题，验证/测试各 500 题。"
            "SFT 训练 10 个 epoch 后测试精确匹配率为 80.8%；以其为初始策略，GRPO 训练 200 步后为 79.6%，"
            "点估计低 1.2 个百分点（McNemar p=0.114），平均输出长度 2.46 token 基本不变，"
            "但训练墙钟时间为 SFT 的 3.44 倍。结果表明：在本实验的模型规模、奖励密度与任务难度下，"
            "GRPO 没有带来可测的准确率收益，却显著增加计算开销；其有效性依赖更强的基座、更密的奖励、"
            "更长的推理链与更大的探索预算。")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.18
p.paragraph_format.space_after = Pt(3)
r = p.add_run(abstract)
set_run(r, size=9.0, color=DARK)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("关键词：")
set_run(r, east="黑体", ascii_f="Arial", size=9.0, bold=True, color=NAVY)
r = p.add_run("SFT；GRPO；可验证奖励；强化学习；计算开销；负结果")
set_run(r, size=9.0, color=DARK)

# ---------------- 1 ----------------
h1("实验问题与假设", "1")
body("SFT 用示范数据最大化答案似然，GRPO 用一组采样答案的相对优劣构造优势，并用可验证奖励直接优化任务指标。"
     "理论上，GRPO 应当突破 SFT 的“只会模仿”上限；但在小模型、短答案、稀疏奖励的条件下，这一优势未必成立。"
     "本实验用两位整数加法做一个可控的“沙盒”，回答三个问题：")
body("**H1（准确率）**：从同一 SFT 检查点出发，GRPO 能否提升测试集精确匹配率？")
body("**H2（长度）**：GRPO 是否会让输出变长，出现文献中常见的长度膨胀？")
body("**H3（开销）**：在相同硬件与实现下，GRPO 的训练时间、采样次数和生成 token 数比 SFT 高多少？")
body("选择两位整数加法的原因是：答案可自动判定，不需要奖励模型；数据可完全枚举且可去重；训练与评测闭环短，"
     "适合在单机 CPU 上重复。它不是要模拟大语言模型的全部能力，而是检验“可验证奖励 + 策略梯度”在小规模下是否仍然有效。")

# ---------------- 2 ----------------
h1("方法", "2")
h2("2.1  任务与数据")
body("任务形式为“Q:a+b=”，其中 a、b 为 0—99 的整数，模型输出 a+b 的十进制结果。"
     "从 10,000 个唯一有序对中随机划分 5,000 训练、500 验证、500 测试，三者互不重叠。"
     "字符级词表包含数字、Q、:、+、=、空格与特殊 token；最大生成长度 6 个 token，足以覆盖 0—198 的答案。"
     "评测指标为精确匹配（exact match），另外记录平均生成长度（不含 EOS）。")
h2("2.2  模型与 SFT")
body("模型是 3 层、4 头、128 维的字符级因果 Transformer，参数量 600,576，上下文长度 24，dropout 为 0。"
     "SFT 采用教师强制交叉熵，只对答案与 EOS 计算损失，prompt 部分 mask 为 -100：")
formula("L_SFT = −E_(x,y)~D [ log πθ(y | x) ]")
body("训练 10 个 epoch，batch size 64，AdamW 学习率 3×10⁻³，余弦退火，weight decay 0.01，共 790 个优化步、"
     "50,000 个训练样本。按验证集精确匹配率选择最佳检查点。")
h2("2.3  GRPO")
body("GRPO 从 SFT 最佳检查点初始化。对每个 prompt 采样 G=8 个完整答案，得到奖励 r_i∈{0,1}；"
     "组内归一化得到优势 Â_i，再用 PPO 式重要性比率与裁剪优化策略：")
formula("Â_i = (r_i − mean(r)) / (std(r) + 1e−4)")
formula("L_GRPO = −E[ min(ρ_iÂ_i, clip(ρ_i, 1−ε, 1+ε)Â_i) ] + β·KL(πθ‖πref)")
body("其中 ρ_i=πθ(y_i|x)/πold(y_i|x)，ε=0.2，β=0.005，KL 使用 token 级 K3 估计；参考模型为冻结的 SFT 检查点。"
     "训练 200 步，每步 8 个 prompt、8 个采样，共 12,800 次生成、32,093 个生成 token；"
     "学习率 3×10⁻⁵，采样温度 1.5，以增加组内多样性。奖励只使用精确匹配，不加入格式奖励或长度奖励。")
h2("2.4  统计与开销口径")
body("测试集 500 题，报告精确匹配率与 95% 正态近似置信区间；由于 SFT 与 GRPO 在同一测试集上评测，"
     "同时使用配对 McNemar 检验比较两者的差异。计算开销记录优化步数、训练样本/生成次数、生成 token 数、"
     "训练墙钟时间；SFT 与 GRPO 在同一进程、同一 CPU（14 线程、无 GPU）上顺序运行。")

# ---------------- 3 ----------------
h1("结果", "3")
h2("3.1  训练损失与奖励")
body("SFT 的 epoch 平均交叉熵从 1.61 降到 0.233，验证准确率从 1.4% 升到 79.0%，说明模型确实学到了加法映射，"
     "但仍有约 20% 的验证错误。GRPO 的采样组精确匹配率在 23.4%—54.7% 之间波动，200 步后为 39.1%，"
     "没有呈现稳定的上升趋势；KL 从 0 缓慢升到 0.0086，策略损失（裁剪替代目标）均值接近 0。")
body("需要强调：GRPO 的“损失”是带基线、带裁剪的策略目标，均值接近 0 并不等于没有梯度；"
     "真正有信息量的是训练奖励、KL 与验证准确率。图 1 显示，GRPO 的验证准确率始终在 78%—80% 附近，"
     "没有随训练步数持续上升。")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
p.add_run().add_picture(os.path.join(FIG, "fig1_training_dynamics.png"), width=Cm(16.4))
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(5)
r = p.add_run("图 1  SFT 与 GRPO 的训练动态")
set_run(r, east="黑体", ascii_f="Arial", size=8.4, bold=True, color=NAVY)
r = p.add_run("（左：SFT 训练损失与验证准确率；右：GRPO 采样奖励与验证准确率）")
set_run(r, size=8.0, color=MUTED)

h2("3.2  任务准确率")
body("SFT 的测试精确匹配率为 80.8%（95% CI ±3.45pp），GRPO 为 79.6%（±3.53pp），点估计低 1.2 个百分点。"
     "配对 McNemar 检验中，SFT 对、GRPO 错的样本为 8 个，SFT 错、GRPO 对的样本为 2 个，"
     "χ²=2.50，p=0.114；在 0.05 水平上不能拒绝“两者无差异”。换句话说，本实验没有证据表明 GRPO 提升了准确率，"
     "也不能把 -1.2pp 解释为可靠的性能下降。")
body("按是否需要个位进位拆分：无进位题上 SFT/GRPO 为 80.6%/79.1%，有进位题为 81.0%/80.2%。"
     "GRPO 在两个子组都没有优势，说明它没有学会新的进位规则，也没有把错误集中到某一类题上。")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0)
p.add_run().add_picture(os.path.join(FIG, "fig2_main_comparison.png"), width=Cm(16.4))
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(5)
r = p.add_run("图 2  测试准确率、输出长度与 CPU 训练开销")
set_run(r, east="黑体", ascii_f="Arial", size=8.4, bold=True, color=NAVY)
r = p.add_run("（误差线为 95% 置信区间；时间含周期性验证）")
set_run(r, size=8.0, color=MUTED)

h2("3.3  输出长度")
body("SFT 平均生成 2.462 个 token，GRPO 为 2.458 个，差异仅 0.004 个 token。本任务答案只有 2—3 位数字，"
     "且训练/评测都限制最多 6 个 token，因此没有出现文献中常见的“GRPO 越训越长”。"
     "这提示长度膨胀并不是 GRPO 的必然属性，而是与任务是否允许长推理链、奖励是否惩罚长度、"
     "以及模型是否拥有可扩展的 CoT 空间有关。")
h2("3.4  计算开销")
body("SFT 训练 790 步、50,000 个样本，墙钟 49.1 秒；GRPO 训练 200 步、1,600 个 prompt、12,800 次生成、"
     "32,093 个生成 token，墙钟 168.7 秒，为 SFT 的 3.44 倍。GRPO 的额外开销主要来自每组 8 个采样的自回归生成，"
     "以及每步的策略前向/反向和参考模型前向；本实验没有使用 GPU，也没有做推理引擎与训练引擎的异步解耦。")

table(
    ["指标", "SFT", "GRPO", "差异 / 说明"],
    [
        ["训练损失", "CE 1.61 → 0.233（epoch 均值）", "策略损失≈0；KL 0 → 0.0086", "目标不同，不可直接比较"],
        ["验证准确率", "79.0%（最佳）", "80.0%（最佳）；79.8%（最终）", "GRPO 最佳高 1.0pp"],
        ["测试准确率", "80.8%（±3.45pp）", "79.6%（±3.53pp）", "−1.2pp；McNemar p=0.114"],
        ["平均输出长度", "2.462 token", "2.458 token", "−0.004 token"],
        ["训练墙钟时间", "49.1 s", "168.7 s", "3.44×"],
    ],
    [Cm(3.0), Cm(5.2), Cm(5.2), Cm(4.6)],
    "表 1  主要结果",
    "注：准确率的括号为 95% 正态近似置信区间；测试集 n=500；SFT 与 GRPO 使用同一测试集，故同时报告配对 McNemar 检验。",
)

table(
    ["计算项", "SFT", "GRPO"],
    [
        ["模型参数", "600,576", "600,576"],
        ["优化步数", "790", "200"],
        ["训练样本 / 生成次数", "50,000 个样本", "1,600 个 prompt；12,800 次生成"],
        ["生成 token 数", "—", "32,093"],
        ["训练墙钟时间", "49.1 s", "168.7 s（3.44×）"],
        ["硬件", "CPU，14 线程，无 GPU", "CPU，14 线程，无 GPU"],
    ],
    [Cm(4.2), Cm(6.0), Cm(7.6)],
    "表 2  计算开销",
    "注：墙钟时间包含训练过程中的周期性验证；SFT 与 GRPO 在同一进程顺序运行，未做异步或 GPU 优化。",
)

# ---------------- 4 ----------------
h1("讨论", "4")
h2("4.1  为什么 GRPO 没有提升")
body("第一，SFT 已经达到 80.8%，剩余错误主要是需要精确进位的“难例”；在二值精确匹配奖励下，"
     "组内 8 个样本往往全部正确或全部错误，优势为 0 或方差很大，有效梯度信号有限。"
     "第二，模型只有 0.6M 参数，容量可能已接近该任务的上限；GRPO 只能在现有策略附近重新加权，"
     "很难“无中生有”地学出新的进位算法。第三，答案只有 2—3 个 token，没有思维链或草稿空间，"
     "GRPO 无法通过延长推理来换取准确率。第四，本实验只做了 200 步、每批一次 on-policy 更新，"
     "并保留 KL 锚点；探索预算和更新次数都远小于大模型 RL 后训练。第五，500 题测试集的 95% 置信区间约 ±3.5pp，"
     "1.2pp 的差异本身也在噪声范围内。")
h2("4.2  与 LLM 后训练实践的关系")
body("这与前文综述的判断一致：GRPO/RLVR 的收益依赖强基座、可验证且足够密的奖励、长 CoT 空间、"
     "大规模 rollout 和稳定的训练-推理一致性。DeepSeek-R1、Qwen3、DeepSeek-V3.2 等报告中的提升，"
     "建立在数百亿至数千亿参数、数万至数十万条可验证样本、以及大量 GPU 采样的基础上；"
     "把这些方法直接缩放到 0.6M 参数、5,000 题、短答案的沙盒里，收益消失是合理的。")
body("这并不意味着 GRPO 无效，而是说明它不是“替换 SFT 即涨点”的免费午餐。对小模型或低算力团队，"
     "更现实的顺序是：先把 SFT 做扎实，确认错误结构与奖励密度，再决定是否引入 RL；"
     "如果引入，应优先增加 rollout 多样性、使用部分信用/过程奖励、扩大模型或数据，而不是只增加训练步数。")
h2("4.3  局限")
body("本实验是单随机种子、单任务、单模型规模的小规模研究；没有做超参数搜索，也没有使用 GPU、"
     "分布式 rollout、奖励模型或课程学习。GRPO 实现是教学级的简化版本：序列级重要性比率、token 级 KL、"
     "固定 KL 系数，未实现 Klear、Dr. GRPO、DAPO 等改进。因此，结论应限定为“在本设置下没有可测收益”，"
     "不能外推为“GRPO 在任何任务上都不如 SFT”。后续工作应至少报告 3 个随机种子、更多任务类型、"
     "以及准确率—计算开销的 Pareto 曲线。")

# ---------------- 5 ----------------
h1("结论", "5")
body("在 0.6M 参数、5,000 条两位加法训练数据、二值精确匹配奖励的设置下，SFT 与 GRPO 的测试准确率为"
     "80.8% 与 79.6%，差异不显著（McNemar p=0.114）；输出长度几乎不变；GRPO 的训练墙钟时间为 SFT 的 3.44 倍。"
     "本实验支持一个审慎的结论：**GRPO 的收益是有条件的，而不是普适的**。在采用 GRPO 之前，"
     "应先确认 SFT 是否已接近任务上限、奖励是否足够密、模型是否有容量与推理空间利用探索信号；"
     "否则，增加的可能只是计算开销，而不是能力。")

# ---------------- references ----------------
h1("参考文献", "")
refs = [
    "[1] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. arXiv:2203.02155, 2022.",
    "[2] Rafailov R, Sharma A, Mitchell E, et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model[J]. arXiv:2305.18290, 2023.",
    "[3] Shao Z, Wang P, Wang Q, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models[J]. arXiv:2402.03300, 2024.",
    "[4] DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning[J]. arXiv:2501.12948, 2025.",
    "[5] Qwen Team. Qwen3 Technical Report[J]. arXiv:2505.09388, 2025.",
    "[6] DeepSeek-AI. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models[J]. arXiv:2512.02556, 2025.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(0.8)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    r = p.add_run(ref)
    set_run(r, size=7.7, color=DARK)

doc.core_properties.title = "SFT 与 GRPO 在可验证算术任务上的对比实验"
doc.core_properties.subject = "SFT vs GRPO 小规模训练实验"
doc.core_properties.author = "实验报告"
doc.core_properties.keywords = "SFT, GRPO, RLVR, 可验证奖励, 负结果"

doc.save(DOCX)
print(DOCX)
