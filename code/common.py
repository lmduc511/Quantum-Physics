"""Hằng số, đường dẫn và kiểu vẽ dùng chung cho mọi script.

Tất cả năng lượng trong code đều tính bằng đơn vị nguyên tử Hartree (Ha).
1 Ha = 27.211386 eV.  1 mHa = 0.001 Ha.  "Độ chính xác hóa học" = 1 kcal/mol ~ 1.6 mHa.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không cần màn hình
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hằng số vật lý và giá trị tham chiếu cho nguyên tử Li
# (nguồn ghi trong báo cáo, mục "Giá trị tham chiếu")
# ---------------------------------------------------------------------------
HARTREE_TO_EV = 27.211386245981      # CODATA 2022
HARTREE_TO_KCAL = 627.5094740631     # CODATA 2018
CHEM_ACC_HA = 1.0 / HARTREE_TO_KCAL  # 1 kcal/mol ~ 1.594 mHa

# Năng lượng "chính xác" phi tương đối tính, hạt nhân khối lượng vô hạn (Hylleraas)
E_EXACT_LI = -7.478060323910
# Giới hạn Hartree-Fock (tập cơ sở vô hạn), Hartree-Fock hạn chế (ROHF)
E_HF_LIMIT_LI = -7.432726931
# Năng lượng ion hóa thứ nhất của Li đo thực nghiệm (NIST ASD)
IE_EXP_LI_EV = 5.391715
# Li+ (2 electron, giống He) - năng lượng chính xác phi tương đối tính
E_EXACT_LI_PLUS = -7.279913412669

# Heli: dùng cho ví dụ khởi động về nguyên lý biến phân
E_EXACT_HE = -2.903724377034

# ---------------------------------------------------------------------------
# Màu (bảng màu phân loại cố định, dùng theo thứ tự, không xoay vòng)
# ---------------------------------------------------------------------------
C = {
    "blue": "#257de0",
    "orange": "#f47230",
    "aqua": "#00b894",
    "yellow": "#f2a600",
    "magenta": "#ee7fac",
    "green": "#008a17",
    "violet": "#6a4ce2",
    "red": "#ec4647",
}
SERIES = list(C.values())
TEXT = "#191919"
TEXT2 = "#666666"
GRID = "#e4e4e4"
NEUTRAL = "#9c9c9c"


def setup_style() -> None:
    """Kiểu vẽ gọn: nét mảnh, lưới nhạt, phông hỗ trợ tiếng Việt."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "mathtext.fontset": "dejavusans",
        }
    )


def save_fig(fig, name: str) -> None:
    """Lưu hình dưới dạng PDF (cho LaTeX) và PNG (để xem trên GitHub)."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  -> đã lưu figures/{name}.pdf và .png")


def save_json(obj, name: str) -> None:
    with open(RES_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  -> đã lưu results/{name}.json")


def load_json(name: str):
    with open(RES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Đo thời gian chạy:  with Timer() as t: ...;  t.seconds"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
