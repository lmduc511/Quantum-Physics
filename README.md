# Quantum-Physics

## Bài toán nhiều electron trong phương trình Schrödinger

Tài liệu tự học (PDF tiếng Việt, 81 trang) kèm code Python chạy được. Tài liệu giải thích **vì sao** phương trình Schrödinger cho nhiều electron không giải được bằng công thức, và **các hướng giải quyết** từ cổ điển tới hiện đại. Ví dụ xuyên suốt là nguyên tử liti (Li, 3 electron).

📄 **Báo cáo:** [`report/bao_cao.pdf`](report/bao_cao.pdf)

Mỗi khái niệm được trình bày hai lớp: giải thích kiểu ELI5 (ví dụ đời thường, không công thức), rồi công thức thật có giải thích từng ký hiệu.

| Chương | Nội dung |
|---|---|
| 1 | Hamiltonian của Li, vì sao không tách biến được, "bức tường hàm mũ" |
| 2 | Hạt đồng nhất, nguyên lý Pauli, định thức Slater, nguyên lý biến phân, năng lượng tương quan, điều kiện cusp |
| 3 | Hartree–Fock: trường trung bình, vòng lặp SCF (tự viết), tập cơ sở, ROHF/UHF, Koopmans |
| 4 | Sau Hartree–Fock: CI, MP2, Coupled Cluster, Full CI, ngoại suy CBS, tương quan tường minh |
| 5 | Lý thuyết phiếm hàm mật độ (DFT): Hohenberg–Kohn, Kohn–Sham, bậc thang Jacob, lỗi tự tương tác |
| 6 | Monte Carlo lượng tử: VMC và DMC tự viết bằng numpy, bài toán dấu, nút cố định |
| 7 | Biên giới: hàm sóng mạng nơ-ron (FermiNet, PauliNet), máy tính lượng tử (VQE), DMRG |
| 8 | Tổng kết: mọi phương pháp trên cùng một biểu đồ |
| Phụ lục | Lộ trình tự học, tài liệu nên đọc (có link), bảng thuật ngữ Việt–Anh (90 thuật ngữ), hướng dẫn chạy code |

## Kết quả chính cho nguyên tử Li

Năng lượng chính xác (phi tương đối tính, Hylleraas): **−7.478 060 Ha**. Code trong kho tính lại:

| Phương pháp | Năng lượng (Ha) | Sai số so với đáp án |
|---|---|---|
| Hartree–Fock (giới hạn) | −7.43273 | +45.3 mHa (thiếu toàn bộ năng lượng tương quan) |
| MP2 / ngoại suy CBS | −7.47401 | +4.05 mHa |
| CCSD(T) / ngoại suy CBS | −7.47791 | +0.15 mHa |
| Full CI / cc-pCVQZ (292 824 định thức) | −7.47637 | +1.69 mHa |
| Full CI / ngoại suy CBS | −7.47794 | +0.12 mHa |
| DFT: LDA / PBE / B3LYP | −7.34392 / −7.46211 / −7.49291 | +134 / +16 / −15 mHa |
| VMC tự viết (Slater–Jastrow, 4 tham số) | −7.4744 ± 0.0002 | +3.6 mHa (92% năng lượng tương quan) |
| DMC tự viết (ngoại suy τ → 0) | −7.47787 ± 0.00007 | +0.19 mHa (99.6% năng lượng tương quan) |

Bảng đầy đủ (kèm DFT) nằm trong chương 8 của báo cáo.

## Cấu trúc thư mục

```
code/      script Python (01 → 06), chạy theo thứ tự
results/   kết quả số (JSON)
figures/   hình vẽ (PDF cho LaTeX, PNG để xem nhanh)
report/    mã nguồn LaTeX và file PDF của báo cáo
```

## Chạy lại

```bash
pip install -r code/requirements.txt
cd code
python 01_nen_tang.py
python 02_hartree_fock.py
python 03_sau_hartree_fock.py
python 04_dft.py
python 05_monte_carlo_luong_tu.py
python 06_tong_hop.py
```

Biên dịch báo cáo (cần XeLaTeX):

```bash
cd report
latexmk -xelatex bao_cao.tex
```

Chi tiết xem Phụ lục D của báo cáo.
