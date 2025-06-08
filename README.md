# SSCBDTA

**SSCBDTA: Prediction of Drug-Target Binding Affinity with Secondary Sequences and Multiple Cross-Attention Blocks**

![GitHub stars](https://img.shields.io/github/stars/4399chengzi/SSCBDTA?style=social)
![License](https://img.shields.io/github/license/4399chengzi/SSCBDTA)

---

## 🔍 Overview

SSCBDTA is a deep learning model for predicting drug-target binding affinity (DTA), which integrates secondary sequences (generated via Byte Pair Encoding) and multiple attention blocks (cross-attention and criss-cross attention).

This repository provides the implementation, data, and scripts for reproducing our results as described in the paper:

> Zuo HW, Zhou PC, Li X, Zhang L*. SSCBDTA: Prediction of Drug-Target Binding Affinity with Secondary Sequences and Multiple Cross-Attention Blocks.

---

## 📌 Key Features

- Utilizes **Byte Pair Encoding (BPE)** to generate secondary sequences for drugs and proteins.
- Incorporates **cross-attention** and **criss-cross attention** mechanisms to enhance feature fusion.
- Achieves **state-of-the-art performance** on Davis and KIBA datasets.
- Demonstrates potential in **anti-COVID drug screening** based on binding predictions with SARS-CoV-2 proteins.

---

## 📁 Folder Structure

```
SSCBDTA/
├── dataset/                 # Benchmark datasets: Davis, KIBA
├── model/                   # Model definitions
├── utils/                   # Preprocessing, evaluation functions
├── train.py                 # Training script
├── test.py                  # Testing script
├── README.md                # This file
└── requirements.txt         # Required packages
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/4399chengzi/SSCBDTA.git
cd SSCBDTA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run training

```bash
python train.py
```

---

## 📊 Results

| Dataset | CI (↑) | MSE (↓) | r<sub>m</sub><sup>2</sup> (↑) |
|---------|--------|---------|------------------------------|
| Davis   | 0.903  | 0.213   | 0.694                        |
| KIBA    | 0.896  | 0.143   | 0.775                        |

For ablation and case study results, please refer to the full manuscript.

---

## 📌 Citation

If you find this work helpful, please cite:

```bibtex
@article{SSCBDTA2024,
  title={SSCBDTA: Prediction of Drug-Target Binding Affinity with Secondary Sequences and Multiple Cross-Attention Blocks},
  author={Zuo, Hai-Wei and Zhou, Peng-Cheng and Li, Xia and Zhang, Li},
  journal={Interdisciplinary Sciences: Computational Life Sciences},
  year={2024}
}
```

---

## 📬 Contact

For questions or collaboration, feel free to reach out:

- Hai-Wei Zuo: zuohw@xzhmu.edu.cn  
- Peng-Cheng Zhou: 981691590@qq.com  
- Li Zhang (corresponding author): l-z@mail.tsinghua.edu.cn

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
