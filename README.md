# CLIP on CIFAR-100 — Experimental Study

This repository documents experiments with **CLIP (Contrastive Language–Image Pre-training)** on the **CIFAR-100** image classification dataset, based on the paper *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021).

## Research goals

- Evaluate pretrained CLIP ViT-B/32 in zero-shot classification on CIFAR-100.
- Examine the impact of prompt engineering and prompt ensembling.
- Compare zero-shot predictions with a linear probe trained on CLIP image features.
- Analyze evaluation metrics and results reported by the project team.

> The project uses pretrained CLIP; it does **not** claim to retrain CLIP from scratch.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── 01_ghi_chu_so_lieu.md
│   ├── 02_noi_dung_slide_2_6.md
│   ├── 03_hoi_dap_va_demo.md
│   └── Ghi_chu_so_lieu_CLIP.docx
├── results/
│   ├── bieu_do_ket_qua.png
│   └── ket_qua_nhom.csv
├── notebooks/
│   └── README.md
└── paper/
    └── README.md
```

## Run the experiment

1. Install Python 3.9+ and Jupyter Notebook or use Google Colab.
2. Install packages: `pip install -r requirements.txt`.
3. Install the official CLIP implementation: `pip install git+https://github.com/openai/CLIP.git`.
4. Open the project notebook and run cells in order. Check data paths, runtime and download settings before execution.

**Notebook:** The provided submission notebook `CLIP_CIFAR100_BAN_NOP_DA_CHINH_LOGIC.ipynb` has not been uploaded to this repository yet. The notebook is kept unchanged. See [notebook information](notebooks/README.md).

## Results and supporting materials

- [Experimental results (CSV)](results/ket_qua_nhom.csv)
- [Results chart](results/bieu_do_ket_qua.png)
- [Data and number notes](docs/01_ghi_chu_so_lieu.md)
- [Slide content](docs/02_noi_dung_slide_2_6.md)
- [Q&A and demo notes](docs/03_hoi_dap_va_demo.md)
- [Word notes](docs/Ghi_chu_so_lieu_CLIP.docx)

Results should be interpreted according to the accompanying notes and the original experiment configuration; no new scores have been invented for this README.

## Reference

Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Official implementation: [openai/CLIP](https://github.com/openai/CLIP).

## Dataset

[CIFAR-100, University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) (100 fine-grained classes, 50,000 training and 10,000 test images).
