# FedMbio Trained Models

The trained FedMbio model files are distributed through **GitHub Releases** because many `.pt` files exceed GitHub's 100 MB size limit for regular Git objects.

The directories in this folder indicate the expected locations of the downloaded model files.

## Directory structure

After downloading all trained models, the directory structure should be:

```text
Models/
├── 16s/
│   ├── CTRADA/
│   │   ├── fold0_*.pt
│   │   ├── fold1_*.pt
│   │   ├── fold2_*.pt
│   │   ├── fold3_*.pt
│   │   └── fold4_*.pt
│   └── CTRCRC/
│       ├── fold0_*.pt
│       ├── fold1_*.pt
│       ├── fold2_*.pt
│       ├── fold3_*.pt
│       └── fold4_*.pt
└── wgs/
    ├── CTRADA/
    │   ├── fold0_*.pt
    │   ├── fold1_*.pt
    │   ├── fold2_*.pt
    │   ├── fold3_*.pt
    │   └── fold4_*.pt
    └── CTRCRC/
        ├── fold0_*.pt
        ├── fold1_*.pt
        ├── fold2_*.pt
        ├── fold3_*.pt
        └── fold4_*.pt