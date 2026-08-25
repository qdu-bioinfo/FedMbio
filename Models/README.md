# FedMbio Trained Models

The pretrained FedMbio model files are distributed through **GitHub Releases** because many `.pt` files exceed GitHub's 100 MB size limit for regular Git objects.

The `Models/` directory in the main repository provides the expected directory structure only. The actual trained model files should be downloaded from the corresponding GitHub Releases and placed in the directories specified below.

## Model directory structure

After downloading all pretrained models, the directory structure should be:

```text
Models/
├── 16s/
│   ├── CTRADA/
│   │   ├── fold0_*.pt
│   │   ├── fold1_*.pt
│   │   ├── fold2_*.pt
│   │   ├── fold3_*.pt
│   │   └── fold4_*.pt
│   │
│   └── CTRCRC/
│       ├── fold0_*.pt
│       ├── fold1_*.pt
│       ├── fold2_*.pt
│       ├── fold3_*.pt
│       └── fold4_*.pt
│
└── wgs/
    ├── CTRADA/
    │   ├── fold0_*.pt
    │   ├── fold1_*.pt
    │   ├── fold2_*.pt
    │   ├── fold3_*.pt
    │   └── fold4_*.pt
    │
    └── CTRCRC/
        ├── fold0_*.pt
        ├── fold1_*.pt
        ├── fold2_*.pt
        ├── fold3_*.pt
        └── fold4_*.pt
```

The exact `.pt` filenames within each directory correspond to the trained client models, server models, prototypes, and other model objects generated for the five cross-validation folds.

## Available model releases

The pretrained models are organized into four GitHub Releases:

| Task        | Release tag         | Destination directory |
| ----------- | ------------------- | --------------------- |
| 16S CTR–ADA | `models-16s-CTRADA` | `Models/16s/CTRADA/`  |
| 16S CTR–CRC | `models-16s-CTRCRC` | `Models/16s/CTRCRC/`  |
| WGS CTR–ADA | `models-wgs-CTRADA` | `Models/wgs/CTRADA/`  |
| WGS CTR–CRC | `models-wgs-CTRCRC` | `Models/wgs/CTRCRC/`  |

Each Release contains the complete set of pretrained model files for the corresponding sequencing modality and classification task.

## Download pretrained models

### 1. Install GitHub CLI

GitHub CLI (`gh`) is recommended for downloading the Release assets directly into the required directories.

On Windows, GitHub CLI can be installed using:

```bash
winget install --id GitHub.cli
```

After installation, close and reopen the terminal, and verify the installation:

```bash
gh --version
```

### 2. Authenticate GitHub CLI

Run:

```bash
gh auth login
```

Select the following options when prompted:

```text
GitHub.com
HTTPS
Login with a web browser
```

Follow the instructions displayed in the terminal to complete authentication.

Verify the login status:

```bash
gh auth status
```

### 3. Clone the FedMbio repository

The repository can be cloned using GitHub CLI:

```bash
gh repo clone qdu-bioinfo/FedMbio
```

Then enter the repository directory:

```bash
cd FedMbio
```

The repository already contains the required model directory skeleton:

```text
Models/
├── 16s/
│   ├── CTRADA/
│   └── CTRCRC/
└── wgs/
    ├── CTRADA/
    └── CTRCRC/
```

### 4. Download the 16S CTR–ADA models

```bash
gh release download models-16s-CTRADA --repo qdu-bioinfo/FedMbio --dir Models/16s/CTRADA
```

The downloaded files will be placed in:

```text
Models/16s/CTRADA/
```

### 5. Download the 16S CTR–CRC models

```bash
gh release download models-16s-CTRCRC --repo qdu-bioinfo/FedMbio --dir Models/16s/CTRCRC
```

The downloaded files will be placed in:

```text
Models/16s/CTRCRC/
```

### 6. Download the WGS CTR–ADA models

```bash
gh release download models-wgs-CTRADA --repo qdu-bioinfo/FedMbio --dir Models/wgs/CTRADA
```

The downloaded files will be placed in:

```text
Models/wgs/CTRADA/
```

### 7. Download the WGS CTR–CRC models

```bash
gh release download models-wgs-CTRCRC --repo qdu-bioinfo/FedMbio --dir Models/wgs/CTRCRC
```

The downloaded files will be placed in:

```text
Models/wgs/CTRCRC/
```

## Download all pretrained models

To download the complete set of pretrained FedMbio models, run the following four commands from the root directory of the cloned FedMbio repository:

```bash
gh release download models-16s-CTRADA --repo qdu-bioinfo/FedMbio --dir Models/16s/CTRADA

gh release download models-16s-CTRCRC --repo qdu-bioinfo/FedMbio --dir Models/16s/CTRCRC

gh release download models-wgs-CTRADA --repo qdu-bioinfo/FedMbio --dir Models/wgs/CTRADA

gh release download models-wgs-CTRCRC --repo qdu-bioinfo/FedMbio --dir Models/wgs/CTRCRC
```

Because the model files are large, downloading all four Releases may take considerable time depending on the network connection.

## Verify the downloaded models

After downloading the models, confirm that the four directories contain the corresponding `.pt` files:

```text
Models/16s/CTRADA/
Models/16s/CTRCRC/
Models/wgs/CTRADA/
Models/wgs/CTRCRC/
```

For example, on Windows:

```bash
dir Models\16s\CTRADA
```

The directory should contain model files from the five cross-validation folds, such as:

```text
fold0_*.pt
fold1_*.pt
fold2_*.pt
fold3_*.pt
fold4_*.pt
```

The same organization applies to the other three model directories.

## Notes

* The `.pt` model files are intentionally excluded from the main Git repository because of their large file sizes.
* The trained models are instead distributed through GitHub Releases.
* The directory structure in the main repository is retained so that downloaded models can be placed directly into the paths expected by the FedMbio code.
* Please do not rename the downloaded `.pt` files or move them to different directories unless the corresponding paths in the FedMbio code are also updated.
* The repository `.gitignore` excludes `Models/**/*.pt`, so downloading the pretrained models into these directories will not add the large model files to normal Git tracking.
