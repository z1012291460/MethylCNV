# MethylCNV

MethylCNV is a deep learning tool for detecting copy number variations (CNVs) from whole-genome bisulfite sequencing (WGBS) data. It uses a bidirectional LSTM (BiLSTM) model with attention mechanisms to integrate multiple genomic features including read depth, GC content, and methylation penalties to identify deletions (DEL) and duplications (DUP) in genomes.

## Installation

### Prerequisites
- Python 3.8
- Reference genome in FASTA format
- Aligned WGBS BAM files
- Methylation data in BED format

### Dependencies Installation

1. Clone the repository:
```bash
git clone https://github.com/z1012291460/MethylCNV.git
cd MethylCNV
```

2. Create a virtual environment (recommended):
```bash
python -m venv methylcnv_env
source methylcnv_env/bin/activate  # On Windows: methylcnv_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required packages:
```
torch>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
pysam>=0.19.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## Quick Start

### 1. Data Preparation

Prepare the following input files:

- **BAM file**: Aligned WGBS reads (e.g., using BWA-meth)
- **Methylation file**: CpG methylation data in BED format with columns:
  ```
  Chr  Start  End  MethPercent  MethCount  UnmethCount
  ```
- **CNV annotation**: Training CNV regions in BED format:
  ```
  Chr  Start  End  Type
  ```
  Where Type is either "DEL" or "DUP"
- **Reference genome**: FASTA file for the reference genome

### 2. Feature Extraction and Dataset Building

Use the feature collection pipeline to extract features from genomic regions:

```bash
python Model/FeatureCollection_pipeline.py
```

**Note**: Modify the file paths in the `main()` function of `FeatureCollection_pipeline.py`:

```python
def main():
    # Update these paths to your data
    reference_file = '/path/to/reference.fa'
    bam_file = '/path/to/sample.bam'
    meth_file = '/path/to/methylation.bed'
    cnv_bed_file = '/path/to/cnv_annotations.bed'
    
    # Build dataset
    X, y = builder.build_dataset(
        bam_file=bam_file,
        meth_file=meth_file,
        cnv_bed_file=cnv_bed_file,
        reference_file=reference_file,
        chrom_lengths_dict=chrom_lengths_dict
    )
```

### 3. Model Training

Train the BiLSTM model using the extracted features:

```bash
python Model/ModelTrain.py \
    --data_path train_data_human_bin100_rep2_14features.npz \
    --epochs 100 \
    --batch_size 192 \
    --lr 0.0001 \
    --seq_length 50 \
    --save_root ./model_output
```

### 4. Key Parameters

#### Data Pipeline Parameters:
- `--bin_size`: Genomic bin size (default: 100bp)
- `--context_size`: Context region around CNVs (default: 10kb)
- `--normal_keep_fraction`: Fraction of normal regions to retain (default: 0.0)

#### Model Parameters:
- `--lr`: Learning rate (default: 0.0001)
- `--seq_length`: Sequence length for LSTM (default: 50)
- `--bin_rnn_size`: LSTM hidden size (default: 128)
- `--dropout`: Dropout rate (default: 0.5)
- `--bidirectional`: Use bidirectional LSTM (default: True)

## Architecture

### Model Components

1. **Feature Encoders**: Individual LSTM encoders for each genomic feature
2. **Attention Mechanisms**: Feature-level and bin-level attention for importance weighting
3. **Feature Fusion**: LSTM-based integration of multi-modal features
4. **Classifier**: Multi-layer neural network for 3-class classification (Normal/DEL/DUP)

### Key Features Extracted

- **Read Depth Features**: Coverage depth, high-quality mapping depth
- **GC Content**: Local GC composition
- **Methylation Features**: 
  - Methylation penalty scores based on bisulfite conversion
  - CpG density and variance
  - Methylation entropy
- **Context Features**:
  - Depth gradients and smoothing
  - Local variation patterns
  - Depth-methylation coherence scores

## Output

The training process generates:

- **Model checkpoint**: Best performing model (`CNV_BiLSTM_best.pth`)
- **Test results**: Predictions and attention weights (`CNV_BiLSTM_test_results.npz`)
- **Visualization**: 
  - Confusion matrix (`CNV_BiLSTM_confusion_matrix.png`)
  - Feature importance plot (`CNV_BiLSTM_feature_importance.png`)
- **Metrics**: Detailed per-class precision, recall, and F1 scores

## Advanced Usage

### Custom Feature Extraction

Use the `DepthCalculator` class independently for custom feature extraction:

```python
from Model.Depth_Methy_Features import DepthCalculator, MethylationProcessor

# Initialize methylation processor
meth_processor = MethylationProcessor(
    meth_file='path/to/methylation.bed',
    reference_filename='path/to/reference.fa',
    bin_size=100
)

# Calculate features for a specific region
calc = DepthCalculator(
    filename='path/to/sample.bam',
    reference_filename='path/to/reference.fa',
    bin_size=100,
    region=(start_pos, end_pos),
    methylation_processor=meth_processor
)

features = calc.process_region(chromosome)
```

### Custom Training Parameters

Fine-tune the model with custom hyperparameters:

```bash
python Model/ModelTrain.py \
    --lr 0.0005 \
    --batch_size 256 \
    --seq_length 100 \
    --bin_rnn_size 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --epochs 150
```
