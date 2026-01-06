# Quick Start Guide - TreeNet AI Pipeline v2

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies

```bash
cd /home/lukovic/codes/treenetai/pipeline_v2
pip install -r requirements.txt
```

### 2. Build Segments

Extract 30-day segments from raw data (takes ~10-30 minutes):

```bash
python 1_build_segments.py \\
    --year 2020 \\
    --test-ratio 0.2 \\
    --verbose
```

This creates:
- Training segments: `processed/model_data/train_*_segments_numpy.pkl`
- Test segments: `processed/model_data/test_*_segments_numpy.pkl`

### 3. Train Model

Train TCN with gap injection (takes ~30-60 minutes on GPU):

```bash
python 2_train_model.py \\
    --epochs 100 \\
    --batch-size 32 \\
    --max-gap-days 12 \\
    --experiment-name baseline \\
    --verbose
```

Results saved to: `experiments/<timestamp>_baseline/`

---

## 📊 Evaluate Model

```bash
python 3_evaluate.py \\
    --model-path experiments/<timestamp>_baseline/best_model.keras \\
    --gap-days 12
```

---

## 🔧 Common Tasks

### Change Segment Length

Edit the default in `src/config.py`:

```python
@dataclass
class SegmentConfig:
    segment_days: int = 60  # Changed from 30
    stride_days: int = 20   # Changed from 10
```

Or pass as argument:

```bash
python 1_build_segments.py --segment-days 60 --stride-days 20
```

### Train Without Gap Injection

```bash
python 2_train_model.py --no-gaps
```

### Use Different Model Size

```bash
# Larger model (more capacity)
python 2_train_model.py --n-blocks 6 --n-filters 128

# Smaller model (faster)
python 2_train_model.py --n-blocks 3 --n-filters 32
```

---

## 📁 Data Paths

Update paths in CLI arguments or `src/config.py`:

```python
@dataclass
class DataPaths:
    data_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/server_data')
    meteo_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/meteo_data')
    output_root: Path = Path('/storage/lukovic/Data/FORWARDS/treenet/processed')
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: metadata_all.pkl"

Check that `data_root` points to the correct directory containing:
- `metadata_all.pkl`
- `thermometer_l1/`
- `hygrometer_l1/`
- `dendrometer_l2/`
- `dendrometer_lm/`

### "Out of Memory" during training

Reduce batch size:
```bash
python 2_train_model.py --batch-size 16
```

### No complete sites found

Check metadata - sites need at least one of each sensor type:
- Air temperature
- Relative humidity
- Tree stem radius change

---

## 📈 Next Steps

1. **Visualize segments**: Implement visualization module
2. **Compare with raw data**: Run validation checks
3. **Try different hyperparameters**: Experiment with model size, gap lengths
4. **Extend to multiple years**: Run for years 2019, 2020, 2021

---

## 💡 Tips

- Start with a single year to test the pipeline
- Use `--verbose` flag to see detailed progress
- Check `training_history.csv` to monitor training
- TensorBoard logs are in `experiments/*/tensorboard/`

For full documentation, see [README.md](README.md).
