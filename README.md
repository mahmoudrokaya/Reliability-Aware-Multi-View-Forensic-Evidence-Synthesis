
# FERF Optimized v2

This package replaces both corrected scripts.

## Main improvements

1. Continuous console and file logging.
2. Time reported for every data part and evidence view.
3. XGBoost progress reported at the view level.
4. Exhaustive simplex search replaced by reproducible Dirichlet random search.
5. Threshold evaluation is vectorized.
6. Search reports elapsed time and estimated remaining time.
7. Equal weighting and every single-view solution are always included as control candidates.

## Files

- `ferf_common.py`
- `06_experiment_02_2_multiview_evidence_v2.py`
- `07_experiment_03_ferf_optimized_v2.py`

Place all three files in:

`D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments\Code`

## Run CSE-CIC-IDS2018

First generate the evidence-view outputs:

```powershell
python 06_experiment_02_2_multiview_evidence_v2.py `
  --dataset CSE-CIC-IDS2018 `
  --max-rows 1000000 `
  --seed 42 `
  --estimators 250
```

Then run optimized FERF:

```powershell
python 07_experiment_03_ferf_optimized_v2.py `
  --dataset CSE-CIC-IDS2018 `
  --weight-candidates 1500 `
  --seed 42 `
  --progress-every 100
```

## Faster diagnostic run

```powershell
python 06_experiment_02_2_multiview_evidence_v2.py `
  --dataset CSE-CIC-IDS2018 `
  --max-rows 300000 `
  --seed 42 `
  --estimators 150

python 07_experiment_03_ferf_optimized_v2.py `
  --dataset CSE-CIC-IDS2018 `
  --weight-candidates 500 `
  --seed 42
```

## Expected runtime

On a typical 8–16 core workstation, the one-million-row run should usually finish in approximately:

- evidence-view training: 20–90 minutes;
- FERF optimization: 2–15 minutes.

The exact duration depends mainly on storage speed, memory, CPU, and categorical feature cardinality.
