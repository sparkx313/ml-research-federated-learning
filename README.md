# FedAvg Tabular Classifier

Train a multiclass linear classifier with sample-weighted Federated Averaging (FedAvg) over real, separately stored client CSV shards. Each client performs local mini-batch updates; the coordinator averages model parameters by the number of local training rows and evaluates on held-out rows from those shards.

The model learns from files you provide. It does not create random client updates or synthetic labels.

## Requirements

- Python 3.10 or newer
- NumPy

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Client data

Place one CSV file per client in a directory. Each file needs a `label` column plus the same numeric feature columns in the same order. Feature values must be finite and reasonably scaled for gradient-based training. Every training row needs a label; the CLI creates a deterministic local holdout split from each client's labeled rows.

```text
clients/
  client_01.csv
  client_02.csv
  client_03.csv
```

## Train

```bash
python main.py train \
  --clients-dir data/clients \
  --rounds 20 \
  --local-epochs 3 \
  --model-out artifacts/fedavg_model.npz
```

Each round reports holdout cross-entropy, accuracy, macro F1, and the validation row count. The saved model records its weights, classes, and feature-column order.

## Predict

Apply the model to a CSV with the same feature columns, optionally including a `label` column:

```bash
python main.py predict \
  --input new_client/unlabeled.csv \
  --model artifacts/fedavg_model.npz \
  --output reports/predictions.csv
```

The output contains each input row number, predicted class, and softmax confidence.

## Scope and privacy

- This is a FedAvg baseline for tabular classification. It uses one model shared across all client shards, full participation each round, local SGD, and sample-count-weighted parameter averaging.
- The implementation runs in one Python process and loads every client CSV into memory. It is useful for training and evaluating on partitioned datasets, but it does not provide separate client services or network transport.
- It does not implement differential privacy, secure aggregation, FedProx, Byzantine-resilient aggregation, or a formal privacy guarantee. No privacy budget or security claim is made.
- Evaluation holds out labels within the same client files. The reported metrics are dataset-specific and are not benchmark claims.
