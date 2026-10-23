"""Train a multiclass linear model with sample-weighted Federated Averaging."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ClientData:
    name: str
    features: np.ndarray
    labels: list[str]
    train_indices: np.ndarray | None = None
    validation_indices: np.ndarray | None = None


def read_client(path: Path) -> tuple[ClientData, list[str]]:
    feature_rows: list[list[float]] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Client file {path.name} is missing a header row.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if any(not name for name in reader.fieldnames):
            raise ValueError(f"Client file {path.name} has an empty column name.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Client file {path.name} has duplicate column names.")
        if "label" not in reader.fieldnames:
            raise ValueError(f"Client file {path.name} must contain a label column.")
        columns = [name for name in reader.fieldnames if name != "label"]
        if not columns:
            raise ValueError(f"Client file {path.name} must contain at least one feature column.")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Client file {path.name}, row {row_number}: extra values beyond the header.")
            label = (row.get("label") or "").strip()
            if not label:
                raise ValueError(f"Client file {path.name}, row {row_number}: label is required for training.")
            values: list[float] = []
            for column in columns:
                try:
                    number = float(row.get(column, ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Client file {path.name}, row {row_number}: feature {column!r} must be numeric."
                    ) from exc
                if not np.isfinite(number):
                    raise ValueError(
                        f"Client file {path.name}, row {row_number}: feature {column!r} must be finite."
                    )
                values.append(number)
            feature_rows.append(values)
            labels.append(label)
    if not feature_rows:
        raise ValueError(f"Client file {path.name} contains no training rows.")
    features = np.asarray(feature_rows, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Client file {path.name} has feature values outside the supported range.")
    return ClientData(path.stem, features, labels), columns


def load_clients(directory: Path) -> tuple[list[ClientData], list[str], list[str]]:
    if not directory.is_dir():
        raise ValueError(f"Client data directory does not exist: {directory}")
    files = sorted(path for path in directory.glob("*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No client CSV files found in {directory}.")
    clients: list[ClientData] = []
    feature_columns: list[str] | None = None
    for path in files:
        client, columns = read_client(path)
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"Client file {path.name} has feature columns that differ from other clients.")
        clients.append(client)
    if feature_columns is None:
        raise ValueError(f"No client CSV files found in {directory}.")
    classes = sorted({label for client in clients for label in client.labels})
    if len(classes) < 2:
        raise ValueError("Federated classification needs at least two distinct labels across clients.")
    return clients, feature_columns, classes


def split_client(
    client: ClientData,
    class_indices: dict[str, int],
    validation_fraction: float,
    rng: np.random.Generator,
) -> None:
    encoded = np.asarray([class_indices[label] for label in client.labels], dtype=np.int64)
    training: list[int] = []
    validation: list[int] = []
    for label_index in np.unique(encoded):
        indices = np.flatnonzero(encoded == label_index)
        rng.shuffle(indices)
        if len(indices) < 2:
            training.extend(indices.tolist())
            continue
        holdout_count = max(1, int(round(len(indices) * validation_fraction)))
        holdout_count = min(holdout_count, len(indices) - 1)
        validation.extend(indices[:holdout_count].tolist())
        training.extend(indices[holdout_count:].tolist())
    rng.shuffle(training)
    rng.shuffle(validation)
    client.train_indices = np.asarray(training, dtype=np.int64)
    client.validation_indices = np.asarray(validation, dtype=np.int64)
    client.labels = [str(value) for value in encoded]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def local_update(
    global_weights: np.ndarray,
    global_bias: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weights = global_weights.copy()
    bias = global_bias.copy()
    for _epoch in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch_x = features[selected]
            batch_y = labels[selected]
            predictions = softmax(batch_x @ weights + bias)
            predictions[np.arange(len(batch_y)), batch_y] -= 1.0
            predictions /= len(batch_y)
            grad_weights = batch_x.T @ predictions + weight_decay * weights
            grad_bias = predictions.sum(axis=0)
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias
    return weights, bias


def evaluate(
    clients: list[ClientData],
    weights: np.ndarray,
    bias: np.ndarray,
    class_count: int,
) -> tuple[float, float, float, int]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for client in clients:
        indices = client.validation_indices
        if indices is None or not len(indices):
            continue
        features = client.features[indices]
        labels = np.asarray([int(client.labels[index]) for index in indices], dtype=np.int64)
        probabilities = softmax(features @ weights + bias)
        predictions.append(np.argmax(probabilities, axis=1))
        truths.append(labels)
        losses.append(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))
    if not truths:
        raise ValueError("No local validation rows were held out; add more labeled rows per client class.")
    predicted = np.concatenate(predictions)
    truth = np.concatenate(truths)
    f1_values: list[float] = []
    for label_index in range(class_count):
        tp = int(np.sum((truth == label_index) & (predicted == label_index)))
        fp = int(np.sum((truth != label_index) & (predicted == label_index)))
        fn = int(np.sum((truth == label_index) & (predicted != label_index)))
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    return (
        float(np.concatenate(losses).mean()),
        float(np.mean(predicted == truth)),
        float(np.mean(f1_values)),
        len(truth),
    )


def save_model(
    path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    classes: list[str],
    feature_columns: list[str],
) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError("Model output filename must end in .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=weights,
        bias=bias,
        classes=np.asarray(classes, dtype=np.str_),
        feature_columns=np.asarray(feature_columns, dtype=np.str_),
    )


def load_model(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Model file does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name].copy() for name in saved.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read model archive: {exc}") from exc


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive finite number.")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a finite non-negative number.")
    return parsed


def fraction(value: str) -> float:
    parsed = positive_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("Validation fraction must be below 1.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sample-weighted FedAvg over user-supplied client CSV shards.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and evaluate a federated linear classifier.")
    train.add_argument("--clients-dir", required=True, type=Path, help="Directory with one labeled CSV per client.")
    train.add_argument("--model-out", type=Path, default=Path("fedavg_model.npz"))
    train.add_argument("--rounds", type=positive_int, default=20)
    train.add_argument("--local-epochs", type=positive_int, default=3)
    train.add_argument("--batch-size", type=positive_int, default=32)
    train.add_argument("--learning-rate", type=positive_float, default=0.01)
    train.add_argument("--weight-decay", type=nonnegative_float, default=0.0001)
    train.add_argument("--validation-fraction", type=fraction, default=0.2)
    train.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser("predict", help="Apply a saved model to a labeled or unlabeled feature CSV.")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def read_prediction_file(path: Path, expected_columns: list[str]) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Input CSV does not exist: {path}")
    rows: list[list[float]] = []
    row_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Prediction CSV is missing a header row.")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        supplied_columns = [name for name in reader.fieldnames if name != "label"]
        if supplied_columns != expected_columns:
            raise ValueError("Prediction feature columns must match the model's training columns and order.")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Prediction CSV row {row_number} has extra values beyond the header.")
            values: list[float] = []
            for column in expected_columns:
                try:
                    value = float(row.get(column, ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Prediction CSV row {row_number}: {column!r} must be numeric.") from exc
                if not np.isfinite(value):
                    raise ValueError(f"Prediction CSV row {row_number}: {column!r} must be finite.")
                values.append(value)
            row_ids.append(str(row_number - 1))
            rows.append(values)
    if not rows:
        raise ValueError("Prediction CSV contains no rows.")
    return row_ids, np.asarray(rows, dtype=np.float32)


def run_train(args: argparse.Namespace) -> None:
    clients, feature_columns, classes = load_clients(args.clients_dir)
    class_indices = {label: index for index, label in enumerate(classes)}
    rng = np.random.default_rng(args.seed)
    for client in clients:
        split_client(client, class_indices, args.validation_fraction, rng)
    if not any(client.validation_indices is not None and len(client.validation_indices) for client in clients):
        raise ValueError("No local validation rows were held out; add more labeled rows per client class.")

    input_dim = clients[0].features.shape[1]
    weights = rng.normal(0, 0.01, (input_dim, len(classes))).astype(np.float32)
    bias = np.zeros(len(classes), dtype=np.float32)
    print(f"Training FedAvg with {len(clients)} local client shard(s) over {args.rounds} rounds.")
    for round_number in range(1, args.rounds + 1):
        weighted_weights = np.zeros_like(weights, dtype=np.float64)
        weighted_bias = np.zeros_like(bias, dtype=np.float64)
        total_samples = 0
        for client in clients:
            indices = client.train_indices
            if indices is None or not len(indices):
                continue
            encoded = np.asarray([int(client.labels[index]) for index in indices], dtype=np.int64)
            local_weights, local_bias = local_update(
                weights,
                bias,
                client.features[indices],
                encoded,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.weight_decay,
                rng,
            )
            sample_count = len(indices)
            weighted_weights += sample_count * local_weights
            weighted_bias += sample_count * local_bias
            total_samples += sample_count
        if not total_samples:
            raise ValueError("No client has training rows.")
        weights = (weighted_weights / total_samples).astype(np.float32)
        bias = (weighted_bias / total_samples).astype(np.float32)
        loss, accuracy, f1, validation_count = evaluate(clients, weights, bias, len(classes))
        print(
            f"Round {round_number:03d}: holdout_loss={loss:.6f} accuracy={accuracy:.4f} "
            f"macro_f1={f1:.4f} validation_rows={validation_count}"
        )
    save_model(args.model_out, weights, bias, classes, feature_columns)
    print(f"Saved model to {args.model_out}.")


def run_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    feature_columns = model["feature_columns"].astype(str).tolist()
    row_ids, features = read_prediction_file(args.input, feature_columns)
    weights = model["weights"]
    bias = model["bias"]
    classes = model["classes"].astype(str).tolist()
    if weights.shape != (features.shape[1], len(classes)) or bias.shape != (len(classes),):
        raise ValueError("Saved model dimensions do not match its feature and class metadata.")
    probabilities = softmax(features @ weights + bias)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("row", "predicted_label", "confidence"))
        for row_id, values in zip(row_ids, probabilities):
            class_index = int(np.argmax(values))
            writer.writerow((row_id, classes[class_index], f"{values[class_index]:.8f}"))
    print(f"Predicted {len(row_ids)} rows; wrote {args.output}.")


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "train":
            if args.model_out.suffix.lower() != ".npz":
                raise ValueError("Model output filename must end in .npz.")
            if args.clients_dir.resolve() == args.model_out.resolve():
                raise ValueError("Model output must not overwrite the client data directory.")
            run_train(args)
        else:
            if args.input.resolve() in {args.output.resolve(), args.model.resolve()}:
                raise ValueError("Prediction output must not overwrite the input CSV or model archive.")
            if args.output.resolve() == args.model.resolve():
                raise ValueError("Prediction output must not overwrite the model archive.")
            run_predict(args)
    except (OSError, ValueError, csv.Error, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
