from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data import make_dataset
from model import bce_dice_loss, binary_iou, build_unet, dice_coefficient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena U-Net para segmentacion binaria.")
    parser.add_argument("--data", default="data/processed", help="Directorio procesado.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def save_history(history: tf.keras.callbacks.History, outputs_dir: Path) -> None:
    keys = list(history.history.keys())
    rows = zip(*[history.history[key] for key in keys])
    with (outputs_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for epoch, values in enumerate(rows, start=1):
            writer.writerow([epoch, *values])


def save_curves(history: tf.keras.callbacks.History, outputs_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["dice_coefficient"], label="train")
    axes[1].plot(history.history["val_dice_coefficient"], label="val")
    axes[1].set_title("Dice")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outputs_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = (base_dir / args.data).resolve()
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(
        data_dir / "images" / "train",
        data_dir / "masks" / "train",
        args.image_size,
        args.batch_size,
        shuffle=True,
    )
    val_ds = make_dataset(
        data_dir / "images" / "val",
        data_dir / "masks" / "val",
        args.image_size,
        args.batch_size,
        shuffle=False,
    )

    model = build_unet(image_size=args.image_size, base_filters=args.base_filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=bce_dice_loss,
        metrics=["binary_accuracy", dice_coefficient, binary_iou],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            outputs_dir / "unet_bloodcells.keras",
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(outputs_dir / "unet_bloodcells_last.keras")
    save_history(history, outputs_dir)
    save_curves(history, outputs_dir)


if __name__ == "__main__":
    main()
