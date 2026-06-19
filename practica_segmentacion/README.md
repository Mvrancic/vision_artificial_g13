# Segmentacion de celulas sanguineas con U-Net

Este proyecto implementa una segmentacion binaria de celulas sanguineas usando U-Net.

El archivo `dataset.zip` contiene imagenes JPG separadas por tipo celular:

- `basophil`
- `eosinophil`
- `erythroblast`
- `ig`
- `lymphocyte`
- `monocyte`
- `neutrophil`
- `platelet`

El dataset no trae mascaras de segmentacion. Por eso el pipeline genera pseudo-mascaras automaticamente con procesamiento clasico de imagenes y luego entrena U-Net para aprender la separacion `celula` vs `fondo`.

## Estructura

```text
practica_segmentacion/
├── dataset.zip
├── requirements.txt
├── README.md
├── src/
│   ├── data.py
│   ├── masks.py
│   ├── model.py
│   ├── prepare_dataset.py
│   ├── predict.py
│   └── train_unet.py
└── outputs/
```

## Instalacion

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparar imagenes y mascaras

Para una prueba rapida:

```bash
python src/prepare_dataset.py --max-per-class 150 --image-size 128
```

Para usar todo el dataset:

```bash
python src/prepare_dataset.py --image-size 128
```

Esto crea:

```text
data/processed/
├── images/train
├── images/val
├── images/test
├── masks/train
├── masks/val
└── masks/test
```

Tambien guarda `outputs/dataset_summary.json`.

## Entrenar U-Net

```bash
python src/train_unet.py --epochs 25 --batch-size 16 --image-size 128
```

Salidas principales:

- `outputs/unet_bloodcells.keras`
- `outputs/training_history.csv`
- `outputs/training_curves.png`

## Probar predicciones

```bash
python src/predict.py --input data/processed/images/test --limit 12
```

Las comparaciones quedan en:

```text
outputs/predictions/
```

## Nota metodologica

Como las mascaras son generadas automaticamente, las metricas miden cuanto la U-Net reproduce esas pseudo-etiquetas, no una anotacion medica manual. Para un trabajo practico esto permite completar el flujo de segmentacion con U-Net usando el dataset disponible; para uso real harian falta mascaras revisadas por expertos.
