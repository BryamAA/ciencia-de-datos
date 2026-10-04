# Datos

| Archivo | Descripción |
|---|---|
| `telco_limpio.csv` | Dataset completo tras la limpieza (7.043 filas × 20 columnas; `TotalCharges` numérico, sin `customerID`, `Churn` = 0/1). |
| `train.csv` | Partición de entrenamiento (5.634 filas × 20 columnas, incluye `Churn`). |
| `test.csv` | Partición de prueba (1.409 filas × 20 columnas, incluye `Churn`). |

Las particiones se generan con `train_test_split(test_size=0.2, random_state=42, stratify=y)`. Ambas conservan el 26,54 % de churn.

## Dataset original (no incluido)

**Telco Customer Churn**, IBM Sample Data Sets, publicado en Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

El archivo original (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) se descarga automáticamente desde el notebook con `kagglehub`, por lo que no se sube al repositorio. Los archivos de esta carpeta son derivados del original.

Fuente: IBM Watson Analytics / IBM

Dataset: Telco Customer Churn

Distribución utilizada: Kaggle, usuario blastchar

Licencia: Apache License 2.0
