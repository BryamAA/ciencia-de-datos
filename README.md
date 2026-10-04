# PyCaret (AutoML) vs. Regresión Logística: predicción de abandono de clientes (Telco Churn)

Trabajo grupal de **Ciencia de Datos** (9.º semestre) sobre *herramientas basadas en IA para análisis de datos*.
Se evalúa **PyCaret** (categoría AutoML, etapa de **modelado** del ciclo de vida de ciencia de datos) y se compara con una **regresión logística implementada manualmente en scikit-learn**, usando F1-score, ROC-AUC y tiempo de entrenamiento.

## Integrantes

| Integrante | Responsabilidad |
|---|---|
| Bryam Astudillo | Implementación de PyCaret y coordinación del pipeline |
| Marcos Naranjo | Análisis exploratorio (EDA) y preprocesamiento |
| Daniel Salcedo | Regresión logística con scikit-learn y comparación de métricas |

## Descripción del proyecto

- **Problema:** predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio (`Churn`).
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data Sets, vía Kaggle): 7.043 clientes, 21 columnas originales, 26,54 % de churn.
- **Herramienta principal:** PyCaret (`compare_models` + `tune_model`, optimizando F1).
- **Herramienta alternativa:** scikit-learn (`Pipeline` con `StandardScaler`, `OneHotEncoder` y `LogisticRegression(class_weight="balanced")`).
- **Protocolo común:** mismo split para ambas (`test_size=0.2`, `random_state=42`, `stratify=y`).

## Resultados (conjunto de prueba, 1.409 clientes)

| Modelo | F1-score | ROC-AUC | Tiempo de entrenamiento |
|---|---|---|---|
| PyCaret (mejor modelo: `LogisticRegression` ajustada) | 0,6164 | 0,8419 | 122,44 s (`setup` + `compare_models` + `tune_model`) |
| Regresión logística (scikit-learn) | 0,6136 | 0,8416 | 0,07 s (un solo `fit`) |

**Lectura de los resultados:** la diferencia de desempeño es prácticamente nula (0,0028 en F1 y 0,0003 en ROC-AUC) y, con un único split de prueba, no permite afirmar que una opción sea mejor. PyCaret comparó 14 modelos y seleccionó una regresión logística, lo que confirma que un modelo lineal simple es competitivo para este dataset. El tiempo de PyCaret incluye la búsqueda de modelos e hiperparámetros, mientras que el de scikit-learn mide un solo ajuste, por lo que no son directamente equivalentes. El detalle y la interpretación completa están en la sección 5 del notebook.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt            # entorno principal (EDA + scikit-learn)
├── requirements-pycaret.txt    # entorno de PyCaret (Python 3.9 - 3.11)
├── notebooks/
│   └── 01_pycaret_vs_regresion_logistica.ipynb   # pipeline completo
└── data/
    ├── README.md               # origen, licencia y descripción de los archivos
    ├── telco_limpio.csv
    ├── train.csv
    └── test.csv
```

## Instrucciones de ejecución

### Opción A: Google Colab (recomendada, es donde se desarrolló)

1. Abre el notebook en Colab: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BryamAA/ciencia-de-datos/blob/main/notebooks/01_pycaret_vs_regresion_logistica.ipynb)
2. Menú *Entorno de ejecución → Ejecutar todo*. Tarda unos minutos (solo el cómputo de PyCaret fue de ~2 minutos, más la instalación).
3. No requiere cuenta de Kaggle: el dataset se descarga con `kagglehub`.

> **Nota sobre PyCaret:** PyCaret requiere Python ≤ 3.11, pero Colab usa Python 3.13, por lo que la instalación directa falla. El notebook crea un entorno aislado con Python 3.11 usando `uv` y ejecuta allí un script (`pycaret_run.py`) con los mismos `train.csv` y `test.csv`.

### Opción B: entorno local

Requiere **dos entornos** por la incompatibilidad de versiones, o uno solo con Python 3.11:

```bash
git clone https://github.com/<USUARIO>/<REPOSITORIO>.git
cd <REPOSITORIO>

# Un solo entorno con Python 3.11 (más simple)
python3.11 -m venv .venv
source .venv/bin/activate            # En Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-pycaret.txt

jupyter notebook notebooks/01_pycaret_vs_regresion_logistica.ipynb
```

Al ejecutar el notebook en local, **omite la celda de instalación con `uv`** (la que crea `/content/pc`); la celda que ejecuta el script detecta que no estás en Colab y usa el Python del entorno actual. Si tienes Kaggle configurado puedes cambiar la descarga, pero `kagglehub` funciona sin credenciales para este dataset.

## Declaración de uso de IA generativa

**Herramienta utilizada:** Claude (Anthropic) [COMPLETAR: añadir cualquier otra herramienta que el grupo haya usado, con su versión si la conocen].

| Tarea | Uso de la IA | Quién lo usó |
|---|---|---|
| Planificación | Cronograma de trabajo y reparto de tareas entre los tres integrantes. | [Marcos] |
| Código | Plantilla inicial del notebook (carga de datos, limpieza, gráficos del EDA, pipeline de scikit-learn, script de PyCaret) y solución del error de instalación de PyCaret en Python 3.13. | [Todos] |
| Redacción | Borradores de las conclusiones del EDA, de la interpretación de la comparación y de este README. | [Marcos] |
| Diseño | [COMPLETAR: por ejemplo, presentación o infografía, cuando estén hechas]. | [Bryam] |

**Cómo verificamos las salidas:**

- Ejecutamos todo el notebook de principio a fin en Google Colab; el código generado por la IA se corrió y se corrigió hasta que funcionó.
- Las cifras del EDA y de la comparación (porcentajes de churn, medias, F1, ROC-AUC, tiempos) **provienen de las salidas reales del notebook**, no de la IA. Las conclusiones se escribieron después de ver los resultados y se contrastaron con los gráficos.
- Revisamos las decisiones técnicas (por ejemplo, imputar con 0 los 11 valores vacíos de `TotalCharges` tras comprobar que todos tienen `tenure = 0`).
- Las referencias y el caso de uso real se comprueban abriendo y leyendo la fuente original. [COMPLETAR cuando estén listos.]
- Cada integrante revisó y entiende el código de su sección. [COMPLETAR/CONFIRMAR]

## Desarrollo y colaboración

El análisis se desarrolló de forma colaborativa en Google Colab, repartido por secciones según la tabla de integrantes, y luego se consolidó en este repositorio. Por eso el historial de commits refleja la consolidación y los ajustes posteriores, no el trabajo previo en Colab. Las contribuciones de cada integrante están indicadas en las secciones del notebook.

## Créditos y referencias

- **Dataset:** IBM Sample Data Sets, *Telco Customer Churn*, publicado en Kaggle por *blastchar*: https://www.kaggle.com/datasets/blastchar/telco-customer-churn.
- **PyCaret:** Ali, M. *PyCaret: An open source, low-code machine learning library in Python*. https://pycaret.org
- **scikit-learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011. https://scikit-learn.org
- Trabajo realizado para la asignatura de Ciencia de Datos. Docente: Ing. Jorge Maldonado.
