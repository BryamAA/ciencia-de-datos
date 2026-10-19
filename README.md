# PyCaret (AutoML) vs. regresión logística: predicción de abandono de clientes (Telco Churn)

Trabajo grupal de **Ciencia de los Datos** sobre *herramientas basadas en IA para análisis de datos*. Se evalúa **PyCaret** (categoría AutoML, etapa de **modelado** del ciclo de vida de los datos) y se compara con una **regresión logística hecha a mano en scikit-learn**, con tres criterios medidos en el propio pipeline: F1-score, ROC-AUC y tiempo de cómputo.

Docente: Ing. Jorge Maldonado.

## Integrantes

| Integrante | Aportes |
|---|---|
| Bryam Astudillo | Sección 3 del notebook (PyCaret: script, tiempos, hiperparámetros y explicación de cada función) y ficha de investigación de PyCaret. |
| Marcos Naranjo | Sección 2 (EDA y preprocesamiento), comparación estadística final con intervalos de confianza, importancia de variables y versiones, casos de uso reales, presentación y README. |
| Daniel Salcedo | Sección 4 (regresión logística con validación cruzada), tabla comparativa de herramientas AutoML e interpretación de la comparación final. |

## Descripción del proyecto

- **Problema:** predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio (`Churn`).
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM Sample Data Sets, publicado en Kaggle): 7.043 clientes, 21 columnas originales y 26,54 % de churn.
- **Herramienta principal:** PyCaret 3.3.2 (`compare_models` y `tune_model`, optimizando F1).
- **Herramienta alternativa:** scikit-learn: `Pipeline` con `StandardScaler`, `OneHotEncoder` y `LogisticRegression`, en una versión base y otra con `GridSearchCV` (10 pliegues, 800 ajustes, optimizando F1).
- **Protocolo común:** el mismo split para ambas (`test_size=0.2`, `random_state=42`, `stratify=y`: 5.634 clientes de entrenamiento y 1.409 de prueba) y validación cruzada estratificada de 10 pliegues.

## Resultados (conjunto de prueba, 1.409 clientes)

| Modelo | F1-score | ROC-AUC | Precisión | Recall | Tiempo de cómputo |
|---|---|---|---|---|---|
| PyCaret (`LogisticRegression` ajustada) | 0,6164 | 0,8419 | 0,5069 | 0,7861 | 218,00 s (`setup` 4,29 + `compare_models` 103,35 + `tune_model` 110,35) |
| Regresión logística base (scikit-learn) | 0,6136 | 0,8416 | 0,5043 | 0,7834 | 0,11 s (un solo ajuste) |
| Regresión logística + `GridSearchCV` | 0,6130 | 0,8417 | 0,5034 | 0,7834 | 85,43 s (800 ajustes) |

Intervalos de confianza al 95 % (bootstrap de 1.000 remuestreos del test, con las mismas filas para los tres modelos): el F1 de PyCaret es 0,6164 [0,5792; 0,6501] y el de la regresión base, 0,6136 [0,5754; 0,6474].

**Lectura de los resultados:**

- **No hay diferencia demostrable de desempeño.** La diferencia PyCaret − regresión base es de +0,0027 en F1 (IC 95 %: −0,0012 a 0,0077) y de +0,0003 en ROC-AUC (−0,0003 a 0,0010). En las ocho diferencias calculadas el intervalo incluye el 0; en el F1 frente a `GridSearchCV` el límite inferior es 0,0000 al redondear. Eso no prueba que los modelos sean iguales, solo que este test no tiene evidencia para separarlos.
- **PyCaret comparó 14 modelos y eligió una regresión logística**, lo que indica que un modelo lineal es competitivo para este dataset.
- **Los tiempos no son equivalentes:** el de PyCaret incluye comparar 14 modelos y ajustar el mejor, mientras que el de scikit-learn es un solo ajuste o una búsqueda de 800 ajustes.
- **Variables asociadas al churn:** en ambos modelos, el contrato mes a mes y la fibra óptica se asocian con más abandono, y el contrato de dos años y DSL con menos. Es asociación, no causalidad.
- Los hiperparámetros elegidos fueron `C = 2,696` y `class_weight="balanced"` en PyCaret, y `C = 0,428`, `class_weight="balanced"` y `solver="liblinear"` en `GridSearchCV`.

La interpretación completa está en las secciones 3 a 5 del notebook.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt            # entorno principal del notebook
├── requirements-pycaret.txt    # entorno de PyCaret (Python 3.11)
├── notebooks/
│   └── 01_pycaret_vs_regresion_logistica.ipynb   # pipeline completo
├── data/
│   ├── README.md               # origen y descripción de los archivos
│   ├── telco_limpio.csv
│   ├── train.csv
│   └── test.csv
├── docs/
│   ├── investigacion_pycaret.pdf
│   ├── casos_de_uso_automl.pdf
│   └── comparacion_herramientas_automl.xlsx
└── material_educativo/
    ├── PyCaret_AutoML_Telco_Churn.pptx
    └── PyCaret_AutoML_Telco_Churn.pdf
```

## Instrucciones de ejecución

### Opción A: Google Colab (recomendada, es donde se desarrolló)

1. Abre el notebook en Colab: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BryamAA/ciencia-de-datos/blob/main/notebooks/01_pycaret_vs_regresion_logistica.ipynb)
2. Menú *Entorno de ejecución → Ejecutar todo*.
3. No requiere cuenta de Kaggle: el dataset se descarga con `kagglehub`.

En nuestra ejecución, PyCaret tardó 218 s y la búsqueda de la regresión logística 85 s; con la instalación y los gráficos, el notebook completo tarda varios minutos.

> **Por qué hay un entorno aparte para PyCaret:** la versión estable 3.3.2 solo admite Python 3.9 a 3.11, y Colab usa Python 3.13, donde `pip install pycaret` falla. El notebook crea con `uv` un entorno aislado con Python 3.11 en `/content/pc` y ejecuta allí los scripts de PyCaret con los mismos `train.csv` y `test.csv`.

### Opción B: ejecución local

El notebook está escrito para Colab y llama al Python del entorno de PyCaret con la ruta `/content/pc/bin/python`. Para ejecutarlo en local se necesitan **dos entornos separados**, porque sus versiones de `pandas`, `numpy` y `scikit-learn` son distintas y no se pueden instalar juntas.

```bash
git clone https://github.com/BryamAA/ciencia-de-datos.git
cd ciencia-de-datos

# Entorno principal (el que ejecuta el notebook)
python3 -m venv .venv
source .venv/bin/activate                  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Entorno de PyCaret (requiere Python 3.9 a 3.11; se probó con 3.11)
python3.11 -m venv .venv-pycaret
.venv-pycaret/bin/pip install -r requirements-pycaret.txt    # En Windows: .venv-pycaret\Scripts\pip

jupyter notebook notebooks/01_pycaret_vs_regresion_logistica.ipynb
```

Ajustes en el notebook para ejecutarlo en local:

1. **No ejecutes** la celda que instala `uv` y crea `/content/pc`.
2. En las tres celdas que ejecutan `!/content/pc/bin/python` (el script `pycaret_run.py`, el script `pycaret_importancia.py` y la celda de versiones), reemplaza esa ruta por la del Python de `.venv-pycaret` (`.venv-pycaret/bin/python`, o `.venv-pycaret\Scripts\python` en Windows).

Versiones con las que se ejecutó el notebook final:

| Entorno | Python | pandas | numpy | scikit-learn | PyCaret |
|---|---|---|---|---|---|
| Notebook (Colab) | 3.13.16 | 2.2.3 | 2.1.3 | 1.6.1 | no aplica |
| PyCaret | 3.11.17 | 2.1.4 | 1.26.4 | 1.4.2 | 3.3.2 |

## Material educativo y documentos

- [Presentación (PDF)](material_educativo/PyCaret_AutoML_Telco_Churn.pdf) y su versión editable [`.pptx`](material_educativo/PyCaret_AutoML_Telco_Churn.pptx): 19 diapositivas con qué es PyCaret, cuándo usarla, instalación, caso de uso con resultados propios, comparación, ventajas, limitaciones, caso real y referencias en APA 7.
- [Investigación de PyCaret](docs/investigacion_pycaret.pdf): ficha con funciones, ciclo de vida, ventajas, limitaciones y riesgos.
- [Casos de uso reales de AutoML](docs/casos_de_uso_automl.pdf): casos publicados con su fuente. En la presentación se usa el de predicción de diabetes tipo 2 con PyCaret.
- [Comparación de herramientas AutoML](docs/comparacion_herramientas_automl.xlsx): PyCaret, H2O AutoML, AutoGluon, TPOT y auto-sklearn, con una fuente por celda. Es una comparación bibliográfica: solo PyCaret y scikit-learn se midieron en este pipeline.

## Declaración de uso de IA generativa

**Herramientas utilizadas:** Claude (Anthropic) y Gemini (Google).

| Tarea | Herramienta | Uso |
|---|---|---|
| Planificación | Claude | Cronograma de trabajo, reparto de tareas entre los tres integrantes y redacción de los prompts que el grupo usó para cada tarea. |
| Código | Claude | Plantilla inicial del notebook (carga de datos, limpieza, gráficos del EDA, pipeline de scikit-learn, script de PyCaret), comparación estadística con bootstrap y solución del error de instalación de PyCaret en Python 3.13. |
| Código | Gemini | Tareas puntuales del notebook, entre ellas la búsqueda de hiperparámetros con validación cruzada de la regresión logística (sección 4), con prompts preparados con Claude. |
| Redacción | Claude y Gemini | Borradores de los textos explicativos del notebook, de las conclusiones del EDA, de la interpretación de la comparación y de este README. |
| Investigación | Claude y Gemini | Apoyo para estructurar y redactar la ficha de PyCaret, los casos de uso y la tabla comparativa de herramientas, con prompts preparados con Claude. |
| Diseño | Claude | Estructura y diseño de la presentación de 19 diapositivas, a partir de los resultados del notebook; los gráficos son los generados en el propio notebook. |

**Cómo verificamos las salidas:**

- **Código probado:** ejecutamos el notebook de principio a fin en Google Colab; el código generado por la IA se corrió y se corrigió hasta que funcionó. También comprobamos que el modelo de PyCaret usado para interpretar variables es el mismo de la comparación (mismos hiperparámetros y mismo F1 en el test).
- **Cifras reales:** todas las cifras del EDA y de la comparación (porcentajes de churn, medias, F1, ROC-AUC, intervalos de confianza y tiempos) provienen de las salidas del notebook, no de la IA. Las conclusiones se escribieron después de ver los resultados y se contrastaron con los gráficos.
- **Decisiones técnicas revisadas:** por ejemplo, imputar con 0 los 11 valores vacíos de `TotalCharges` solo después de comprobar que todos son de clientes con `tenure = 0`, y poner el preprocesamiento dentro de un `Pipeline` para evitar fuga de datos en la validación cruzada.
- **Afirmaciones contrastadas con fuentes:** los datos técnicos se compararon con la documentación oficial (licencia y versiones de Python de PyCaret en `pycaret.org` y en su repositorio de GitHub) y los del caso real, con el artículo original (Gul et al., 2024). Esa revisión corrigió salidas de la IA: la licencia de PyCaret 4.0 no es solo MIT, y una métrica del caso real se tomó de una tabla inadecuada.
- **Errores de la IA corregidos:** se eliminó una columna de "líneas de código" que no estaba medida y una razón de tiempos calculada a mano con un valor incorrecto; ahora se calcula en el código.
- **Datos:** se usó un dataset público y no se ingresaron datos personales ni sensibles en herramientas de IA.
- **Comprensión del código:** cada integrante revisó su sección y puede explicar cualquier fragmento del notebook.

## Desarrollo y colaboración

El análisis se desarrolló de forma colaborativa en Google Colab, repartido por secciones según la tabla de integrantes, y luego se consolidó en este repositorio. Por eso el historial de commits refleja la consolidación y los ajustes posteriores, no el trabajo previo en Colab. Desde la consolidación, cada cambio se integró mediante ramas y pull requests, con commits de cada integrante.

## Créditos y referencias

- **Dataset:** IBM Sample Data Sets, *Telco Customer Churn*, publicado en Kaggle por *blastchar*: https://www.kaggle.com/datasets/blastchar/telco-customer-churn. Un espejo del dataset en Hugging Face (`scikit-learn/churn-prediction`) lo distribuye con licencia CC BY 4.0. No se redistribuye el archivo original: el notebook lo descarga y en `data/` solo se incluyen los archivos derivados.
- **PyCaret:** Ali, M., y colaboradores. (2024). *pycaret: An open source, low-code machine learning library in Python* (Versión 3.3.2) [Software]. Licencia MIT. https://pypi.org/project/pycaret/
- **scikit-learn:** Pedregosa, F., y colaboradores. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://scikit-learn.org
- **Caso real:** Gul, S., Ayturan, K., y Hardalaç, F. (2024). PyCaret for predicting type 2 diabetes: A phenotype- and gender-based approach with the “Nurses’ Health Study” and the “Health Professionals’ Follow-Up Study” datasets. *Journal of Personalized Medicine, 14*(8), 804. https://doi.org/10.3390/jpm14080804
- La lista completa de referencias en APA 7 está en la última diapositiva de la presentación.