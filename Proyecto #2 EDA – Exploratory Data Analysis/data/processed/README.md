Datos procesados.

# Datos procesados

Este directorio contiene los datasets generados después del proceso de ingeniería de características.

- **Archivo principal:** `feature_engineered.csv`
- **Origen:** generado a partir de `train.csv` y `stores.csv`
- **Notebook:** `notebooks/02_feature_exploration.ipynb`
- **Descripción:**
  - Se aplicó imputación de variables numéricas y categóricas
  - Winsorización del 0.5%
  - Transformación Yeo-Johnson
  - Escalado estándar (`StandardScaler`)
- **Regeneración:**
  Si el archivo no existe, ejecutar nuevamente el notebook `02_feature_exploration.ipynb`, que lo generará automáticamente.
