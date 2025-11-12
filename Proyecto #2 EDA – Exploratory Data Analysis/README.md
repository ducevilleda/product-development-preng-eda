
# Proyecto 2 – EDA

Dataset: Store Sales – Time Series Forecasting (Kaggle).

## Pasos
1) Entorno
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) Jupyter
```
jupyter notebook
```
3) Git
```
git init
git add .
git commit -m "Proyecto 2: estructura EDA y dataset"
git branch -M main
git remote add origin <URL_DE_TU_REPO>
git push -u origin main
```

# Proyecto 3 - Feature Engineering

**IMPORTANTE:**
Esta fase se desarrolló en branch `daniel-tarea3`.
Ejecutar `git checkout daniel-tarea3` antes de correr los notebooks o modificar los datos procesados.

## Datos procesados
* Los datasets generados tras el proceso de ingeniería de características se guardan en `data/processed/`.

* El archivo principal es `feature_engineered.csv`, producido automáticamente por el notebook  
`notebooks/02_feature_exploration.ipynb`.

Si no existe, el notebook lo regenerará aplicando imputaciones, winsorización, transformación Yeo-Johnson y escalado estándar.
