# Projet MLOps

## Lancer l'app via Docker

Assurez-vous d'avoir **Docker Desktop** installé sur votre machine.

 ```bash 
docker pull asobrino9789/projet-mlops:latest
docker run -p 8501:8501 asobrino9789/projet-mlops:latest
``` 

Puis ouvrir **http://localhost:8501** 

## Lancer le pipeline en local 

```bash 
python src/feature_engineering.py
python src/preprocessing.py
python src/train.py 
streamlit run app/app.py 
``` 

## MLflow UI 
```bash 
mlflow ui --backend-store-uri ./MLruns 
``` 

Puis ouvrir **http://127.0.0.1:5000**
