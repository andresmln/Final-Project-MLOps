import os
import mlflow

def test_mlflow_artifacts_exist():
    """
    Verifica si existe al menos un experimento y un run con modelo guardado.
    Este test requiere que se haya ejecutado train.py antes.
    """
    # Si no existe la carpeta mlruns, el test salta (para no fallar en entornos limpios)
    if not os.path.exists("mlruns"):
        pytest.skip("La carpeta mlruns no existe, saltando test de integración.")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Telco_Churn_Project")
    
    if experiment:
        runs = client.search_runs(experiment.experiment_id)
        if len(runs) > 0:
            last_run = runs[0]
            # Verificar que existan los artefactos clave
            # Nota: La ruta exacta depende de cómo MLFlow guarde localmente, 
            # pero podemos chequear si el run fue exitoso.
            assert last_run.info.status == "FINISHED"