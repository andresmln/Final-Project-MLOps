# Usamos una imagen ligera de Python 3.11
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalar herramientas básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de configuración de dependencias
COPY pyproject.toml .

# Instalar dependencias del sistema y del proyecto
# Usamos pip directamente leyendo el toml
RUN pip install --no-cache-dir .

# Copiar el código fuente
COPY mylib/ mylib/
COPY api/ api/
COPY data/ data/
# Copiamos mlruns temporalmente para que la API tenga algo que cargar 
# (En producción real usaríamos un bucket S3 o servidor remoto MLFlow)
COPY mlruns/ mlruns/

# Exponer el puerto de la API
EXPOSE 8000

# Variables de entorno por defecto
ENV MLFLOW_TRACKING_URI=mlruns

# Comando para arrancar la API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]