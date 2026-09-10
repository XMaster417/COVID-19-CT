# COVID-19-CT

Notebook para entrenar y evaluar un modelo U-Net con encoder EfficientNet-B2 que segmenta tomografías de tórax en cuatro clases. El proyecto descarga automáticamente los datos de la competencia **COVID Segmentation** de Kaggle, preprocesa las imágenes, entrena el modelo y genera el archivo de predicciones `sub.csv`.

## Requisitos

- Git.
- Python 3.13 (versión utilizada por el notebook).
- Una cuenta de [Kaggle](https://www.kaggle.com/).
- Haber iniciado sesión y aceptado las reglas de la competencia [COVID Segmentation](https://www.kaggle.com/competitions/covid-segmentation).
- Conexión a internet durante la primera ejecución para descargar el dataset y los pesos preentrenados.

El entrenamiento funciona con CPU o GPU. Una GPU compatible con CUDA reduce considerablemente el tiempo de ejecución.

## Instalación

1. Clona el repositorio y entra en su directorio:

   ```bash
   git clone https://github.com/XMaster417/COVID-19-CT.git
   cd COVID-19-CT
   ```

2. Crea un entorno virtual:

   ```bash
   python3 -m venv venv
   ```

3. Activa el entorno virtual.

   En macOS o Linux:

   ```bash
   source venv/bin/activate
   ```

   En Windows con PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Actualiza `pip` e instala las dependencias:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Configuración del `.env`

1. En Kaggle abre **Settings > API > API Tokens** y selecciona **Generate New Token**.
2. En la raíz del repositorio crea un archivo llamado `.env` con el token generado:

   ```dotenv
   KAGGLE_API_TOKEN=tu_token_de_kaggle
   ```

No agregues comillas ni espacios alrededor del token. El archivo `.env` está incluido en `.gitignore`, por lo que tus credenciales no se subirán al repositorio. Nunca compartas ni confirmes este archivo en Git.

## Ejecución

Con el entorno virtual activo y desde la raíz del repositorio, inicia Jupyter:

```bash
jupyter notebook main.ipynb
```

También puedes usar JupyterLab:

```bash
jupyter lab
```

Abre `main.ipynb`, selecciona el kernel del entorno virtual y ejecuta las celdas en orden. Al comenzar, el notebook:

1. Carga `KAGGLE_API_TOKEN` desde `.env`.
2. Descarga la competencia `covid-segmentation` mediante `kagglehub` o reutiliza su caché local.
3. Preprocesa los datos y entrena el modelo.
4. Guarda el modelo como `Unet-efficientnet.pt` y genera las predicciones en `sub.csv`.

## Problemas comunes

- **`No se encontró KAGGLE_API_TOKEN`**: confirma que el archivo se llama exactamente `.env`, está en la raíz del repositorio y contiene la variable indicada.
- **Error `401`, `403` o de autenticación de Kaggle**: genera un token nuevo, actualiza `.env` y verifica que aceptaste las reglas de la competencia.
- **Kernel incorrecto o módulos no encontrados**: activa nuevamente el entorno virtual antes de iniciar Jupyter y selecciona su kernel.
- **Memoria insuficiente**: reduce `batch_size` en el notebook. El entrenamiento y los arreglos del dataset pueden requerir varios GB de RAM o VRAM.

## Archivos principales

- `main.ipynb`: descarga de datos, preprocesamiento, entrenamiento, evaluación y predicción.
- `requirements.txt`: dependencias de Python.
- `.env`: credencial local de Kaggle; no está versionada.
- `sub.csv`: archivo de predicciones con formato para envío a Kaggle.

## Licencia

Este proyecto se distribuye bajo los términos de [LICENSE](LICENSE).
