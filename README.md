# COVID-19-CT

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XMaster417/COVID-19-CT/blob/main/main.ipynb)

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

## Ejecución en Google Colab con GPU

El notebook incluye una celda de configuración que detecta automáticamente si se está ejecutando en Google Colab. En ese entorno:

- Instala solamente las dependencias que no estén disponibles.
- Obtiene `KAGGLE_API_TOKEN` desde los secretos de Colab o, si no existe, lo solicita mediante una entrada oculta.
- Descarga el dataset mediante `kagglehub` y guarda los archivos en la caché temporal de Colab.
- Selecciona CUDA automáticamente cuando el entorno tiene una GPU activa.

No es necesario clonar el repositorio, crear un entorno virtual, subir los archivos `.npy` ni montar Google Drive para ejecutar el entrenamiento.

### 1. Preparar Kaggle

1. Inicia sesión en [Kaggle](https://www.kaggle.com/).
2. Abre la competencia [COVID Segmentation](https://www.kaggle.com/competitions/covid-segmentation) y acepta sus reglas. Aunque el dataset sea público, Kaggle no permitirá descargarlo hasta aceptar las condiciones.
3. En [Kaggle Settings](https://www.kaggle.com/settings), busca **API > API Tokens** y selecciona **Generate New Token**.
4. Copia el token generado. No lo escribas directamente dentro del notebook ni lo compartas en Git.

### 2. Abrir el notebook en Colab

Utiliza el botón **Abrir en Google Colab** ubicado al inicio de este README. También puedes hacerlo manualmente:

1. Descarga `main.ipynb` desde este repositorio.
2. Abre [Google Colab](https://colab.research.google.com/).
3. Selecciona **File > Upload notebook** y carga `main.ipynb`.

### 3. Configurar el token como secreto

1. En la barra lateral izquierda de Colab, abre la sección **Secrets**, identificada con el icono de una llave.
2. Crea un secreto con el nombre exacto `KAGGLE_API_TOKEN`.
3. Pega el token de Kaggle como valor.
4. Activa la opción que permite al notebook acceder al secreto.

`kagglehub` admite oficialmente `KAGGLE_API_TOKEN` como secreto de Google Colab. Si se omite este paso, la primera celda solicitará el token de forma oculta durante la ejecución.

### 4. Activar la GPU

1. En Colab abre **Runtime > Change runtime type** o **Entorno de ejecución > Cambiar tipo de entorno de ejecución**.
2. En **Hardware accelerator**, selecciona **GPU**.
3. Selecciona una GPU disponible y guarda la configuración. El modelo no requiere TPU.

La disponibilidad y el tipo de GPU dependen del plan, la cuota y la capacidad disponible de Colab. Iniciar un entorno con GPU no basta por sí solo: el notebook debe mostrar `Using device: cuda` en la celda de importaciones. Si muestra `Using device: cpu`, revisa el tipo de entorno y vuelve a conectar la sesión.

### 5. Ejecutar el proyecto

Selecciona **Runtime > Run all** o **Entorno de ejecución > Ejecutar todas**. El flujo realizará automáticamente lo siguiente:

1. Detectará que se está ejecutando en Colab.
2. Instalará las dependencias faltantes.
3. Cargará el token sin imprimirlo.
4. Descargará o reutilizará el dataset de Kaggle.
5. Preprocesará las tomografías.
6. Entrenará y evaluará el modelo utilizando la GPU.
7. Generará los checkpoints y los archivos `sub.csv` y `sub_improved.csv`.

La descarga inicial del dataset y de los pesos preentrenados requiere conexión a internet. El entrenamiento puede tardar dependiendo de la GPU asignada.

### 6. Descargar los resultados

Los archivos almacenados en una sesión de Colab son temporales y se eliminan al cerrar o restablecer el entorno. Al finalizar:

1. Abre **Files** en la barra lateral de Colab.
2. Localiza el checkpoint y el archivo de entrega que quieras conservar.
3. Utiliza **Download** para guardarlos en tu equipo.

Los principales resultados son:

- `Unet-efficientnet.pt`: modelo del experimento original.
- `Unet-efficientnet-improved.pt`: pesos del mejor modelo según el mIoU de las lesiones.
- `sub.csv`: predicciones del experimento original.
- `sub_improved.csv`: predicciones de la mejora con pérdida ponderada, Dice y *test-time augmentation*.

## Problemas comunes

- **`No se encontró KAGGLE_API_TOKEN`**: confirma que el archivo se llama exactamente `.env`, está en la raíz del repositorio y contiene la variable indicada.
- **Colab no encuentra `KAGGLE_API_TOKEN`**: confirma que el secreto tiene exactamente ese nombre, que habilitaste su acceso al notebook y que no contiene espacios adicionales. Como alternativa, vuelve a ejecutar la primera celda e introduce el token cuando se solicite.
- **Error `401`, `403` o de autenticación de Kaggle**: genera un token nuevo, actualiza `.env` y verifica que aceptaste las reglas de la competencia.
- **Colab muestra `Using device: cpu`**: activa una GPU desde **Runtime > Change runtime type**, guarda la configuración y vuelve a conectar o reiniciar el entorno.
- **La sesión de Colab se desconectó**: los archivos temporales y el estado del entrenamiento pueden perderse. Descarga los checkpoints al terminar cada experimento importante.
- **Kernel incorrecto o módulos no encontrados**: activa nuevamente el entorno virtual antes de iniciar Jupyter y selecciona su kernel.
- **Memoria insuficiente**: reduce `batch_size` en el notebook. El entrenamiento y los arreglos del dataset pueden requerir varios GB de RAM o VRAM.

## Archivos principales

- `main.ipynb`: configuración local/Colab, descarga de datos, preprocesamiento, entrenamiento, evaluación y predicción.
- `requirements.txt`: dependencias de Python.
- `.env`: credencial local de Kaggle; no está versionada.
- `sub.csv`: archivo de predicciones con formato para envío a Kaggle.

## Licencia

Este proyecto se distribuye bajo los términos de [LICENSE](LICENSE).
