# Mejora del modelo de segmentación

## Objetivo

El proyecto entrena una U-Net con encoder EfficientNet-B2 para asignar a cada píxel de una tomografía una de cuatro clases mutuamente excluyentes:

1. *Ground glass*.
2. Consolidación.
3. Pulmón u otro tejido pulmonar.
4. Fondo.

La mejora busca detectar mejor las dos clases de lesión sin permitir que el fondo, que ocupa la mayor parte de las imágenes, domine el entrenamiento. También corrige inconsistencias del preprocesamiento y hace más estable la inferencia.

## Problema del modelo original

El análisis de las máscaras de Radiopaedia muestra un desbalance importante: aproximadamente el 84.25 % de los píxeles corresponde al fondo, mientras que *ground glass* representa cerca del 1.02 % y consolidación apenas el 0.23 %.

Con `CrossEntropyLoss` sin pesos, equivocarse en una lesión pequeña afecta muy poco la pérdida total comparado con equivocarse en el fondo. Por eso el modelo puede alcanzar una exactitud por píxel aparentemente alta aunque no segmente correctamente las lesiones.

Además, el flujo original presenta dos puntos mejorables:

- `preprocess_images` ya estandariza las tomografías, pero el `Dataset` original vuelve a normalizarlas. Esta doble normalización cambia innecesariamente la distribución de entrada.
- En la inferencia original se umbralizan canales por separado. Como las clases son excluyentes, esto puede producir más de una clase activa para el mismo píxel o dejarlo sin clase.

## Resumen de los cambios

| Componente | Implementación original | Implementación mejorada | Propósito |
|---|---|---|---|
| Pérdida | Cross Entropy sin pesos | Cross Entropy ponderada + Dice multiclase | Dar relevancia a las lesiones y optimizar su solapamiento |
| Dataset | Segunda normalización | Conserva la estandarización previa | Mantener una escala de entrada consistente |
| Aumentos | Transformaciones más generales | Rotación máxima de ±15°, desplazamiento, escala y reflejo | Crear variación sin deformar excesivamente la anatomía |
| Máscaras | Riesgo de interpolación continua | Vecino más cercano | Evitar clases fraccionarias o etiquetas nuevas |
| Entrenamiento | Seguimiento de métricas globales | Selección por `lesion_mIoU` | Elegir el modelo que segmenta mejor las lesiones |
| Estabilidad | Entrenamiento básico | Semilla, AdamW, ajuste del aprendizaje y recorte del gradiente | Reducir inestabilidad y mejorar reproducibilidad |
| Inferencia | Umbrales independientes | TTA horizontal, promedio de probabilidades y `argmax` | Obtener una única clase consistente por píxel |

## Por qué se mantiene Cross Entropy y no Binary Cross Entropy

La tarea es multiclase, no multietiqueta: cada píxel debe pertenecer a exactamente una de las cuatro clases. `CrossEntropyLoss` aplica la competencia entre clases mediante *softmax* y recibe una máscara de índices enteros con valores de 0 a 3.

La forma esperada de los datos es:

```python
logits.shape  # [lote, 4, alto, ancho]
target.shape  # [lote, alto, ancho]
```

`BCEWithLogitsLoss` trataría cada canal como una decisión binaria independiente. En consecuencia, podría considerar un mismo píxel como *ground glass* y consolidación al mismo tiempo. BCE sería adecuada si el problema se redujera a lesión contra no lesión con un solo canal, o si las clases pudieran superponerse realmente.

Por ello, la mejora conserva Cross Entropy y la complementa con pesos por clase y Dice.

## Componentes de la mejora

### `ImprovedCTDataset`

Organiza las imágenes y máscaras para entregarlas al modelo.

- Recibe imágenes ya estandarizadas por `preprocess_images` y no vuelve a normalizarlas.
- Mantiene las imágenes como `float32`.
- Convierte la imagen de `[alto, ancho, canales]` a `[canales, alto, ancho]`, que es el formato esperado por PyTorch.
- Convierte las máscaras a `int64`, formato requerido por Cross Entropy.
- Aplica la misma transformación geométrica a la imagen y su máscara.

Su objetivo es preservar la correspondencia entre cada tomografía y sus etiquetas y evitar cambios innecesarios en la escala de intensidades.

### `balanced_class_weights`

Cuenta cuántos píxeles existen de cada clase y calcula pesos mediante la raíz cuadrada de la frecuencia inversa:

```text
peso_clase = sqrt(total_de_píxeles / píxeles_de_la_clase)
```

Después normaliza los pesos para que su promedio sea uno. Las clases escasas reciben más importancia, pero la raíz cuadrada evita pesos tan extremos como los que produciría la frecuencia inversa directa.

### `WeightedCrossEntropyDiceLoss`

Combina dos objetivos con la misma importancia inicial:

```text
pérdida = 0.5 × Cross Entropy ponderada + 0.5 × Dice loss
```

La Cross Entropy ponderada evalúa si cada píxel recibió la clase correcta y aumenta el costo de equivocarse en las clases minoritarias. Dice compara la intersección entre la predicción y la máscara real con el tamaño total de ambas regiones; por eso se relaciona directamente con la calidad espacial de la segmentación.

Dentro de la función:

1. Se calcula Cross Entropy directamente a partir de los *logits*.
2. Se aplica `softmax` para obtener probabilidades por clase.
3. La máscara de índices se convierte temporalmente a *one-hot*.
4. Se calcula Dice para cada una de las cuatro clases.
5. Se promedian las clases y se combinan ambas pérdidas.

El parámetro `smooth` evita divisiones entre cero cuando una región es muy pequeña o no aparece en un lote.

### `update_confusion_matrix` e `iou_from_confusion`

`update_confusion_matrix` convierte los *logits* en una clase con `argmax` y acumula las coincidencias entre clase real y predicha. De esta matriz se obtiene el IoU de cada clase:

```text
IoU = intersección / unión
```

`iou_from_confusion` permite reportar por separado:

- IoU de *ground glass*.
- IoU de consolidación.
- IoU de pulmón.
- IoU de fondo.
- mIoU de lesiones, calculado con las dos primeras clases.
- mIoU global de las cuatro clases.

Esto evita depender únicamente de `pixel_accuracy`, que puede verse inflada por la abundancia de fondo.

### Aumentos de datos conservadores

Durante el entrenamiento se aplican pequeños desplazamientos, cambios de escala, rotaciones de hasta ±15° y reflejos horizontales. Las tomografías se interpolan de forma lineal, mientras que las máscaras usan vecino más cercano.

La diferencia de interpolación es importante: una imagen contiene intensidades continuas, pero una máscara contiene identificadores discretos. Interpolar una máscara linealmente podría inventar valores que no corresponden a ninguna clase.

En validación solamente se redimensionan los datos, para que la métrica no dependa de aumentos aleatorios.

### `fit_improved`

Realiza el ciclo de entrenamiento y validación:

1. Ejecuta la predicción del modelo.
2. Calcula la pérdida híbrida.
3. Propaga el error y actualiza los pesos.
4. Limita la norma del gradiente a 1.0 para reducir actualizaciones inestables.
5. Evalúa el IoU por clase sobre validación.
6. Ajusta la tasa de aprendizaje cuando `lesion_mIoU` deja de mejorar.
7. Guarda el checkpoint con el mejor `lesion_mIoU`.
8. Detiene el entrenamiento después de seis épocas sin mejora.
9. Restaura el mejor checkpoint antes de devolver el resultado.

El optimizador cambia a AdamW, que separa la regularización por decaimiento de pesos de la actualización adaptativa. El tamaño de lote se reduce a cuatro para disminuir el uso de memoria y realizar más actualizaciones por época.

### `predict_with_horizontal_tta`

Aplica *test-time augmentation* (TTA) durante la inferencia:

1. Predice las probabilidades de la imagen original.
2. Refleja horizontalmente la imagen y vuelve a predecir.
3. Deshace el reflejo sobre la segunda predicción.
4. Promedia ambas distribuciones de probabilidad.
5. Usa `argmax` una sola vez para seleccionar una clase por píxel.
6. Conserva los dos canales de lesión solicitados por la competencia.
7. Recupera el tamaño original con vecino más cercano.

El resultado se guarda en `sub_improved.csv`, sin sobrescribir `sub.csv` del experimento original.

### `evaluate_for_comparison`

Evalúa tanto el modelo original como el mejorado sobre validación y construye una tabla con las mismas métricas por clase. Su objetivo es comprobar si la mejora beneficia las lesiones y no solamente la puntuación global.

## Flujo completo de la versión mejorada

```text
Tomografías y máscaras
        ↓
Preprocesamiento y una sola estandarización
        ↓
ImprovedCTDataset + aumentos conservadores
        ↓
U-Net EfficientNet-B2
        ↓
Cross Entropy ponderada + Dice
        ↓
Validación con IoU por clase y lesion_mIoU
        ↓
Mejor checkpoint + TTA horizontal
        ↓
sub_improved.csv
```

## Riesgos y controles

| Riesgo | Causa posible | Control aplicado | Comprobación recomendada |
|---|---|---|---|
| Sobresegmentar lesiones | Pesos minoritarios demasiado altos | Raíz cuadrada de frecuencia inversa y combinación con Dice | Revisar falsos positivos e IoU por lesión |
| Ignorar lesiones pequeñas | Dominio del fondo en la pérdida | Pesos por clase, Dice y checkpoint por `lesion_mIoU` | Comparar IoU de las clases 0 y 1 |
| Doble normalización | Estandarizar en dos etapas | `ImprovedCTDataset` no normaliza nuevamente | Inspeccionar histogramas y rangos de entrada |
| Corromper etiquetas al redimensionar | Interpolación continua sobre máscaras | Interpolación de vecino más cercano | Verificar que solo existan las clases 0, 1, 2 y 3 |
| Inferencia multiclase inconsistente | Umbralizar cada canal por separado | `softmax` seguido de un único `argmax` | Confirmar una sola clase por píxel |
| Gradientes inestables | Desbalance o actualizaciones grandes | Recorte de norma a 1.0 y reducción de tasa de aprendizaje | Observar pérdidas, gradientes y valores no finitos |
| Sobreajuste | Pocos datos o demasiadas épocas | Aumentos, AdamW, validación y *early stopping* | Comparar curvas de entrenamiento y validación |
| TTA perjudicial | El reflejo no representa algunos patrones asimétricos | Promediar solo dos vistas y medir el resultado en validación | Comparar inferencia con y sin TTA |
| Resultado dependiente del azar | Inicialización y orden de los lotes | Semilla fija en Python, NumPy, PyTorch y `DataLoader` | Repetir el experimento si se necesita mayor certeza |

Los controles reducen los riesgos, pero no garantizan por sí solos una mejora. El criterio final debe ser la comparación sobre datos de validación no usados para actualizar el modelo.

## Cómo validar la mejora

La comparación debe usar el mismo conjunto de validación para ambos modelos. Las métricas principales son `IoU ground glass`, `IoU consolidación` y `mIoU lesiones`. `mIoU global`, pérdida y exactitud por píxel pueden conservarse como métricas secundarias.

La mejora se considera útil si aumenta de forma consistente el desempeño de las lesiones sin provocar una caída inaceptable en pulmón y fondo. También deben inspeccionarse predicciones representativas, porque una sola media no revela si el modelo produce falsos positivos grandes o pierde lesiones pequeñas.

Como la versión mejorada modifica simultáneamente la pérdida, el dataset, los aumentos, el optimizador y la inferencia, la comparación indica el efecto del conjunto de cambios, pero no demuestra cuánto aporta cada uno. Para aislarlos se necesitaría un estudio de ablación: añadir un cambio a la vez y repetir la evaluación con varias semillas.

## Archivos generados

- `Unet-efficientnet.pt`: checkpoint del modelo original.
- `Unet-efficientnet-improved.pt`: mejor checkpoint según `lesion_mIoU`.
- `sub.csv`: predicciones del modelo original.
- `sub_improved.csv`: predicciones de la versión mejorada con TTA.

En conjunto, la mejora cambia el objetivo práctico del entrenamiento: en lugar de premiar principalmente la clasificación del fondo, dirige el aprendizaje y la selección del modelo hacia el solapamiento correcto de las regiones de lesión.
