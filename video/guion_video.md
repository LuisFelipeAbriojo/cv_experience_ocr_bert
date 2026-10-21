# Guion del video (máx. 5 min) — "¿Cuántos años de experiencia tiene este CV?"

Formato: pitch del sistema, no recorrido de código. Graba la pantalla con la **app de Streamlit**
(`streamlit run app.py`) y habla encima. Tiempo objetivo por bloque entre corchetes; total ≈ 4:45.
Lo que dices va en texto normal; lo que muestras va en *[cursiva entre corchetes]*.

---

## 1. El problema y quién lo necesita [0:00 – 0:35]

*[Pestaña 1 · Problem & Methodology, arriba]*

> Una agencia de reclutamiento o una bolsa de trabajo recibe miles de CVs como imágenes o PDF.
> Uno de los primeros datos que necesitan es: ¿cuántos años de experiencia tiene esta persona?
> Casi nunca está escrito. Un reclutador lo calcula mentalmente: mira cada trabajo, lee sus fechas,
> suma las duraciones y no cuenta los huecos ni los años de estudio.
> Construimos un sistema que hace eso automáticamente, desde la imagen hasta un número entre 0 y 20,
> y medimos con números si un modelo tipo BERT realmente vale la pena.

## 2. De la imagen al texto: PaddleOCR [0:35 – 1:20]

*[Pestaña 2 · OCR & Exploratory Analysis → bajar a "Example: PaddleOCR boxes" y mostrar el CV con cajas al lado del texto]*

> Usamos 4.000 CVs sintéticos del dataset PopResume, 800 por cada una de cinco ocupaciones, solo
> versiones sin foto y con semilla fija. Como la entrada es una imagen, primero necesitamos OCR.
> PaddleOCR detecta cada línea — estas cajas — y la transcribe con su nivel de confianza.
> Usamos los modelos móviles PP-OCRv5 porque todo corre en la CPU de una laptop: 2,7 segundos por CV,
> cero imágenes fallidas.

*[Subir a las métricas de OCR]*

> La calidad es muy alta: 1,2 % de error por carácter y el 99,996 % de los años se leen con los dígitos
> correctos. Pero encontramos un fallo importante: en la mitad de los rangos, PaddleOCR **se come el
> guion** entre fechas — "2017.10–2020.06" sale como "2017.102020.06". Nuestra limpieza lo repara, y
> además elimina nombre, teléfono, correo y estado, y enmascara cualquier frase que diga la respuesta
> directamente. Revisamos 50 textos a mano: ningún CV decía su experiencia ni su edad.

## 3. Por qué es difícil — un ejemplo real [1:20 – 1:55]

*[Pestaña 4 · Browse test CVs → abrir el CV **05013**]*

> Este contador tiene tres trabajos: 2011 a 2014, 2015 a 2018 y 2019 a 2023. Si restas el primer año
> del último te da casi 12 años. Pero entre los trabajos hay casi dos años de huecos: la respuesta
> correcta es 9,9. Para acertar, el modelo tiene que encontrar cada rango, restarlo, sumarlo y saltarse
> los huecos — y no confundirse con el año de graduación.

## 4. Cuatro formas de adivinar [1:55 – 2:40]

*[Pestaña 1 · tabla "The models"]*

> Comparamos cuatro modelos, todos con el mismo OCR, la misma partición 70/15/15 y el mismo test:
> - La **mediana**: siempre dice 5,6 años. Es el piso; cualquier modelo útil debe ganarle.
> - **TF-IDF + Ridge**: cuenta qué palabras y qué años aparecen, pero no sabe cómo se relacionan.
> - **Embeddings + Ridge**: resume el "significado" del CV con un encoder congelado.
> - **BERT afinado**: DistilBERT entrenado como regresor; lee la secuencia completa y puede aprender a
>   relacionar fechas.
> Y como bonus, una **regla** de 20 líneas que extrae los rangos de fechas y los suma.

## 5. Resultados [2:40 – 3:40]

*[Pestaña 3 · Model Results → tabla comparativa]*

> El error medio de la mediana es 3,5 años. TF-IDF baja a 0,87, los embeddings a 1,22.
> **BERT llega a 0,20 años — unos dos meses y medio — y acierta dentro de ±2 años en el 100 % de los CVs.**
> Frente a TF-IDF reduce el error en 0,67 años, con un intervalo de confianza bootstrap de 0,61 a 0,73:
> la mejora es real.

*[Seleccionar el gráfico predicted vs true y mostrar el de 5 modelos]*

> Aquí se ve: los modelos simples aplanan las predicciones hacia el medio — subestiman a los seniors —;
> BERT sigue la diagonal.

*[Pestaña 4 · ejemplos buenos y malos]*

> Un acierto: en el CV que vimos, BERT dice 9,9 exacto; TF-IDF dice 11,7 porque cuenta el lapso
> completo. Un error: en el CV 25879, una enfermera con 3 años y un hueco de seis meses, BERT dice 4,2.
> Seis de sus diez peores errores son justamente huecos de carrera.

## 6. ¿Vale la pena BERT? ¿Es justo? [3:40 – 4:45]

*[Pestaña 3 → sección Q1, tabla del probe]*

> Ahora lo incómodo: la regla de sumar fechas tiene un error de solo 0,025 años, ocho veces menos que
> BERT, sin entrenamiento. Y cuando movimos todas las fechas cinco años — algo que no cambia la
> experiencia — BERT cambió su predicción en casi 2 años. BERT aprendió un atajo estadístico, no la
> aritmética. Entonces: BERT cuesta 88 minutos de entrenamiento y supera claramente a los modelos de
> texto, pero en este dataset tan regular una regla simple es mejor; en CVs reales con formatos
> variados, BERT sería más robusto — de hecho, cuando el OCR cortó una fecha, la regla falló por 2,6
> años y BERT solo por 0,08.

*[Bajar a Q4 · tabla de fairness]*

> En equidad, no vemos diferencias por sexo: el error medio es prácticamente cero para mujeres y
> hombres. Por raza, el grupo afroamericano muestra una leve subestimación de 0,13 años, pero con solo
> 31 CVs no podemos distinguirla del ruido. Sexo y raza nunca son entradas; si hay sesgo, entra por
> variables proxy como la duración de la carrera.

> Conclusión: este sistema sirve para **pre-llenar** un campo que un reclutador revisa. Subestimar
> experiencia descarta en silencio a buenos candidatos, y los años no miden competencia.
> **Nunca debe decidir a quién se contrata.** Gracias.

---

### Checklist de la rúbrica cubierto
- Definición del problema y quién lo necesita → bloque 1
- Por qué OCR, qué extrae PaddleOCR y qué tan bien lee fechas → bloque 2
- Por qué es difícil, con un CV real → bloque 3
- Intuición de mediana, TF-IDF, embeddings y BERT → bloque 4
- MAE y ±2 años por modelo, gráfico predicho vs real, un acierto y un error → bloque 5
- Ganancia vs costo, fairness y lo que nunca debe hacer → bloque 6

### Consejos
- Ensaya con cronómetro; si te pasas, recorta el ejemplo del OCR que corta una fecha (bloque 6).
- Abre la app y deja cargadas las pestañas antes de grabar (la primera carga tarda unos segundos).
- Sube el video (YouTube no listado o Google Drive con acceso por enlace) y pega el link en `video/link.txt`.
