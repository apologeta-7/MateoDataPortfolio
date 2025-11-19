# Día 2 – Revisión de EDA externos y toma de decisiones

## 1. Contexto del Día 2

En este segundo día de trabajo con el reto **Santander Customer Transaction Prediction**, mi objetivo no ha sido tanto escribir código nuevo, sino **entender cómo otros Data Scientists abordan el EDA (Análisis Exploratorio de Datos)** y tomar decisiones conscientes sobre qué me sirve y qué no en esta etapa de aprendizaje.

He revisado **tres EDAs de proyectos distintos** relacionados con este reto.

---

## 2. Proyectos revisados

1. **Proyecto A – EDA + modelo avanzado de competición**
   - Tipo: código orientado a Kaggle, muy optimizado.
   - Nivel: avanzado / competición.
   - Comentarios personales:
     - Aquí puedes escribir qué te encontraste, qué técnicas viste, qué te llamó la atención, etc.
     - Ejemplo: uso de detección de datos sintéticos, frequency encoding masivo, normalización global…

2. **Proyecto B – EDA_001 / EDA_002 (Basic Stats + análisis por target)**
   - Tipo: notebooks centrados en estadísticas básicas y relación con el target.
   - Nivel: intermedio-alto.
   - Comentarios personales:
     - Aquí resume qué viste: `.info()`, `.describe()`, histogramas por target, etc.
     - Qué te pareció útil y qué se te hace todavía grande.

3. **Proyecto C – EDA de referencia (el que me quedo como guía)**
   - Tipo: EDA más simple y legible.
   - Nivel: adecuado para mi momento actual.
   - Comentarios personales:
     - Aquí describe por qué este sí encaja contigo.
     - Qué hace bien: estructura, claridad, gráficos, etc.

---

## 3. Decisión: qué descarto y qué mantengo

### 3.1 Proyectos que descarto (por ahora)

He decidido **no seguir profundizando** en:

- **Proyecto A**  
- **Proyecto B**  

Los motivos:

- Utilizan técnicas de competición (bagging, pseudo-labelling, redes neuronales, optimización agresiva) que **superan mi nivel actual**.
- El EDA está muy mezclado con la lógica de modelado, lo que dificulta usarlo como material didáctico.
- No aportan claridad en esta fase; ahora necesito **comprensión**, no maximizar leaderboard.

Esta decisión no es una renuncia, sino una **foto honesta de dónde estoy ahora**.

> 💡 *Aprendizaje:* ver código que me supera me ayuda a entender que hay niveles muy avanzados a los que quiero llegar, pero no necesito imitarlos hoy.

---

### 3.2 Proyecto que sí mantengo como referencia

Me quedo con el **Proyecto C** como EDA de referencia porque:

- Es más **claro y estructurado**.
- Se centra en:
  - análisis general de columnas,
  - estadísticas descriptivas,
  - relación básica con el target,
  - gráficos interpretables.
- Es **alcanzable** con los conocimientos que tengo ahora mismo.
- Puedo usarlo como “plantilla mental” para diseñar **mi propio EDA**.

---

## 4. Aprendizajes clave del Día 2

1. **Reconocer mi nivel actual es parte del progreso.**  
   No necesito entenderlo todo hoy. Puedo aceptar que hay código que ahora mismo me queda grande y usarlo como referencia futura.

2. **Ver niveles superiores me da dirección, no frustración.**  
   Saber que existen EDAs mucho más avanzados me ayuda a trazar un objetivo profesional:  
   > *poder acercarme a ese nivel en mi práctica laboral dentro de unos años.*

3. **Elegir bien las referencias ahorra tiempo y energía.**  
   No todo código que está en GitHub/Kaggle es material adecuado para aprender.  
   Es mejor tener **una buena referencia alineada a mi nivel** que tres que me saturen.

4. **El siguiente paso ya está claro: construir mi propio EDA.**  
   El Día 3 lo dedicaré a:
   - cargar los datos yo mismo,
   - hacer mi inspección básica,
   - explorar variables,
   - y dejar mi primera versión de EDA lista.

---

## 5. Plan para el Día 3

En el **Día 3** voy a empezar a generar **mi propio EDA** sobre este reto, tomando como referencia el EDA que he seleccionado hoy.

El plan incluye:

1. Cargar `train.csv` y `test.csv`.
2. Revisar:
   - dimensiones de los datasets (`shape`),
   - tipos de datos (`info`),
   - estadísticas básicas (`describe`),
   - número de valores únicos por columna.
3. Analizar la variable `target`:
   - distribución de 0 vs 1,
   - posibles desbalances.
4. Explorar algunas variables individuales:
   - histogramas,
   - boxplots sencillos,
   - distribución por target.
5. Documentar todo lo que vaya viendo y las dudas que aparezcan.

El objetivo no es “hacer el EDA perfecto”, sino **tener una primera versión propia** que luego pueda comparar con el EDA de referencia.

---

## 6. Estado final del Día 2

- ✅ Tres EDAs revisados.  
- ✅ Dos descartados por ser de nivel avanzado de competición.  
- ✅ Uno elegido como referencia didáctica.  
- ✅ Aprendizajes anotados sobre mi nivel y mis objetivos.  
- ✅ Siguiente paso definido: generar mi propio EDA en el Día 3.
