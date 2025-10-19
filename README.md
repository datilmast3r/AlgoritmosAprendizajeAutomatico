# Algoritmo Candidate-Elimination en Weka

Este proyecto es una implementación en Java del algoritmo de aprendizaje conceptual **Candidate-Elimination**, utilizando la librería Weka. El programa entrena el clasificador con un conjunto de datos en formato ARFF y muestra la evolución del espacio de versiones (los conjuntos de hipótesis S y G) a medida que procesa cada instancia.

## Requisitos Previos

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

*   **Java Development Kit (JDK)**: Versión 17 o superior. Puedes verificar tu versión con `java -version`.
*   **Apache Maven**: Para compilar el proyecto y gestionar las dependencias. Puedes verificar tu instalación con `mvn -version`.

Ambos (`java` y `mvn`) deben estar configurados en las variables de entorno (PATH) de tu sistema para poder ejecutarlos desde cualquier ubicación en la consola.

## Cómo Ejecutar el Programa

Sigue estos pasos para compilar y ejecutar el algoritmo:

### 1. Abrir una Consola

Abre una terminal o línea de comandos y navega hasta el directorio raíz del proyecto (la carpeta `ev` que contiene el archivo `pom.xml`).

```bash
cd ruta/a/tu/proyecto/ev
```

### 2. Compilar y Ejecutar con Maven

Ejecuta el siguiente comando. Maven se encargará de descargar las dependencias (como Weka), compilar el código y ejecutar la clase principal (`Main.java`).

```bash
mvn compile exec:java -Dexec.mainClass="aprendizaje.automatico.Main"
```

Por defecto, el programa está configurado para usar el archivo `data/weather.nominal2.arff`.

## Usar un Conjunto de Datos Diferente

1.  **Coloca tu archivo `.arff`** en el directorio `data/`.
2.  **Abre el archivo** `src/main/java/aprendizaje/automatico/Main.java`.
3.  **Modifica la variable `dataPath`** para que apunte a tu nuevo archivo. Por ejemplo, para usar el conjunto de datos completo de `weather`:
    ```java
    String dataPath = "data/weather.nominal.arff";
    ```
4.  **Vuelve a ejecutar** el comando de Maven como se indicó en el paso anterior.