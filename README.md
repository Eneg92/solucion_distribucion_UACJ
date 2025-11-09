
Aplicación de Optimización de Red Logística
Esta es una aplicación web de Streamlit que permite a los usuarios cargar datos de una red logística, ejecutar un modelo de optimización (Pyomo con GLPK) y visualizar los resultados en un dashboard interactivo.

Cómo Usar la Aplicación (Usuario Final)
Abre el enlace web de la aplicación.
En el panel lateral, carga los 5 archivos CSV de datos:
plantas.csv, centros.csv, clientes.csv, costos.csv, productos.csv

Una vez cargados los 5 archivos, el botón "Ejecutar Optimización" se activará. Haz clic en él.
Espera a que el modelo se resuelva (verás un indicador).
¡Listo! El dashboard aparecerá con todos los gráficos y KPIs de la solución óptima.

---------------------------------------------------------------------

Instrucciones de Despliegue (Para el Desarrollador)

Este método publica tu aplicación en un enlace web público y gratuito.

Crea un Repositorio en GitHub:

Regístrate gratis en GitHub.

Crea un nuevo repositorio público.

Sube los siguientes 4 archivos a ese repositorio:

solucion_distribucion_ENEG.py (la aplicación de Streamlit)

requirements.txt (las librerías de Python)

packages.txt (el archivo que instala GLPK en Streamlit Cloud)

Este archivo README.md (opcional, pero recomendado)

Despliega en Streamlit Community Cloud:

Regístrate gratis en Streamlit Community Cloud (puedes usar tu cuenta de GitHub).

Haz clic en "New App" (Nueva Aplicación).

En "Repository", selecciona el repositorio de GitHub que acabas de crear.

Asegúrate de que la "Main file path" (Ruta del archivo principal) sea solucion_distribucion_ENEG.py.

Haz clic en "Deploy!" (Desplegar).

Streamlit leerá automáticamente tus archivos requirements.txt y packages.txt, instalará todo (incluido GLPK) y pondrá tu aplicación en línea.
