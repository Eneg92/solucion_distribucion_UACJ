Optimización de Red Logística · Streamlit + Pyomo
Aplicación web en Streamlit para optimizar una red de suministro multinivel (Planta - Centro - Cliente) con Pyomo
Sube 5 archivos CSV con la red, ejecuta la optimización y explora un dashboard interactivo con KPIs, utilización de capacidad, paretos por costo y tablas de envío.
________________________________________
Características
•	Carga de datos vía 5 CSV (formato controlado).
•	Transformación automática de matrices de costos a formato largo (long).
•	Optimización del costo total:
o	Costo de producción por planta–producto.
o	Costo de transporte Planta-Centro.
o	Costo de transporte Centro-Cliente.
•	Restricciones:
o	Satisfacción exacta de demanda por cliente–producto.
o	Balance de flujo en cada centro.
o	Capacidades por planta y por centro.
•	Dashboard:
o	KPIs globales (costo óptimo, demanda cubierta, producción/flujo vs capacidad).
o	Gauges de utilización total (producción y centros).
o	Utilización filtrada por planta/centro/producto/cliente.
o	Paretos de costos (Total, C-J directo, P-C asignado, Producción asignado).
o	Scatter de clientes menos rentables (cuartil inferior de demanda).
o	Tablas de envíos detalladas (P-C y C-J) con filtros.
________________________________________
Estructura de Datos (CSV)
Los nombres de archivo y columnas deben coincidir exactamente.
1) plantas.csv
Columnas:
Planta, Producto, Capacidad_Produccion, Costo_Produccion
2) centros.csv
Columnas:
Centro, Producto, Capacidad_Almacenamiento
3) clientes.csv
Columnas:
Cliente, Producto, Demanda
4) Costos Plantas x CeDis.csv (matriz P→C)
Columnas:
Planta, Producto , centro1, centro2, centro3, ...
Nota: la columna Producto puede venir con espacio al final; la app la normaliza a Producto.
5) Costos CeDis x Cliente.csv (matriz C→J)
Columnas:
Producto, Centro, Cliente1, Cliente2, Cliente3, ...
________________________________________
Uso (Usuario Final)
1.	Abre el enlace público de la app.
2.	En el panel lateral, sube los 5 CSV:
o	plantas.csv
o	centros.csv
o	clientes.csv
o	Costos Plantas x CeDis.csv
o	Costos CeDis x Cliente.csv
3.	Presiona “Ejecutar Optimización” y espera el indicador de proceso.
4.	Explora el dashboard:
o	KPIs globales, gauges de utilización.
o	Filtros por Producto, Planta, Centro, Cliente.
o	Paretos y scatter de clientes menos rentables.
o	Tablas de envíos con los filtros aplicados.
________________________________________
Despliegue (Streamlit Community Cloud)
Método gratuito recomendado.
1.	GitHub
o	Crea un repo público y sube:
	solucion_distribucion_ENEG.py (app Streamlit)
	requirements.txt
	packages.txt (para instalar GLPK)
	README.md (este archivo)
	(opcional) logo_uacj.png para el encabezado
2.	Streamlit Cloud
o	Inicia sesión con tu cuenta de GitHub.
o	New app → selecciona tu repo.
o	Campo Main file path: solucion_distribucion_ENEG.py
o	Click Deploy.
o	La nube instalará dependencias desde requirements.txt y packages.txt (incluye GLPK).
________________________________________
Navegación del Dashboard
1.	KPIs globales: costo, demanda cubierta, producción/flujo vs capacidades.
2.	Gauges: utilización total de producción y centros.
3.	Filtros: Producto, Planta, Centro, Cliente.
4.	Utilización:
o	Barras por planta y por centro (con porcentajes).
5.	Paretos (Top 25):
o	Costo Total asignado por cliente.
o	Costo C-J (directo).
o	Costo P-C asignado.
o	Costo de Producción asignado.
6.	Scatter: clientes del 25% inferior de demanda con alto costo total.
7.	Tablas: envíos Planta-Centro y Centro-Cliente con los filtros vigentes.
________________________________________
Tips y resolución de problemas
1.	Nombres y columnas: deben coincidir exactamente con la sección de estructura de datos.
2.	GLPK no disponible: revisa packages.txt (nube) o tu instalación local.
3.	Capacidad insuficiente: si la suma de capacidades < demanda, Pyomo puede no encontrar solución.
4.	IDs inconsistentes: la app normaliza Planta_i, Centro_i, Cliente_i, Producto_i; evita duplicados raros.
________________________________________
Privacidad
Los CSV se usan solo en memoria para resolver el modelo y construir el dashboard. No se almacenan de forma persistente en el servidor de Streamlit Community Cloud.
________________________________________
Stack
1.	Frontend: Streamlit, Plotly (Graph Objects / Express)
2.	Optimización: Pyomo + GLPK
3.	Pandas para ETL y agregaciones
________________________________________
Archivos del repo
solucion_distribucion_ENEG.py   # App Streamlit
requirements.txt                # Dependencias Python
packages.txt                    # Instalación GLPK (Streamlit Cloud)
README.md                       # Este documento
logo_uacj.png                   # (opcional) Logo para el header
________________________________________
