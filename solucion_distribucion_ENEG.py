import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyomo.environ import *
import io

st.set_page_config(layout="wide")

# --- NUEVA FUNCIÓN DE TRANSFORMACIÓN Y RESOLUCIÓN ---
# Modificada para aceptar los nuevos archivos de costos y transformarlos
def transformar_y_resolver(df_plantas, df_centros, df_clientes, df_costos_pc_raw, df_costos_cj_raw):
    try:
        # --- 1. Transformación de Costos (Planta-Centro) ---
        
        # Renombrar la columna 'Producto ' (con espacio) a 'Producto'
        df_costos_pc_raw = df_costos_pc_raw.rename(columns={'Producto ': 'Producto'})
        
        # 'Derretir' (un-pivot) el DataFrame
        id_vars_pc = ['Planta', 'Producto']
        value_vars_pc = [col for col in df_costos_pc_raw.columns if col not in id_vars_pc]
        df_costos_pc_long = pd.melt(df_costos_pc_raw, 
                                    id_vars=id_vars_pc, 
                                    value_vars=value_vars_pc, 
                                    var_name='Centro', 
                                    value_name='Costo_Plant_Centro')

        # Estandarizar los IDs para que coincidan con los otros archivos
        df_costos_pc_long['Planta'] = 'Planta_' + df_costos_pc_long['Planta'].astype(str)
        df_costos_pc_long['Producto'] = 'Producto_' + df_costos_pc_long['Producto'].astype(str)
        df_costos_pc_long['Centro'] = 'Centro_' + df_costos_pc_long['Centro'].str.replace('centro', '')

        # --- 2. Transformación de Costos (Centro-Cliente) ---
        
        # 'Derretir' (un-pivot) el DataFrame
        id_vars_cj = ['Producto', 'Centro']
        value_vars_cj = [col for col in df_costos_cj_raw.columns if col not in id_vars_cj]
        df_costos_cj_long = pd.melt(df_costos_cj_raw, 
                                    id_vars=id_vars_cj, 
                                    value_vars=value_vars_cj, 
                                    var_name='Cliente', 
                                    value_name='Costo_Centro_Cliente')
        
        # Estandarizar los IDs
        df_costos_cj_long['Producto'] = 'Producto_' + df_costos_cj_long['Producto'].astype(str)
        df_costos_cj_long['Centro'] = 'Centro_' + df_costos_cj_long['Centro'].astype(str)
        df_costos_cj_long['Cliente'] = 'Cliente_' + df_costos_cj_long['Cliente'].str.replace('Cliente', '')

        
        # --- 3. Lógica de Optimización (como antes, pero con los DFs transformados) ---
        
        # Calcular totales
        total_demanda_requerida = df_clientes['Demanda'].sum()
        total_capacidad_produccion = df_plantas['Capacidad_Produccion'].sum()
        total_capacidad_almacenamiento = df_centros['Capacidad_Almacenamiento'].sum()

        # Conjuntos
        P = df_plantas['Planta'].unique()
        C = df_centros['Centro'].unique()
        J = df_clientes['Cliente'].unique()
        # El conjunto K ahora se deriva de df_plantas, no del df_productos
        K = df_plantas['Producto'].unique() 

        # Parámetros (usando los DFs originales y los nuevos DFs transformados)
        demanda = df_clientes.set_index(['Cliente', 'Producto'])['Demanda'].to_dict()
        cap_prod = df_plantas.set_index(['Planta', 'Producto'])['Capacidad_Produccion'].to_dict()
        cost_prod = df_plantas.set_index(['Planta', 'Producto'])['Costo_Produccion'].to_dict()
        cap_alm = df_centros.set_index(['Centro', 'Producto'])['Capacidad_Almacenamiento'].to_dict()
        
        # Usar los DFs largos y limpios directamente
        cost_pc = df_costos_pc_long.set_index(['Planta', 'Centro', 'Producto'])['Costo_Plant_Centro'].to_dict()
        cost_cj = df_costos_cj_long.set_index(['Centro', 'Cliente', 'Producto'])['Costo_Centro_Cliente'].to_dict()

        model = ConcreteModel()
        model.P, model.C, model.J, model.K = Set(initialize=P), Set(initialize=C), Set(initialize=J), Set(initialize=K)
        
        model.d = Param(model.J, model.K, initialize=demanda, default=0)
        model.cap_prod = Param(model.P, model.K, initialize=cap_prod, default=0)
        model.cap_alm = Param(model.C, model.K, initialize=cap_alm, default=0)
        model.cost_prod = Param(model.P, model.K, initialize=cost_prod, default=0)
        model.cost_pc = Param(model.P, model.C, model.K, initialize=cost_pc, default=0)
        model.cost_cj = Param(model.C, model.J, model.K, initialize=cost_cj, default=0)
        
        # Variables
        model.x = Var(model.P, model.C, model.K, within=NonNegativeReals)
        model.y = Var(model.C, model.J, model.K, within=NonNegativeReals)

        # Función Objetivo
        def funcion_objetivo_rule(model):
            costo_produccion = sum(model.cost_prod[p, k] * model.x[p, c, k]  
                                   for p in model.P for c in model.C for k in model.K if (p,k) in model.cost_prod and (p,c,k) in model.x)
            costo_transporte_pc = sum(model.cost_pc[p, c, k] * model.x[p, c, k]
                                      for p in model.P for c in model.C for k in model.K if (p,c,k) in model.cost_pc)
            costo_transporte_cj = sum(model.cost_cj[c, j, k] * model.y[c, j, k]
                                      for c in model.C for j in model.J for k in model.K if (c,j,k) in model.cost_cj)
            return costo_produccion + costo_transporte_pc + costo_transporte_cj
        
        model.objetivo = Objective(rule=funcion_objetivo_rule, sense=minimize)

        # Restricciones
        def satisfaccion_demanda_rule(model, j, k):
            return sum(model.y[c, j, k] for c in model.C) == model.d[j, k]
        model.r_demanda = Constraint(model.J, model.K, rule=satisfaccion_demanda_rule)

        def balance_flujo_rule(model, c, k):
            return sum(model.x[p, c, k] for p in model.P) == sum(model.y[c, j, k] for j in model.J)
        model.r_flujo = Constraint(model.C, model.K, rule=balance_flujo_rule)

        def capacidad_produccion_rule(model, p, k):
            return sum(model.x[p, c, k] for c in model.C) <= model.cap_prod[p, k]
        model.r_cap_prod = Constraint(model.P, model.K, rule=capacidad_produccion_rule)

        def capacidad_almacenamiento_rule(model, c, k):
            return sum(model.x[p, c, k] for p in model.P) <= model.cap_alm[c, k]
        model.r_cap_alm = Constraint(model.C, model.K, rule=capacidad_almacenamiento_rule)
        
        # Resolución
        solver = SolverFactory('glpk')
        if not solver.available():
            st.error("Error: El solver GLPK no está disponible.")
            return None, None, None, None, None, "Solver no encontrado"
            
        results = solver.solve(model, tee=False)

        # Procesamiento de resultados
        if (results.solver.status == SolverStatus.ok) and \
           (results.solver.termination_condition == TerminationCondition.optimal):
            
            costo_optimo = model.objetivo()
            
            total_produccion_real = 0
            resultados_x = []
            for (p, c, k) in model.x:
                cantidad = model.x[p, c, k].value
                if cantidad > 0.01:
                    total_produccion_real += cantidad
                    resultados_x.append({'Planta': p, 'Centro': c, 'Producto': k, 'Cantidad': cantidad})
            df_x = pd.DataFrame(resultados_x)
            
            total_flujo_centros = total_produccion_real

            resultados_y = []
            for (c, j, k) in model.y:
                cantidad = model.y[c, j, k].value
                if cantidad > 0.01:
                    resultados_y.append({'Centro': c, 'Cliente': j, 'Producto': k, 'Cantidad': cantidad})
            df_y = pd.DataFrame(resultados_y)
            
            kpi_data = {
                'costo_total_optimizado': float(costo_optimo),
                'total_demanda_cubierta': int(total_demanda_requerida),
                'total_produccion_real': float(total_produccion_real),
                'total_capacidad_produccion': int(total_capacidad_produccion),
                'total_flujo_centros': float(total_flujo_centros),
                'total_capacidad_almacenamiento': int(total_capacidad_almacenamiento)
            }
            
            # Devolver los DFs de costos transformados para el dashboard
            return kpi_data, df_x, df_y, df_costos_pc_long, df_costos_cj_long, None

        else:
            msg = f"No se encontró una solución óptima. Estado: {results.solver.status}, Condición: {results.solver.termination_condition}. (Esto puede ocurrir si la capacidad es insuficiente para la demanda)"
            return None, None, None, None, None, msg

    except Exception as e:
        return None, None, None, None, None, f"Error durante la optimización o transformación: {str(e)}"

# Inicializar estado de la sesión
if 'model_run_success' not in st.session_state:
    st.session_state['model_run_success'] = False
if 'kpis' not in st.session_state:
    st.session_state['kpis'] = None
if 'df_pc' not in st.session_state:
    st.session_state['df_pc'] = None
if 'df_cj' not in st.session_state:
    st.session_state['df_cj'] = None
# Reemplazar 'df_costos' por los dos nuevos DFs de costos transformados
if 'df_costos_pc_long' not in st.session_state:
    st.session_state['df_costos_pc_long'] = None
if 'df_costos_cj_long' not in st.session_state:
    st.session_state['df_costos_cj_long'] = None
if 'df_plantas' not in st.session_state:
    st.session_state['df_plantas'] = None
if 'df_centros' not in st.session_state:
    st.session_state['df_centros'] = None

# --- BARRA LATERAL ACTUALIZADA ---
st.sidebar.header("Panel de Control")

with st.sidebar.expander("1. Cargar Archivos de Datos", expanded=True):
    file_plantas = st.file_uploader("Cargar 'plantas.csv'", type="csv")
    file_centros = st.file_uploader("Cargar 'centros.csv'", type="csv")
    file_clientes = st.file_uploader("Cargar 'clientes.csv'", type="csv")
    # Nuevos file uploaders para los archivos de costos
    file_costos_pc = st.file_uploader("Cargar 'Costos Plantas x CeDis.csv'", type="csv")
    file_costos_cj = st.file_uploader("Cargar 'Costos CeDis x Cliente.csv'", type="csv")

# Actualizar la lista de archivos cargados
files_uploaded = [file_plantas, file_centros, file_clientes, file_costos_pc, file_costos_cj]
all_files_loaded = all(f is not None for f in files_uploaded)

if st.sidebar.button("Ejecutar Optimización", disabled=not all_files_loaded, type="primary"):
    if all_files_loaded:
        with st.spinner("Leyendo archivos, transformando datos y ejecutando optimización..."):
            try:
                # Resetear el puntero de los archivos antes de leer
                file_plantas.seek(0); df_plantas = pd.read_csv(file_plantas)
                file_centros.seek(0); df_centros = pd.read_csv(file_centros)
                file_clientes.seek(0); df_clientes = pd.read_csv(file_clientes)
                # Cargar los nuevos archivos raw de costos
                file_costos_pc.seek(0); df_costos_pc_raw = pd.read_csv(file_costos_pc)
                file_costos_cj.seek(0); df_costos_cj_raw = pd.read_csv(file_costos_cj)
                
                # Llamar a la función actualizada
                kpis, df_x, df_y, df_costos_pc_long, df_costos_cj_long, error_msg = transformar_y_resolver(
                    df_plantas, df_centros, df_clientes, df_costos_pc_raw, df_costos_cj_raw
                )
                
                if kpis:
                    st.session_state['kpis'] = kpis
                    st.session_state['df_pc'] = df_x
                    st.session_state['df_cj'] = df_y
                    # Guardar los DFs transformados en el estado
                    st.session_state['df_costos_pc_long'] = df_costos_pc_long
                    st.session_state['df_costos_cj_long'] = df_costos_cj_long
                    # Guardar DFs base para los filtros de capacidad
                    st.session_state['df_plantas'] = df_plantas
                    st.session_state['df_centros'] = df_centros
                    
                    st.session_state['model_run_success'] = True
                    st.success("¡Optimización completada con éxito!")
                    st.balloons()
                else:
                    st.session_state['model_run_success'] = False
                    st.error(f"Fallo en la optimización: {error_msg}")

            except Exception as e:
                st.session_state['model_run_success'] = False
                st.error(f"Error al procesar los archivos: {e}")
    else:
        st.sidebar.warning("Por favor, cargue los 5 archivos CSV requeridos.")

# --- LÓGICA DE VISUALIZACIÓN DEL DASHBOARD ---
if not st.session_state['model_run_success']:
    
    # INFORMACIÓN DEL PROYECTO
    LOGO_FILE = "logo_uacj.png"  
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            try:
                st.image(LOGO_FILE, use_column_width='auto')
            except Exception as e:
                st.warning(f"No se pudo cargar el logo. Asegúrate de que 'logo_uacj.png' esté en tu repositorio de GitHub.")
        
        st.markdown("<h3 style='text-align: center;'>Universidad Autónoma de Ciudad Juárez</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><strong>Programa:</strong> Maestría en Inteligencia Artificial y Analítica de Datos</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><strong>Materia:</strong> Programación para Analítica Prescriptiva y de la Decisión</p>", unsafe_allow_html=True)
        
        st.markdown("---")  
        
        st.markdown("<p style='text-align: center;'><strong>Integrantes:</strong></p>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center;'>
        Esther Nohemi Encinas Guerrero<br>
        Jesús Alejandro Gutiérrez Araiza<br>
        Luis Alonso Lira Mota<br><br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><strong>Profesor:</strong></p>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center;'>
        Gilberto Rivera Zarate
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---") # La línea divisoria principal de la app
    
    #MENSAJE DE BIENVENIDA
    st.info("Bienvenido. Por favor, cargue los 5 archivos de datos en el panel lateral y haga clic en 'Ejecutar Optimización' para ver el dashboard.")
    
    st.subheader("Archivos Requeridos (Nuevo Formato):")
    st.markdown("""
    * **plantas.csv**: `Planta`, `Producto`, `Capacidad_Produccion`, `Costo_Produccion`
    * **centros.csv**: `Centro`, `Producto`, `Capacidad_Almacenamiento`
    * **clientes.csv**: `Cliente`, `Producto`, `Demanda`
    * **Costos Plantas x CeDis.csv**: *(Formato Matriz)* `Planta`, `Producto `, `centro1`, `centro2`, ...
    * **Costos CeDis x Cliente.csv**: *(Formato Matriz)* `Producto`, `Centro`, `Cliente1`, `Cliente2`, ...
    """)

else:
    
    st.markdown("<h1 style='text-align: center; color: #0047AB;'>📊 Dashboard de Optimización de Red Logística</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Cargar datos desde el estado (nombres actualizados)
    kpis = st.session_state['kpis']
    df_pc_full = st.session_state['df_pc']
    df_cj_full = st.session_state['df_cj']
    # Cargar los DFs de costos transformados
    df_costos_pc_unicos = st.session_state['df_costos_pc_long']
    df_costos_cj_unicos = st.session_state['df_costos_cj_long']
    
    df_plantas_full = st.session_state['df_plantas']
    df_centros_full = st.session_state['df_centros']
    
    # Validar que los nuevos DFs de costos se cargaron
    if df_costos_cj_unicos is None or df_costos_cj_unicos.empty or \
       df_costos_pc_unicos is None or df_costos_pc_unicos.empty or \
       df_plantas_full is None or df_centros_full is None:
        st.error("No se pudieron cargar todos los datos para el análisis. Por favor, intente ejecutar de nuevo.")
        st.stop()
        
    # La lógica para extraer costos únicos del 'df_costos' unificado ya no es necesaria,
    # porque df_costos_pc_unicos y df_costos_cj_unicos ya están listos.

    # Filtros del dashboard en la barra lateral
    st.sidebar.header("2. Filtros")
    
    productos_unicos = sorted(list(df_pc_full['Producto'].unique())) if not df_pc_full.empty else []
    
    producto_seleccionado = st.sidebar.selectbox(
        "Filtrar por Producto",
        ["Todos"] + productos_unicos
    )
    
    # Aplicar filtros
    if producto_seleccionado != "Todos":
        df_pc_working = df_pc_full[df_pc_full['Producto'] == producto_seleccionado]
        df_cj_working = df_cj_full[df_cj_full['Producto'] == producto_seleccionado]
        df_plantas_working = df_plantas_full[df_plantas_full['Producto'] == producto_seleccionado]
        df_centros_working = df_centros_full[df_centros_full['Producto'] == producto_seleccionado]
    else:
        df_pc_working = df_pc_full
        df_cj_working = df_cj_full
        df_plantas_working = df_plantas_full
        df_centros_working = df_centros_full

    plantas_disponibles = sorted(list(df_pc_working['Planta'].unique())) if not df_pc_working.empty else []
    centros_pc_disp = set(df_pc_working['Centro'].unique()) if not df_pc_working.empty else set()
    centros_cj_disp = set(df_cj_working['Centro'].unique()) if not df_cj_working.empty else set()
    centros_disponibles = sorted(list(centros_pc_disp | centros_cj_disp))
    clientes_disponibles = sorted(list(df_cj_working['Cliente'].unique())) if not df_cj_working.empty else []


    planta_seleccionada = st.sidebar.selectbox(
        "Filtrar por Planta", ["Todos"] + plantas_disponibles
    )
    centro_seleccionado = st.sidebar.selectbox(
        "Filtrar por Centro", ["Todos"] + centros_disponibles
    )
    cliente_seleccionado = st.sidebar.selectbox(
        "Filtrar por Cliente", ["Todos"] + clientes_disponibles
    )

    # Aplicar filtros de geografía
    df_pc_filt = df_pc_working
    df_cj_filt = df_cj_working
    df_plantas_filt = df_plantas_working
    df_centros_filt = df_centros_working

    if planta_seleccionada != "Todos":
        df_pc_filt = df_pc_filt[df_pc_filt['Planta'] == planta_seleccionada]
        df_plantas_filt = df_plantas_filt[df_plantas_filt['Planta'] == planta_seleccionada]
    if cliente_seleccionado != "Todos":
        df_cj_filt = df_cj_filt[df_cj_filt['Cliente'] == cliente_seleccionado]
    if centro_seleccionado != "Todos":
        df_pc_filt = df_pc_filt[df_pc_filt['Centro'] == centro_seleccionado]
        df_cj_filt = df_cj_filt[df_cj_filt['Centro'] == centro_seleccionado]
        df_centros_filt = df_centros_filt[df_centros_filt['Centro'] == centro_seleccionado]

    #KPIs
    col1, col2 = st.columns(2)
    
    label_css = "font-size: 1rem; font-weight: bold; color: rgba(0, 0, 0, 0.7);"
    value_css = "font-size: 24px; font-weight: bold; line-height: 1.5;"  

    costo_label = "Costo Total Óptimo"
    costo_value = f"${kpis['costo_total_optimizado']:,.2f}"
    
    col1.markdown(f"<div style='text-align: left;'><span style='{label_css}'>{costo_label}</span><br><span style='{value_css}'>{costo_value}</span></div>", unsafe_allow_html=True)

    demanda_label = "Demanda Total Cubierta"
    demanda_value = f"{kpis['total_demanda_cubierta']:,.0f} Unidades"

    col2.markdown(f"<div style='text-align: left;'><span style='{label_css}'>{demanda_label}</span><br><span style='{value_css}'>{demanda_value}</span></div>", unsafe_allow_html=True)
    st.markdown("---")
 
    #Gauges
    st.header("Utilización de Capacidad Total (General)")
    gauge1, gauge2 = st.columns(2)

    with gauge1:
        if kpis['total_capacidad_produccion'] > 0:
            util_prod = (kpis['total_produccion_real'] / kpis['total_capacidad_produccion']) * 100
        else:
            util_prod = 0
        fig_gauge_prod = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = util_prod,
            number = {'suffix': "%", 'font': {'size': 40}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Producción vs. Capacidad Total", 'font': {'size': 24}},
            delta = {'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},  
            gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                     'bar': {'color': "#0047AB"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                     'steps': [{'range': [0, 50], 'color': '#FFB0B0'}, {'range': [50, 80], 'color': '#FFF1B0'}, {'range': [80, 100], 'color': '#B0FFB0'}]
            }))
        fig_gauge_prod.update_layout(height=350, margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig_gauge_prod, use_container_width=True)

    with gauge2:
        if kpis['total_capacidad_almacenamiento'] > 0:
            util_cd = (kpis['total_flujo_centros'] / kpis['total_capacidad_almacenamiento']) * 100
        else:
            util_cd = 0
        fig_gauge_cd = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = util_cd,
            number = {'suffix': "%", 'font': {'size': 40}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Flujo vs. Capacidad Total de Centros", 'font': {'size': 24}},
            delta = {'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                     'bar': {'color': "#0047AB"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                     'steps': [{'range': [0, 50], 'color': '#FFB0B0'}, {'range': [50, 80], 'color': '#FFF1B0'}, {'range': [80, 100], 'color': '#B0FFB0'}]
            }))
        fig_gauge_cd.update_layout(height=350, margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig_gauge_cd, use_container_width=True)
    st.markdown("---")

    # Infraestructura
    st.header("Utilización de Infraestructura (Filtrado)")
    st.caption("Muestra la utilización real vs. la capacidad disponible para los filtros seleccionados.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Utilización de Plantas")
        cap_plantas = df_plantas_filt.groupby('Planta')['Capacidad_Produccion'].sum().reset_index()
        prod_real = df_pc_filt.groupby('Planta')['Cantidad'].sum().reset_index()
        df_util_plantas = pd.merge(cap_plantas, prod_real, on='Planta', how='left').fillna(0)
        df_util_plantas = df_util_plantas[df_util_plantas['Capacidad_Produccion'] > 0]
        
        if not df_util_plantas.empty:
            df_util_plantas['Utilización (%)'] = (df_util_plantas['Cantidad'] / df_util_plantas['Capacidad_Produccion']) * 100
            df_util_plantas = df_util_plantas.sort_values(by='Utilización (%)', ascending=False)
            
            fig_util_p = px.bar(df_util_plantas, y='Planta', x='Utilización (%)',  
                                text=df_util_plantas['Utilización (%)'].apply(lambda x: f'{x:.1f}%'),
                                title='Utilización por Planta',
                                hover_data=['Cantidad', 'Capacidad_Produccion'])
            fig_util_p.update_layout(xaxis_range=[0, 110])
            st.plotly_chart(fig_util_p, use_container_width=True)
        else:
            st.info("No hay datos de utilización de plantas para la selección actual.")

    with col2:
        st.subheader("Utilización de Centros (Flujo)")
        cap_centros = df_centros_filt.groupby('Centro')['Capacidad_Almacenamiento'].sum().reset_index()
        flujo_real = df_pc_filt.groupby('Centro')['Cantidad'].sum().reset_index()
        df_util_centros = pd.merge(cap_centros, flujo_real, on='Centro', how='left').fillna(0)
        df_util_centros = df_util_centros[df_util_centros['Capacidad_Almacenamiento'] > 0]

        if not df_util_centros.empty:
            df_util_centros['Utilización (%)'] = (df_util_centros['Cantidad'] / df_util_centros['Capacidad_Almacenamiento']) * 100
            df_util_centros = df_util_centros.sort_values(by='Utilización (%)', ascending=False)
            
            fig_util_c = px.bar(df_util_centros, y='Centro', x='Utilización (%)',  
                                text=df_util_centros['Utilización (%)'].apply(lambda x: f'{x:.1f}%'),
                                title='Utilización por Centro',
                                hover_data=['Cantidad', 'Capacidad_Almacenamiento'])
            fig_util_c.update_layout(xaxis_range=[0, 110])
            st.plotly_chart(fig_util_c, use_container_width=True)
        else:
            st.info("No hay datos de utilización de centros para la selección actual.")
    st.markdown("---")

    #Desglose de Costos
    st.header("Desglose de Costos de la Red (Filtrado)")
    
    # Recalcular los costos basados en los filtros
    # df_costos_pc_unicos y df_costos_cj_unicos AHORA VIENEN DEL SESSION STATE (YA TRANSFORMADOS)
    
    # Aplicar filtros de producto a los DFs de costos también
    if producto_seleccionado != "Todos":
        df_costos_pc_filt = df_costos_pc_unicos[df_costos_pc_unicos['Producto'] == producto_seleccionado]
        df_costos_cj_filt = df_costos_cj_unicos[df_costos_cj_unicos['Producto'] == producto_seleccionado]
    else:
        df_costos_pc_filt = df_costos_pc_unicos
        df_costos_cj_filt = df_costos_cj_unicos
        
    df_prod_cost = pd.merge(df_pc_filt, df_plantas_full[['Planta', 'Producto', 'Costo_Produccion']].drop_duplicates(), on=['Planta', 'Producto'], how='left')
    costo_produccion_calc = (df_prod_cost['Cantidad'] * df_prod_cost['Costo_Produccion']).sum()

    df_pc_cost = pd.merge(df_pc_filt, df_costos_pc_filt, on=['Planta', 'Centro', 'Producto'], how='left')
    costo_pc_calc = (df_pc_cost['Cantidad'] * df_pc_cost['Costo_Plant_Centro']).sum()

    df_cj_cost = pd.merge(df_cj_filt, df_costos_cj_filt, on=['Centro', 'Cliente', 'Producto'], how='left')
    costo_cj_calc = (df_cj_cost['Cantidad'] * df_cj_cost['Costo_Centro_Cliente']).sum()

    df_costos_pie = pd.DataFrame({
        'Componente de Costo': ['Producción', 'Transporte (Planta-Centro)', 'Transporte (Centro-Cliente)'],
        'Costo': [costo_produccion_calc, costo_pc_calc, costo_cj_calc]
    })
    
    df_costos_pie = df_costos_pie[df_costos_pie['Costo'] > 0]

    if not df_costos_pie.empty:
        fig_pie_costos = px.pie(df_costos_pie, names='Componente de Costo', values='Costo',  
                                title='Desglose de Costos Totales (Filtrado)',
                                hole=0.3)
        # Mover etiquetas afuera
        fig_pie_costos.update_traces(textposition='outside', textinfo='percent+label+value')
        st.plotly_chart(fig_pie_costos, use_container_width=True)
    else:
        st.info("No hay desglose de costos para la selección actual.")
    st.markdown("---")

    # Paretos
    st.header("Análisis de Costos por Cliente (Top 25)")
    st.caption(f"Muestra los 25 clientes principales para cada componente de costo. Todos los costos (excepto C-J) se asignan en función del flujo del cliente.")

    #  Lógica de Asignación de Costos  
    
    # 1. Costo_CJ (Centro-Cliente) - Directo
    # df_costos_cj_filt YA ESTÁ FILTRADO POR PRODUCTO
    df_cj_cost = pd.merge(df_cj_filt, df_costos_cj_filt, on=['Centro', 'Cliente', 'Producto'], how='left')
    df_cj_cost['Costo_CJ_Calc'] = df_cj_cost['Cantidad'] * df_cj_cost['Costo_Centro_Cliente'].fillna(0)
    costos_cj_por_cliente = df_cj_cost.groupby('Cliente')['Costo_CJ_Calc'].sum().reset_index()

    # 2. Base de asignación (Cuota de cliente por Centro-Producto)
    flujo_out_por_centro_prod = df_cj_filt.groupby(['Centro', 'Producto'])['Cantidad'].sum().reset_index().rename(columns={'Cantidad': 'Total_Flujo_Out'})
    df_cj_share = pd.merge(df_cj_filt, flujo_out_por_centro_prod, on=['Centro', 'Producto'], how='left')
    df_cj_share['Total_Flujo_Out'] = df_cj_share['Total_Flujo_Out'].replace(0, 1) # Evitar división por cero
    df_cj_share['Pct_Share'] = df_cj_share['Cantidad'] / df_cj_share['Total_Flujo_Out']

    # 3. Costo_Prod (Producción) - Asignado
    df_prod_cost_base = pd.merge(df_pc_filt, df_plantas_full[['Planta', 'Producto', 'Costo_Produccion']].drop_duplicates(), on=['Planta', 'Producto'], how='left')
    df_prod_cost_base['Costo_Prod_Calc'] = df_prod_cost_base['Cantidad'] * df_prod_cost_base['Costo_Produccion'].fillna(0)
    costo_prod_por_centro_prod = df_prod_cost_base.groupby(['Centro', 'Producto'])['Costo_Prod_Calc'].sum().reset_index()
    
    df_prod_alloc = pd.merge(df_cj_share, costo_prod_por_centro_prod, on=['Centro', 'Producto'], how='left')
    df_prod_alloc['Costo_Prod_Allocated'] = df_prod_alloc['Pct_Share'] * df_prod_alloc['Costo_Prod_Calc'].fillna(0)
    costos_prod_por_cliente = df_prod_alloc.groupby('Cliente')['Costo_Prod_Allocated'].sum().reset_index()

    # 4. Costo_PC (Planta-Centro) - Asignado
    # df_costos_pc_filt YA ESTÁ FILTRADO POR PRODUCTO
    df_pc_cost_base = pd.merge(df_pc_filt, df_costos_pc_filt, on=['Planta', 'Centro', 'Producto'], how='left')
    df_pc_cost_base['Costo_PC_Calc'] = df_pc_cost_base['Cantidad'] * df_pc_cost_base['Costo_Plant_Centro'].fillna(0)
    costo_pc_por_centro_prod = df_pc_cost_base.groupby(['Centro', 'Producto'])['Costo_PC_Calc'].sum().reset_index()

    df_pc_alloc = pd.merge(df_cj_share, costo_pc_por_centro_prod, on=['Centro', 'Producto'], how='left')
    df_pc_alloc['Costo_PC_Allocated'] = df_pc_alloc['Pct_Share'] * df_pc_alloc['Costo_PC_Calc'].fillna(0)
    costos_pc_por_cliente = df_pc_alloc.groupby('Cliente')['Costo_PC_Allocated'].sum().reset_index()

    # 5. Combinar todo
    df_final_costos = pd.merge(costos_cj_por_cliente, costos_prod_por_cliente, on='Cliente', how='outer')
    df_final_costos = pd.merge(df_final_costos, costos_pc_por_cliente, on='Cliente', how='outer')
    df_final_costos = df_final_costos.fillna(0)
    df_final_costos['Costo_Total'] = df_final_costos['Costo_CJ_Calc'] + df_final_costos['Costo_Prod_Allocated'] + df_final_costos['Costo_PC_Allocated']
    

    # Gráficos de Pareto

    def crear_pareto(df_base, columna_costo, titulo, color_barra):
        df_pareto = df_base[['Cliente', columna_costo]].copy()
        df_pareto = df_pareto[df_pareto[columna_costo] > 0.01].sort_values(by=columna_costo, ascending=False)
        
        if not df_pareto.empty:
            df_pareto['Pct'] = df_pareto[columna_costo] / df_pareto[columna_costo].sum()
            df_pareto['Pct_Acumulado'] = df_pareto['Pct'].cumsum()
            df_pareto_top25 = df_pareto.head(25)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_pareto_top25['Cliente'], y=df_pareto_top25[columna_costo], name='Costo', marker_color=color_barra), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_pareto_top25['Cliente'], y=df_pareto_top25['Pct_Acumulado'], name='Acumulado', mode='lines+markers', line_color=color_barra), secondary_y=True)
            fig.add_hline(y=0.8, line_dash="dot", secondary_y=True, line_color="gray")
            fig.update_layout(
                title_text=titulo,
                yaxis_title="Costo ($)",
                yaxis_tickformat="$,.0f",
                yaxis2_title="Pct. Acumulado",
                yaxis2_tickformat=".0%",
                height=400,
                margin=dict(t=50, l=10, r=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No hay datos para '{titulo}'.")

    col1, col2 = st.columns(2)
    with col1:
        crear_pareto(df_final_costos, 'Costo_Total', 'Top 25 Clientes por Costo Total (Asignado)', '#0047AB') # Azul
    with col2:
        crear_pareto(df_final_costos, 'Costo_CJ_Calc', 'Top 25 Clientes por Costo C-J (Directo)', '#d62728') # Rojo

    col3, col4 = st.columns(2)
    with col3:
        crear_pareto(df_final_costos, 'Costo_PC_Allocated', 'Top 25 Clientes por Costo P-C (Asignado)', '#ff7f0e') # Naranja
    with col4:
        crear_pareto(df_final_costos, 'Costo_Prod_Allocated', 'Top 25 Clientes por Costo Producción (Asignado)', '#2ca02c') # Verde

    st.markdown("---")
    
    # Scatter
    st.header("Análisis de Clientes Menos Rentables (25% de Menor Demanda)")
    
    if cliente_seleccionado != "Todos":
        st.info(f"Ha seleccionado un solo cliente ({cliente_seleccionado}). Para ver el análisis de clientes problemáticos, cambie el filtro 'Cliente' a 'Todos'.")
    else:
        df_scatter = df_final_costos.copy()
        df_cantidad_total = df_cj_filt.groupby('Cliente')['Cantidad'].sum().reset_index().rename(columns={'Cantidad': 'Cantidad_Total'})
        df_scatter = pd.merge(df_scatter, df_cantidad_total, on='Cliente', how='left').fillna(0)

        df_scatter = df_scatter[(df_scatter['Cantidad_Total'] > 0) & (df_scatter['Costo_Total'] > 0)]

        if not df_scatter.empty and len(df_scatter) > 1:
            q1_cantidad = df_scatter['Cantidad_Total'].quantile(0.25)
            df_scatter_bottom25 = df_scatter[df_scatter['Cantidad_Total'] <= q1_cantidad].copy()
            if not df_scatter_bottom25.empty:
                fig_scatter = px.scatter(
                    df_scatter_bottom25,  
                    x='Cantidad_Total',
                    y='Costo_Total',  
                    text='Cliente',  
                    title=f"Clientes con Demanda Menor o Igual a {q1_cantidad:,.0f} unidades",
                    hover_data=['Costo_Prod_Allocated', 'Costo_PC_Allocated', 'Costo_CJ_Calc'] # Añadido para más detalle
                )
                fig_scatter.update_layout(
                    xaxis_title="CANTIDAD TOTAL (Baja Demanda)",
                    yaxis_title="COSTO TOTAL ASIGNADO (Alto = Más Caro)",
                    xaxis_tickformat=",.0f",
                    yaxis_tickformat="$,.0f",
                    height=600
                )
                fig_scatter.update_traces(textposition='top center')
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No hay clientes en el 25% inferior para la selección actual (o no hay datos).")
        else:
            st.info("No hay suficientes datos de clientes para la selección actual para realizar este análisis.")

    # Tablas
    st.markdown("---")  
    with st.expander("Ver Tablas de Envío Detalladas (Filtradas)"):
        st.caption("Estas tablas se actualizan con todos los filtros seleccionados.")
        tbl1, tbl2 = st.columns(2)
        with tbl1:
            st.subheader(f"Planta -> Centro")
            st.dataframe(df_pc_filt)
        with tbl2:
            st.subheader(f"Centro -> Cliente")
            st.dataframe(df_cj_filt)
