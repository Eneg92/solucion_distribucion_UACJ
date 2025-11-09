import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyomo.environ import *
import io

st.set_page_config(layout="wide")

def resolver_modelo_distribucion(df_plantas, df_centros, df_clientes, df_costos, df_productos):
    try:
        # Calcular totales
        total_demanda_requerida = df_clientes['Demanda'].sum()
        total_capacidad_produccion = df_plantas['Capacidad_Produccion'].sum()
        total_capacidad_almacenamiento = df_centros['Capacidad_Almacenamiento'].sum()

        # Conjuntos
        P = df_plantas['Planta'].unique()
        C = df_centros['Centro'].unique()
        J = df_clientes['Cliente'].unique()
        K = df_productos['Producto'].unique()

        # Parámetros
        demanda = df_clientes.set_index(['Cliente', 'Producto'])['Demanda'].to_dict()
        cap_prod = df_plantas.set_index(['Planta', 'Producto'])['Capacidad_Produccion'].to_dict()
        cost_prod = df_plantas.set_index(['Planta', 'Producto'])['Costo_Produccion'].to_dict()
        cap_alm = df_centros.set_index(['Centro', 'Producto'])['Capacidad_Almacenamiento'].to_dict()
        cost_pc_df = df_costos.drop_duplicates(subset=['Planta', 'Centro', 'Producto'])
        cost_pc = cost_pc_df.set_index(['Planta', 'Centro', 'Producto'])['Costo_Plant_Centro'].to_dict()
        cost_cj_df = df_costos.drop_duplicates(subset=['Centro', 'Cliente', 'Producto'])
        cost_cj = cost_cj_df.set_index(['Centro', 'Cliente', 'Producto'])['Costo_Centro_Cliente'].to_dict()

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

        # Función Objetivo (Sin penalización)
        def funcion_objetivo_rule(model):
            costo_produccion = sum(model.cost_prod[p, k] * model.x[p, c, k] 
                                    for p in model.P for c in model.C for k in model.K)
            costo_transporte_pc = sum(model.cost_pc[p, c, k] * model.x[p, c, k]
                                        for p in model.P for c in model.C for k in model.K)
            costo_transporte_cj = sum(model.cost_cj[c, j, k] * model.y[c, j, k]
                                        for c in model.C for j in model.J for k in model.K)
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
            return None, None, None, "Solver no encontrado"
            
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
            
            return kpi_data, df_x, df_y, None

        else:
            msg = f"No se encontró una solución óptima. Estado: {results.solver.status}, Condición: {results.solver.termination_condition}. (Esto puede ocurrir si la capacidad es insuficiente para la demanda)"
            return None, None, None, msg

    except Exception as e:
        return None, None, None, f"Error durante la optimización: {str(e)}"

# --- CAMBIO 1: Inicializar los nuevos dataframes ---
if 'model_run_success' not in st.session_state:
    st.session_state['model_run_success'] = False
if 'kpis' not in st.session_state:
    st.session_state['kpis'] = None
if 'df_pc' not in st.session_state:
    st.session_state['df_pc'] = None
if 'df_cj' not in st.session_state:
    st.session_state['df_cj'] = None
if 'df_costos' not in st.session_state:
    st.session_state['df_costos'] = None
if 'df_plantas' not in st.session_state:
    st.session_state['df_plantas'] = None
if 'df_centros' not in st.session_state:
    st.session_state['df_centros'] = None

# Barra lateral para carga y ejecución
st.sidebar.header("Panel de Control")

with st.sidebar.expander("1. Cargar Archivos de Datos", expanded=True):
    file_plantas = st.file_uploader("Cargar 'plantas.csv'", type="csv")
    file_centros = st.file_uploader("Cargar 'centros.csv'", type="csv")
    file_clientes = st.file_uploader("Cargar 'clientes.csv'", type="csv")
    file_costos = st.file_uploader("Cargar 'costos.csv'", type="csv")
    file_productos = st.file_uploader("Cargar 'productos.csv'", type="csv")

files_uploaded = [file_plantas, file_centros, file_clientes, file_costos, file_productos]
all_files_loaded = all(f is not None for f in files_uploaded)

if st.sidebar.button("Ejecutar Optimización", disabled=not all_files_loaded, type="primary"):
    if all_files_loaded:
        with st.spinner("Leyendo archivos y ejecutando optimización..."):
            try:
                # Resetear el puntero de los archivos antes de leer
                file_plantas.seek(0); df_plantas = pd.read_csv(file_plantas)
                file_centros.seek(0); df_centros = pd.read_csv(file_centros)
                file_clientes.seek(0); df_clientes = pd.read_csv(file_clientes)
                file_costos.seek(0); df_costos = pd.read_csv(file_costos)
                file_productos.seek(0); df_productos = pd.read_csv(file_productos)
                
                kpis, df_x, df_y, error_msg = resolver_modelo_distribucion(
                    df_plantas, df_centros, df_clientes, df_costos, df_productos
                )
                
                if kpis:
                    # --- CAMBIO 2: Guardar los dataframes originales ---
                    st.session_state['kpis'] = kpis
                    st.session_state['df_pc'] = df_x
                    st.session_state['df_cj'] = df_y
                    st.session_state['df_costos'] = df_costos
                    st.session_state['df_plantas'] = df_plantas
                    st.session_state['df_centros'] = df_centros
                    # --- Fin del Cambio ---
                    
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

# Lógica de visualización del dashboard
if not st.session_state['model_run_success']:
    
    # --- BLOQUE DE INFORMACIÓN DEL PROYECTO (SOLO BIENVENIDA) ---
    LOGO_FILE = "logo_uacj.png" 
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
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
    
    # --- MENSAJE DE BIENVENIDA ---
    st.info("Bienvenido. Por favor, cargue los 5 archivos de datos en el panel lateral y haga clic en 'Ejecutar Optimización' para ver el dashboard.")
    
    st.subheader("Archivos Requeridos:")
    st.markdown("""
    * **plantas.csv**: `Planta`, `Producto`, `Capacidad_Produccion`, `Costo_Produccion`
    * **centros.csv**: `Centro`, `Producto`, `Capacidad_Almacenamiento`
    * **clientes.csv**: `Cliente`, `Producto`, `Demanda`
    * **costos.csv**: `Planta`, `Centro`, `Producto`, `Cliente`, `Costo_Plant_Centro`, `Costo_Centro_Cliente`
    * **productos.csv**: `Producto`
    """)

else:
    
    st.markdown("<h1 style='text-align: center; color: #0047AB;'>📊 Dashboard de Optimización de Red Logística</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Cargar datos desde el estado
    kpis = st.session_state['kpis']
    df_pc_full = st.session_state['df_pc']
    df_cj_full = st.session_state['df_cj']
    # --- CAMBIO 3: Cargar los dataframes originales ---
    df_costos = st.session_state['df_costos']
    df_plantas_full = st.session_state['df_plantas']
    df_centros_full = st.session_state['df_centros']
    
    if df_costos is None or df_costos.empty or df_plantas_full is None or df_centros_full is None:
        st.error("No se pudieron cargar todos los datos para el análisis. Por favor, intente ejecutar de nuevo.")
        st.stop()
        
    df_costos_cj_unicos = df_costos[['Centro', 'Cliente', 'Producto', 'Costo_Centro_Cliente']].drop_duplicates()
    df_costos_pc_unicos = df_costos[['Planta', 'Centro', 'Producto', 'Costo_Plant_Centro']].drop_duplicates()

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

    # KPIs principales (Sin st.header, como pediste)
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
 
    # Gráficos de Medidor (Gauge) (Basados en grafica.py original)
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


    st.header("Desglose de Costos de la Red (Filtrado)")
    
    # Recalcular los costos basados en los filtros
    # 1. Costo de Producción
    df_prod_cost = pd.merge(df_pc_filt, df_plantas_full[['Planta', 'Producto', 'Costo_Produccion']].drop_duplicates(), on=['Planta', 'Producto'], how='left')
    costo_produccion_calc = (df_prod_cost['Cantidad'] * df_prod_cost['Costo_Produccion']).sum()

    # 2. Costo Planta -> Centro
    df_pc_cost = pd.merge(df_pc_filt, df_costos_pc_unicos, on=['Planta', 'Centro', 'Producto'], how='left')
    costo_pc_calc = (df_pc_cost['Cantidad'] * df_pc_cost['Costo_Plant_Centro']).sum()

    # 3. Costo Centro -> Cliente
    df_cj_cost = pd.merge(df_cj_filt, df_costos_cj_unicos, on=['Centro', 'Cliente', 'Producto'], how='left')
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
        
        fig_pie_costos.update_traces(textposition='outside', textinfo='percent+label+value')
        
        st.plotly_chart(fig_pie_costos, use_container_width=True)
    else:
        st.info("No hay desglose de costos para la selección actual.")
    
    st.markdown("---")

    st.header("Utilización de Infraestructura (Filtrado)")
    st.caption("Muestra la utilización real vs. la capacidad disponible para los filtros seleccionados.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Utilización de Plantas")
        # 1. Capacidad de Plantas (filtrada)
        cap_plantas = df_plantas_filt.groupby('Planta')['Capacidad_Produccion'].sum().reset_index()
        # 2. Producción Real (filtrada)
        prod_real = df_pc_filt.groupby('Planta')['Cantidad'].sum().reset_index()
        # 3. Mergear
        df_util_plantas = pd.merge(cap_plantas, prod_real, on='Planta', how='left').fillna(0)
        df_util_plantas = df_util_plantas[df_util_plantas['Capacidad_Produccion'] > 0]
        
        if not df_util_plantas.empty:
            df_util_plantas['Utilización (%)'] = (df_util_plantas['Cantidad'] / df_util_plantas['Capacidad_Produccion']) * 100
            df_util_plantas = df_util_plantas.sort_values(by='Utilización (%)', ascending=False)
            
            fig_util_p = px.bar(df_util_plantas, y='Planta', x='Utilización (%)', 
                                text=df_util_plantas['Utilización (%)'].apply(lambda x: f'{x:.1f}%'),
                                title='Utilización por Planta',
                                hover_data=['Cantidad', 'Capacidad_Produccion'])
            fig_util_p.update_layout(xaxis_range=[0, 110]) # Poner el rango hasta 110 para ver el 100%
            st.plotly_chart(fig_util_p, use_container_width=True)
        else:
            st.info("No hay datos de utilización de plantas para la selección actual.")

    with col2:
        st.subheader("Utilización de Centros (Flujo)")
        # 1. Capacidad de Centros (filtrada)
        cap_centros = df_centros_filt.groupby('Centro')['Capacidad_Almacenamiento'].sum().reset_index()
        # 2. Flujo Real (filtrado)
        flujo_real = df_pc_filt.groupby('Centro')['Cantidad'].sum().reset_index()
        # 3. Mergear
        df_util_centros = pd.merge(cap_centros, flujo_real, on='Centro', how='left').fillna(0)
        df_util_centros = df_util_centros[df_util_centros['Capacidad_Almacenamiento'] > 0]

        if not df_util_centros.empty:
            df_util_centros['Utilización (%)'] = (df_util_centros['Cantidad'] / df_util_centros['Capacidad_Almacenamiento']) * 100
            df_util_centros = df_util_centros.sort_values(by='Utilización (%)', ascending=False)
            
            fig_util_c = px.bar(df_util_centros, y='Centro', x='Utilización (%)', 
                                text=df_util_centros['Utilización (%)'].apply(lambda x: f'{x:.1f}%'),
                                title='Utilización por Centro',
                                hover_data=['Cantidad', 'Capacidad_Almacenamiento'])
            fig_util_c.update_layout(xaxis_range=[0, 110]) # Poner el rango hasta 110 para ver el 100%
            st.plotly_chart(fig_util_c, use_container_width=True)
        else:
            st.info("No hay datos de utilización de centros para la selección actual.")

    st.markdown("---")


    st.header(f"Análisis de Clientes: Volumen vs. Costo (Filtrado)") 
    st.caption(f"Mostrando: Prod ({producto_seleccionado}) | Planta ({planta_seleccionada}) | Centro ({centro_seleccionado}) | Cliente ({cliente_seleccionado})")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pareto de Demanda (Volumen)")
        df_pareto_demanda = df_cj_filt.groupby('Cliente')['Cantidad'].sum().reset_index()
        df_pareto_demanda = df_pareto_demanda[df_pareto_demanda['Cantidad'] > 0].sort_values(by='Cantidad', ascending=False)
        
        if not df_pareto_demanda.empty and df_pareto_demanda['Cantidad'].sum() > 0:
            df_pareto_demanda['Pct'] = df_pareto_demanda['Cantidad'] / df_pareto_demanda['Cantidad'].sum()
            df_pareto_demanda['Pct_Acumulado'] = df_pareto_demanda['Pct'].cumsum()
            df_pareto_demanda_top25 = df_pareto_demanda.head(25)
            
            fig_pareto_d = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto_d.add_trace(go.Bar(x=df_pareto_demanda_top25['Cliente'], y=df_pareto_demanda_top25['Cantidad'], name='Demanda'), secondary_y=False)
            fig_pareto_d.add_trace(go.Scatter(x=df_pareto_demanda_top25['Cliente'], y=df_pareto_demanda_top25['Pct_Acumulado'], name='Acumulado', mode='lines+markers'), secondary_y=True)
            fig_pareto_d.add_hline(y=0.8, line_dash="dot", secondary_y=True, line_color="gray")
            fig_pareto_d.update_layout(
                title_text="Top 25 Clientes por Volumen de Demanda",
                yaxis_title="Cantidad de Unidades",
                yaxis2_title="Porcentaje Acumulado",
                yaxis2_tickformat=".0%",
                height=400,
                margin=dict(t=50, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_pareto_d, use_container_width=True)
        else:
            st.info("No hay datos de demanda por cliente para la selección actual.")

    with col2:
        st.subheader("Pareto de Costo (Rentabilidad)")
        
        df_costo_base = pd.merge(df_cj_filt, df_costos_cj_unicos, on=['Centro', 'Cliente', 'Producto'], how='left')
        df_costo_base['Costo_Centro_Cliente'] = df_costo_base['Costo_Centro_Cliente'].fillna(0)
        df_costo_base['Costo_Envio'] = df_costo_base['Cantidad'] * df_costo_base['Costo_Centro_Cliente']
        
        df_pareto_costo = df_costo_base.groupby('Cliente').agg(Costo_Total=('Costo_Envio', 'sum')).reset_index()
        df_pareto_costo = df_pareto_costo[df_pareto_costo['Costo_Total'] > 0].sort_values(by='Costo_Total', ascending=False)
        
        if not df_pareto_costo.empty and df_pareto_costo['Costo_Total'].sum() > 0:
            df_pareto_costo['Pct'] = df_pareto_costo['Costo_Total'] / df_pareto_costo['Costo_Total'].sum()
            df_pareto_costo['Pct_Acumulado'] = df_pareto_costo['Pct'].cumsum()
            df_pareto_costo_top25 = df_pareto_costo.head(25)
            
            fig_pareto_c = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto_c.add_trace(go.Bar(x=df_pareto_costo_top25['Cliente'], y=df_pareto_costo_top25['Costo_Total'], name='Costo', marker_color='red'), secondary_y=False)
            fig_pareto_c.add_trace(go.Scatter(x=df_pareto_costo_top25['Cliente'], y=df_pareto_costo_top25['Pct_Acumulado'], name='Acumulado', mode='lines+markers', line_color='red'), secondary_y=True)
            fig_pareto_c.add_hline(y=0.8, line_dash="dot", secondary_y=True, line_color="gray")
            fig_pareto_c.update_layout(
                title_text="Top 25 Clientes por Costo de Envío",
                yaxis_title="Costo Total de Envío ($)",
                yaxis_tickformat="$,.0f",
                yaxis2_title="Porcentaje Acumulado",
                yaxis2_tickformat=".0%",
                height=400,
                margin=dict(t=50, l=10, r=10, b=10)
            )
            st.plotly_chart(fig_pareto_c, use_container_width=True)
        else:
            st.info("No hay datos de costo por cliente para la selección actual.")
    
    st.markdown("---")
    
    # Análisis de Clientes Problemáticos (Basado en grafica.py original)
    st.header("Análisis de Clientes Menos Rentables (25% de Menor Demanda)")
    
    if cliente_seleccionado != "Todos":
        st.info(f"Ha seleccionado un solo cliente ({cliente_seleccionado}). Para ver el análisis de clientes problemáticos, cambie el filtro 'Cliente' a 'Todos'.")
    else:
        df_scatter_base_data = df_cj_working
        
        if planta_seleccionada != "Todos":
            centros_de_planta = set(df_pc_working[df_pc_working['Planta'] == planta_seleccionada]['Centro'].unique()) if not df_pc_working.empty else set()
            df_scatter_base_data = df_scatter_base_data[df_scatter_base_data['Centro'].isin(centros_de_planta)]
        if centro_seleccionado != "Todos":
            df_scatter_base_data = df_scatter_base_data[df_scatter_base_data['Centro'] == centro_seleccionado]
            
        df_scatter_base = pd.merge(df_scatter_base_data, df_costos_cj_unicos, on=['Centro', 'Cliente', 'Producto'], how='left')
        
        df_scatter_base['Costo_Centro_Cliente'] = df_scatter_base['Costo_Centro_Cliente'].fillna(0)
        
        df_scatter_base['Costo_Envio'] = df_scatter_base['Cantidad'] * df_scatter_base['Costo_Centro_Cliente']

        df_scatter = df_scatter_base.groupby('Cliente').agg(
            Cantidad_Total=('Cantidad', 'sum'),
            Costo_Envio_Total=('Costo_Envio', 'sum')
        ).reset_index()

        df_scatter = df_scatter[(df_scatter['Cantidad_Total'] > 0) & (df_scatter['Costo_Envio_Total'] > 0)]

        if not df_scatter.empty and len(df_scatter) > 1:
            q1_cantidad = df_scatter['Cantidad_Total'].quantile(0.25)
            df_scatter_bottom25 = df_scatter[df_scatter['Cantidad_Total'] <= q1_cantidad].copy()
            if not df_scatter_bottom25.empty:
                fig_scatter = px.scatter(
                    df_scatter_bottom25, 
                    x='Cantidad_Total',
                    y='Costo_Envio_Total',
                    text='Cliente', 
                    title=f"Clientes con Demanda Menor o Igual a {q1_cantidad:,.0f} unidades"
                )
                fig_scatter.update_layout(
                    xaxis_title="CANTIDAD TOTAL (Baja Demanda)",
                    yaxis_title="COSTO TOTAL DE ENVÍO (Alto = Más Caro)",
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

    # Tablas de Detalle 
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
