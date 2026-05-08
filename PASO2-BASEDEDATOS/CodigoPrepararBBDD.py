
"""
# Para hacer la consulta masiva a JPL Horizons y preparar la base de datos con los vectores de posición y velocidad

import pandas as pd
from astroquery.jplhorizons import Horizons
from tqdm import tqdm

# 1. Cargar el archivo CSV
archivo_entrada = 'bbddAntigua/asteroid_dataset_20251019.csv'
df_completo = pd.read_csv(archivo_entrada)

# 2. Seleccionar solo las primeras 50,000 interacciones
df = df_completo.head(50000).copy()

# Listas para almacenar los resultados
vector_ubicacion_tierra = []
vector_ubicacion_asteroide = []
vector_velocidad_asteroide = []

print(f"Iniciando consulta de {len(df)} registros a JPL Horizons...")

try:
    # Optimizamos: Consultamos la Tierra una sola vez (ahorra miles de peticiones)
    q_tierra = Horizons(id='399', location='@sun', epochs=None).vectors()
    v_tierra_constante = [q_tierra['x'][0], q_tierra['y'][0], q_tierra['z'][0]]
except Exception as e:
    print(f"Error al obtener datos de la Tierra: {e}")
    v_tierra_constante = None

# 3. Bucle de consulta para los asteroides
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        # Limpiar el nombre del asteroide
        nombre_ast = str(row['asteroid_fullname']).replace('(', '').replace(')', '').strip()
        
        # Consulta del Asteroide respecto al Sol
        q_ast = Horizons(id=nombre_ast, location='@sun', epochs=None).vectors()
        
        # Extraer vectores de posición y velocidad
        v_pos_ast = [q_ast['x'][0], q_ast['y'][0], q_ast['z'][0]]
        v_vel_ast = [q_ast['vx'][0], q_ast['vy'][0], q_ast['vz'][0]]
        
        vector_ubicacion_tierra.append(v_tierra_constante)
        vector_ubicacion_asteroide.append(v_pos_ast)
        vector_velocidad_asteroide.append(v_vel_ast)
        
    except Exception:
        # En caso de error (nombre no encontrado, timeout, etc.)
        vector_ubicacion_tierra.append(None)
        vector_ubicacion_asteroide.append(None)
        vector_velocidad_asteroide.append(None)

# 4. Añadir las columnas al DataFrame segmentado
df['vector_posicion_tierra'] = vector_ubicacion_tierra
df['vector_posicion_asteroide'] = vector_ubicacion_asteroide
df['vector_velocidad_asteroide'] = vector_velocidad_asteroide

# 5. Guardar el resultado (solo los 50k procesados)
nombre_salida = 'asteroides_50k_con_vectores.csv'
df.to_csv(nombre_salida, index=False)

print(f"\n¡Listo! Se han procesado las primeras 50,000 filas y se guardaron en: {nombre_salida}")"""


"""
# PARA PROBAR QUE ME ENCUENTRA DIAMETROS

from astroquery.jplhorizons import Horizons
import numpy as np
import pandas as pd

test_asteroids = ['2020 AC1', '2019 YK', '2013 EC20', '2020 AP3', '2011 YE40']

resultados = []

print("Consultando JPL Horizons via Ephemerides (Quantity 9)...")

for name in test_asteroids:
    try:

        # Usamos el prefijo 'DES=' por seguridad con designaciones provisionales
        obj = Horizons(id=f"DES={name};", location='@sun', epochs=2460000.5)
        
        # Pedimos la cantidad 9 (incluye la magnitud absoluta H)
        eph = obj.ephemerides(quantities='9')
        
        # Extraemos H (Magnitud Absoluta)
        h_mag = eph['H'][0]
        
        # Aplicamos la fórmula científica de estimación de diámetro
        # Albedo 0.15 es el estándar para asteroides de este tipo (S-type/rocosos)
        albedo = 0.15
        d_km = (1329 / np.sqrt(albedo)) * (10**(-0.2 * h_mag))
        
        resultados.append({
            'Asteroide': name,
            'H_Mag': h_mag,
            'Diámetro_Estimado_km': round(d_km, 4),
            'Estado': 'Éxito'
        })
        
    except Exception as e:
        resultados.append({
            'Asteroide': name,
            'H_Mag': np.nan,
            'Diámetro_Estimado_km': np.nan,
            'Estado': f'Error: {str(e)[:30]}'
        })

# Mostrar resultados
df_res = pd.DataFrame(resultados)
print("\n", df_res)"""
"""
# Funcion para aplicar lo de arriba a toda la base de datos

import pandas as pd
import numpy as np
from astroquery.jplhorizons import Horizons
from tqdm import tqdm
import time

def procesar_asteroides_con_query(input_file, output_file):
    # 1. Cargar la base de datos original
    print(f"Cargando {input_file}...")
    df = pd.read_csv(input_file)
    
    # Verificamos que exista la columna de identificación (ajusta el nombre si es distinto, ej: 'full_name' o 'name')
    columna_nombre = 'asteroid_fullname'
    
    if columna_nombre not in df.columns:
        print(f"Error: No existe la columna '{columna_nombre}' en el CSV.")
        return

    # Inicializamos columnas nuevas
    df['H'] = np.nan
    df['Diametro'] = np.nan
    
    albedo = 0.15
    constante = 1329 / np.sqrt(albedo)
    
    print(f"Iniciando consultas a JPL Horizons para {len(df)} asteroides...")

    # Bucle de consulta con barra de progreso
    for i, row in tqdm(df.iterrows(), total=len(df)):
        nombre_ast = str(row['asteroid_fullname']).replace('(', '').replace(')', '').strip()
        
        
        try:
            # Consulta a JPL Horizons
            obj = Horizons(id=f"DES={nombre_ast};", location='@sun', epochs=2460000.5)
            eph = obj.ephemerides(quantities='9')
            
            h_mag = eph['H'][0]
            
            if h_mag is not None:
                # Calcular diámetro
                d_km = constante * (10**(-0.2 * h_mag))
                
                # Asignar valores al DataFrame
                df.at[i, 'H'] = h_mag
                df.at[i, 'Diametro'] = round(d_km, 4)
                
        except Exception as e:
            # Si falla un objeto, saltamos al siguiente para no detener todo el proceso
            continue

        # 3. Guardado de seguridad cada 10000 registros
        if i % 10000 == 0 and i > 0:
            df.to_csv(output_file, index=False)

    # 4. Guardado final
    df.to_csv(output_file, index=False)
    print(f"\n¡Proceso finalizado! Archivo guardado como: {output_file}")

# --- EJECUCIÓN ---
# Asegúrate de que el nombre del archivo de entrada coincida con el tuyo
archivo_entrada = 'asteroides_50k_con_vectores.csv' 
archivo_salida = 'asteroides_50k_FINAL_CON_DIAMETRO.csv'

# Si el archivo es un CSV, descomenta la línea de abajo:
procesar_asteroides_con_query(archivo_entrada, archivo_salida)
"""

"""
# Para hacer limpito

import pandas as pd
import numpy as np
from astroquery.jplhorizons import Horizons
from tqdm import tqdm
from datetime import datetime, timedelta

# 1. Cargar la base de datos
input_file = 'asteroides_50k_FINAL_CON_DIAMETRO.csv'
output_file = 'dataset_preparado.csv'

print(f"Cargando {input_file}...")
df = pd.read_csv(input_file)

# Contenedor para las nuevas columnas
data_ia = []

print("Procesando: Asteroide (T-30 días) y Tierra (Día del Close Approach)...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # --- MANEJO DE FECHAS ---
        fecha_ca_str = row['close_approach_date']
        fecha_ca_dt = datetime.strptime(fecha_ca_str, '%Y-%m-%d %H:%M:%S')
        
        # Fecha para el Asteroide (30 días antes)
        fecha_ast_pre = (fecha_ca_dt - timedelta(days=30)).strftime('%Y-%m-%d %H:%M')
        # Fecha para la Tierra (El día del encuentro)
        fecha_tierra_ca = fecha_ca_str # Usamos la original
        
        nombre_ast = f"DES={row['asteroid_designation'].strip()};")

        # --- CONSULTA 1: LA TIERRA EN EL ACERCAMIENTO ---
        vt = Horizons(id='399', location='@sun', epochs=fecha_tierra_ca).vectors()
        
        # --- CONSULTA 2: EL ASTEROIDE 30 DÍAS ANTES ---
        va = Horizons(id=nombre_ast, location='@sun', epochs=fecha_ast_pre).vectors()
        
        data_ia.append({
            # Tierra en el futuro (Día del Close Approach)
            'tierra_ca_x': vt['x'][0], 
            'tierra_ca_y': vt['y'][0], 
            'tierra_ca_z': vt['z'][0],
            # Asteroide en el pasado (30 días antes)
            'ast_pre_x': va['x'][0], 
            'ast_pre_y': va['y'][0], 
            'ast_pre_z': va['z'][0],
            'ast_pre_vx': va['vx'][0], 
            'ast_pre_vy': va['vy'][0], 
            'ast_pre_vz': va['vz'][0]
        })
        
    except Exception:
        data_ia.append({k: np.nan for k in [
            'tierra_ca_x','tierra_ca_y','tierra_ca_z',
            'ast_pre_x','ast_pre_y','ast_pre_z',
            'ast_pre_vx','ast_pre_vy','ast_pre_vz'
        ]})

    # Guardado de seguridad cada 500
    if i % 500 == 0 and i > 0:
        pd.concat([df.iloc[:len(data_ia)], pd.DataFrame(data_ia)], axis=1).to_csv(output_file, index=False)

# 2. Unión y Limpieza final
df_nuevo = pd.concat([df, pd.DataFrame(data_ia)], axis=1)

columnas_sucias = [
    'vector_posicion_tierra', 'vector_posicion_asteroide', 'vector_velocidad_asteroide',
    'sentry_impact_prob', 'sentry_torino_scale', 'sentry_palermo_scale', 'sentry_diameter_km'
]
df_final = df_nuevo.drop(columns=columnas_sucias, errors='ignore')

# 3. Guardar
df_final.to_csv(output_file, index=False)
print(f"\n¡Dataset preparado! Archivo: {output_file}")
"""

























import pandas as pd
import numpy as np
from astroquery.jplhorizons import Horizons
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from astropy.time import Time


# --- CONFIGURACIÓN ---
input_file = 'bbddAntigua/asteroid_dataset_20251019.csv'
output_file = 'dataset_final.csv'
n_registros = 50000  # Número de asteroides a procesar
albedo = 0.15
constante_diametro = 1329 / np.sqrt(albedo)

# Cargar la base de datos original
print(f"Cargando {input_file}...")
df_original = pd.read_csv(input_file)
df = df_original.head(n_registros).copy()

# Contenedor para los nuevos datos
data_acumulada = []

print(f"Iniciando procesamiento de {n_registros} registros...")
print("Consultando: Diámetro, Vectores Asteroide (T-30) y Tierra (T-0)")

# Bucle principal
for i, row in tqdm(df.iterrows(), total=len(df)):

    # 1. Convertimos la fecha del CSV a objeto datetime
    fecha_ca_dt = pd.to_datetime(row['close_approach_date'])

    # 2. Calculamos las fechas que necesitamos
    fecha_tierra_dt = fecha_ca_dt
    fecha_ast_pre_dt = fecha_ca_dt - timedelta(days=30)

    # Conversion a JulianDate para consultas a Horizons
    jd_tierra = Time(fecha_tierra_dt).jd
    jd_ast_pre = Time(fecha_ast_pre_dt).jd

    try:
        
        nombre_ast = str(row['asteroid_fullname']).replace('(', '').replace(')', '').strip()

        # --- CONSULTA A: DIÁMETRO Y MAGNITUD H ---
        # Usamos una fecha estándar para la magnitud absoluta
        obj_h = Horizons(id=nombre_ast, location='@sun', epochs=2460000.5)
        eph = obj_h.ephemerides(quantities='9')
        h_mag = float(eph['H'][0])
        d_km = constante_diametro * (10**(-0.2 * h_mag))

        # --- CONSULTA B: LA TIERRA EN EL CLOSE APPROACH (T-0) ---
        obj_t = Horizons(id='399', location='@sun', epochs=jd_tierra)
        vt = obj_t.vectors()
        
        # --- CONSULTA C: EL ASTEROIDE 30 DÍAS ANTES (T-30) ---
        obj_a = Horizons(id=nombre_ast, location='@sun', epochs=jd_ast_pre)
        va = obj_a.vectors()
        
        # --- GUARDAR TODO EN EL DICCIONARIO ---
        data_acumulada.append({
            'H_calculado': h_mag,
            'diametro_km': round(d_km, 4),
            'tierra_ca_x': vt['x'][0], 
            'tierra_ca_y': vt['y'][0], 
            'tierra_ca_z': vt['z'][0],
            'ast_pre_x': va['x'][0], 
            'ast_pre_y': va['y'][0], 
            'ast_pre_z': va['z'][0],
            'ast_pre_vx': va['vx'][0], 
            'ast_pre_vy': va['vy'][0], 
            'ast_pre_vz': va['vz'][0]
        })
        
    except Exception:
        # Si algo falla, llenamos con NaN para no perder el índice
        data_acumulada.append({k: np.nan for k in [
            'H_calculado', 'diametro_km', 'tierra_ca_x', 'tierra_ca_y', 'tierra_ca_z',
            'ast_pre_x', 'ast_pre_y', 'ast_pre_z', 'ast_pre_vx', 'ast_pre_vy', 'ast_pre_vz'
        ]})

    # Guardado de seguridad cada 500 filas
    if i % 500 == 0 and i > 0:
        df_temp = pd.concat([df.iloc[:len(data_acumulada)].reset_index(drop=True), 
                             pd.DataFrame(data_acumulada)], axis=1)
        df_temp.to_csv(output_file, index=False)

# Finalización y Limpieza
df_final = pd.concat([df.reset_index(drop=True), pd.DataFrame(data_acumulada)], axis=1)

# Eliminamos columnas que puedan genenrar ruido, también las columnas que puedan dar pistas al modelo sobre la cercanía o peligrosidad del asteroide.
columnas_a_eliminar = [
    'sentry_diameter_km ', 'days_until_aproach', 'risk_score', 'thread_category', 
    'panic_verdict', 'is_past_event', 'is_future_event'
]

df_final = df_final.drop(columns=columnas_a_eliminar, errors='ignore')

# Guardado final
df_final.to_csv(output_file, index=False)

print(f"\n¡Proceso finalizado con éxito!")
print(f"Archivo generado: {output_file}")






"""
#Me daba error, esto es para probar
from astroquery.jplhorizons import Horizons
import pandas as pd
from astropy.time import Time

# Datos de tu ejemplo
nombre_original = "2020 AY1"
nombre_limpio = nombre_original.replace('(', '').replace(')', '').strip()

print(f"Probando con: '{nombre_limpio}'")

try:
    # 1. Magnitud H (esto sabemos que funciona)
    obj = Horizons(id=nombre_limpio, location='@sun', epochs=2460000.5, id_type='smallbody')
    eph = obj.ephemerides(quantities='9')
    print("✅ Magnitud H:", eph['H'][0])

    # 2. Vectores usando JULIAN DATE (La solución definitiva)
    # Convertimos tu string a objeto Time y luego a jd (Julian Date)
    fecha_str = '2020-01-01 00:54:00'
    t = Time(fecha_str)
    fecha_jd = t.jd
    
    print(f"Consultando con fecha JD: {fecha_jd}")
    
    # IMPORTANTE: Pasamos el número directamente a epochs
    vec = Horizons(id=nombre_limpio, location='@sun', epochs=fecha_jd, id_type='smallbody').vectors()
    
    if len(vec) > 0:
        print("✅ Vector X:", vec['x'][0])
        print("\n¡POR FIN! Usar Julian Dates ha saltado el error de formato de la API.")
    else:
        print("❌ La tabla volvió vacía.")

except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    print("\nSi esto falla, el problema es que Horizons no tiene la órbita calculada para esa fecha específica.")"""