import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline, AutoImageProcessor
from roboflow import Roboflow
import requests
import warnings
import os

warnings.filterwarnings("ignore")

# Configuración inicial de la página
st.set_page_config(layout="wide", page_title="Biotech - Detector de Consumo", page_icon="🍊")

# ==========================================
# 1. ESTILOS CSS PERSONALIZADOS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Poppins:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 45%, #fff7ed 100%);
        color: #0f172a;
        font-family: 'Poppins', sans-serif;
    }

    .biotech-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(45deg, #0284c7, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .biotech-subtitle {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
    }

    .instruction-card {
        background-color: #ffffff;
        border-left: 8px solid #f97316;
        border-radius: 12px;
        padding: 20px 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 25px;
    }

    .instruction-card span {
        color: #ea580c;
        font-weight: 700;
    }

    [data-testid="stFileUploader"] label p {
        color: #0284c7 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 6px 16px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #1d4ed8 !important;
    }

    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ENCABEZADO Y TEXTOS PRINCIPALES
# ==========================================
st.markdown('<div class="biotech-title">BIOTECH</div>', unsafe_allow_html=True)
st.markdown('<div class="biotech-subtitle">🍊🍌 Detector de orden de consumo de plátanos y mandarinas</div>', unsafe_allow_html=True)

st.markdown('''
<div class="instruction-card">
    📌 <span>PASO 1:</span> Suba una imagen para analizar las frutas.<br>
    📌 <span>PASO 2:</span> Cada fruta se marcará con un <b>punto indicador individual</b> con el color de su prioridad.<br>
    📌 <span>PASO 3:</span> Consulte la leyenda en la esquina superior derecha de la imagen para guiarse.
</div>
''', unsafe_allow_html=True)

# Configuración en la barra lateral
st.sidebar.header("🔑 Conexión Roboflow")
roboflow_key = st.sidebar.text_input("Roboflow API Key", value="", type="password", help="Ingresa tu API Key de Roboflow Universe").strip()
version_modelo = st.sidebar.number_input("Versión del Proyecto", min_value=1, max_value=10, value=1)

# ==========================================
# PALETA DE COLORES (ORDEN DE CONSUMO)
# ==========================================
COLOR_PALETTE = [
    {"nombre": "Rojo", "bgr": (0, 0, 255), "desc": "Consumir 1ero (Urgente)"},
    {"nombre": "Morado", "bgr": (211, 0, 148), "desc": "Consumir 2do"},
    {"nombre": "Azul", "bgr": (255, 0, 0), "desc": "Consumir 3ero"},
    {"nombre": "Verde", "bgr": (100, 255, 100), "desc": "Consumir 4to"},
    {"nombre": "Negro", "bgr": (0, 0, 0), "desc": "Consumir 5to"},
    {"nombre": "Blanco", "bgr": (255, 255, 255), "desc": "Consumir 6to"}
]

# ==========================================
# ALGORITMO DE SEPARACIÓN INDIVIDUAL DE PLÁTANOS
# ==========================================
def separar_racimo_en_platanos_individuales(img_array, poligono_racimo):
    """
    Toma un racimo o masa detectada y lo divide geométricamente
    en plátanos individuales utilizando gradientes de iluminación y PCA.
    """
    x, y, w, h = cv2.boundingRect(poligono_racimo)
    area_bbox = w * h
    
    if area_bbox < 6000:
        return [poligono_racimo]
        
    h_img, w_img = img_array.shape[:2]
    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask_full, [poligono_racimo], 255)
    
    roi_img = img_array[y:y+h, x:x+w]
    roi_mask = mask_full[y:y+h, x:x+w]
    
    # 1. Separación por surcos de sombra en espacio LAB
    lab = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    _, grooves = cv2.threshold(magnitude, 25, 255, cv2.THRESH_BINARY)
    grooves = cv2.bitwise_and(grooves, roi_mask)
    
    kernel_groove = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grooves_dilated = cv2.dilate(grooves, kernel_groove, iterations=1)
    
    split_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(grooves_dilated))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_OPEN, kernel_open)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(split_mask)
    
    sub_poligonas = []
    area_minima = area_bbox * 0.04
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > area_minima:
            comp_mask = np.zeros_like(split_mask)
            comp_mask[labels == i] = 255
            comp_mask = cv2.dilate(comp_mask, kernel_groove, iterations=2)
            comp_mask = cv2.bitwise_and(comp_mask, roi_mask)
            
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_max) > area_minima:
                    sub_poligonas.append(c_max + np.array([x, y]))
                    
    # 2. Si la división devolvió un solo bloque, forzar segmentación longitudinal por PCA
    if len(sub_poligonas) <= 1:
        num_bananas = max(2, min(5, int(round(area_bbox / 15000.0))))
        pixel_coords = np.argwhere(roi_mask == 255)
        
        if len(pixel_coords) > 400:
            coords_pts = np.float32(pixel_coords[:, [1, 0]])
            mean, eigenvectors = cv2.PCACompute(coords_pts, mean=None)
            
            perp_vec = np.array([-eigenvectors[0, 1], eigenvectors[0, 0]])
            projections = np.dot(coords_pts - mean, perp_vec)
            
            min_p, max_p = np.min(projections), np.max(projections)
            step = (max_p - min_p) / float(num_bananas)
            
            sub_poligonas = []
            for b in range(num_bananas):
                p_low = min_p + b * step
                p_high = min_p + (b + 1) * step
                
                indices = np.where((projections >= p_low) & (projections < p_high))[0]
                if len(indices) > 40:
                    b_pts = coords_pts[indices].astype(np.int32)
                    b_mask = np.zeros_like(roi_mask)
                    for pt in b_pts:
                        b_mask[pt[1], pt[0]] = 255
                        
                    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel_smooth)
                    
                    cnts_b, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts_b:
                        c_b = max(cnts_b, key=cv2.contourArea)
                        if cv2.contourArea(c_b) > area_minima:
                            sub_poligonas.append(c_b + np.array([x, y]))
                            
    return sub_poligonas if len(sub_poligonas) > 1 else [poligono_racimo]

# ==========================================
# CARGA DE MODELOS Y CLASIFICADOR
# ==========================================
@st.cache_resource
def load_base_models():
    detector_orange = YOLO("yolov8m-seg.pt")
    detector_banana_fallback = YOLO("yolov8x-seg.pt")
    procesador = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    clasificador_estado = pipeline(
        "image-classification", 
        model="Dhahlan2000/ripeness_detection",
        image_processor=procesador
    )
    return detector_orange, detector_banana_fallback, clasificador_estado

detector_orange, detector_banana_fallback, clasificador_estado = load_base_models()

def inferir_roboflow_rest(image_path, api_key, model_ver=1):
    """Llamada REST con multipart/form-data robusta para ali-s-kantong/banana-8uvoq"""
    try:
        url = f"https://detect.roboflow.com/banana-8uvoq/{model_ver}?api_key={api_key}&confidence=15"
        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                files={"file": image_file},
                timeout=12
            )
        if response.status_code == 200:
            data = response.json()
            return data.get("predictions", []), None
        else:
            return [], f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return [], str(e)

# ==========================================
# PROCESAMIENTO DE IMAGEN
# ==========================================
uploaded_file = st.file_uploader("📂 Suba una imagen aquí (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_path = "temp_input.jpg"
    image = Image.open(uploaded_file).convert("RGB")
    image.save(temp_path)
    
    img_array = np.array(image)
    img_alto, img_ancho = img_array.shape[:2]
    
    frutas_detectadas = []

    with st.spinner("Procesando la imagen con ali-s-kantong/banana-8uvoq..."):
        
        # 1. DETECCIÓN DE PLÁTANOS CON ROBOFLOW
        platanos_brutos = []
        error_rf = None
        
        if roboflow_key:
            # Intento por REST API directa
            platanos_brutos, error_rf = inferir_roboflow_rest(temp_path, roboflow_key, version_modelo)
            
            # Si fallo REST, intento por SDK
            if not platanos_brutos and not error_rf:
                try:
                    rf = Roboflow(api_key=roboflow_key)
                    project = rf.workspace("ali-s-kantong").project("banana-8uvoq")
                    model_rf = project.version(version_modelo).model
                    res_json = model_rf.predict(temp_path, confidence=15, overlap=30).json()
                    platanos_brutos = res_json.get("predictions", [])
                except Exception as e:
                    error_rf = str(e)

        if platanos_brutos:
            st.sidebar.success("✅ Conectado a Roboflow (banana-8uvoq)")
            for pred in platanos_brutos:
                x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                x2, y2 = min(img_ancho, x + w // 2), min(img_alto, y + h // 2)
                
                points = pred.get("points")
                if points and len(points) > 2:
                    poly_base = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
                else:
                    poly_base = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

                # Separar racimo en plátanos individuales automáticamente
                sub_polys = separar_racimo_en_platanos_individuales(img_array, poly_base)
                for sp in sub_polys:
                    bx, by, bw, bh = cv2.boundingRect(sp)
                    frutas_detectadas.append({
                        'tipo': 'banana', 
                        'polygon': sp, 
                        'bbox': (bx, by, bx + bw, by + bh)
                    })
        else:
            if roboflow_key:
                if error_rf:
                    st.sidebar.error(f"❌ Error Roboflow: {error_rf}")
                else:
                    st.sidebar.warning("⚠️ Roboflow no devolvió detecciones. Usando respaldo con segmentación individual.")
            
            if os.path.exists("banana_8uvoq.pt"):
                model_local = YOLO("banana_8uvoq.pt")
                res_b = model_local(image, conf=0.08, iou=0.30, imgsz=1024)
            else:
                res_b = detector_banana_fallback(image, conf=0.08, iou=0.30, imgsz=1024)
                
            for r in res_b:
                if r.masks is not None:
                    for i, b in enumerate(r.boxes):
                        clase_id = int(b.cls[0].item())
                        if r.names[clase_id] == 'banana':
                            poly_base = r.masks[i].xy[0].astype(np.int32)
                            sub_polys = separar_racimo_en_platanos_individuales(img_array, poly_base)
                            for sp in sub_polys:
                                bx, by, bw, bh = cv2.boundingRect(sp)
                                frutas_detectadas.append({
                                    'tipo': 'banana', 
                                    'polygon': sp, 
                                    'bbox': (bx, by, bx + bw, by + bh)
                                })

        # 2. DETECCIÓN DE NARANJAS (YOLO)
        res_yolo_o = detector_orange(image, conf=0.15, iou=0.30, imgsz=1024)
        for result in res_yolo_o:
            if result.masks is not None:
                for i, box in enumerate(result.boxes):
                    clase_id = int(box.cls[0].item())
                    if result.names[clase_id] == 'orange':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        poly = result.masks[i].xy[0].astype(np.int32)
                        frutas_detectadas.append({'tipo': 'orange', 'polygon': poly, 'bbox': (x1, y1, x2, y2)})

        # 3. EVALUACIÓN DE MADUREZ CON Dhahlan2000/ripeness_detection
        frutas_evaluadas = []
        for item in frutas_detectadas:
            x1, y1, x2, y2 = item['bbox']
            poly = item['polygon']
            
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                try:
                    prediccion = clasificador_estado(crop)
                    etiqueta = prediccion[0]['label'].lower()
                    confianza = prediccion[0]['score']
                    
                    if 'rotten' in etiqueta or 'stale' in etiqueta:
                        indice_prioridad, estado_legible = 1, "Urgente (Pasada/Muy madura)"
                    else:
                        indice_prioridad, estado_legible = 3, "Buen estado"
                        
                    puntaje_final = indice_prioridad - (confianza * 0.5)
                    
                    # Centroide de la fruta para colocar el punto indicador
                    M = cv2.moments(poly)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    frutas_evaluadas.append({
                        'tipo': item['tipo'],
                        'polygon': poly, 
                        'centroid': (cx, cy), 
                        'puntaje': puntaje_final,
                        'estado_legible': estado_legible,
                        'confianza': confianza
                    })
                except Exception:
                    continue

    naranjas = [f for f in frutas_evaluadas if f['tipo'] == 'orange']
    platanos = [f for f in frutas_evaluadas if f['tipo'] == 'banana']

    def dibujar_leyenda_en_imagen(img):
        h, w = img.shape[:2]
        scale = max(0.55, min(w, h) / 750.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(scale * 1.8))
        
        margin = int(12 * scale)
        padding = int(12 * scale)
        box_width = int(285 * scale)
        row_height = int(30 * scale)
        box_height = padding * 2 + row_height * 7.2
        
        x2 = w - margin
        x1 = x2 - box_width
        y1 = margin
        y2 = int(y1 + box_height)
        
        overlay_box = img.copy()
        cv2.rectangle(overlay_box, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay_box, 0.70, img, 0.30, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, int(2 * scale)))
        
        y_cursor = y1 + padding + int(16 * scale)
        cv2.putText(img, "ORDEN DE CONSUMO", (x1 + padding, y_cursor), font, scale * 0.70, (255, 255, 255), thickness, cv2.LINE_AA)
        y_cursor += int(row_height * 1.05)
        
        for item in COLOR_PALETTE:
            color_bgr = item["bgr"]
            desc = item["desc"]
            
            sq_size = int(18 * scale)
            sq_x1 = x1 + padding
            sq_y1 = y_cursor - sq_size + int(2 * scale)
            sq_x2 = sq_x1 + sq_size
            sq_y2 = sq_y1 + sq_size
            
            cv2.rectangle(img, (sq_x1, sq_y1), (sq_x2, sq_y2), color_bgr, -1)
            cv2.rectangle(img, (sq_x1, sq_y1), (sq_x2, sq_y2), (255, 255, 255), 1)
            
            cv2.putText(img, desc, (sq_x2 + int(10 * scale), y_cursor), font, scale * 0.58, (240, 240, 240), max(1, thickness - 1), cv2.LINE_AA)
            y_cursor += row_height

    def dibujar_seccion_fruta(lista_frutas, nombre_tipo):
        if not lista_frutas:
            st.info(f"No se detectaron {nombre_tipo} en la imagen.")
            st.image(image, use_container_width=True, caption=f"Vista sin {nombre_tipo}")
            return
            
        lista_ordenada = sorted(lista_frutas, key=lambda x: x['puntaje'])
        # Mantiene la imagen original limpia sin pintar máscaras de colores
        img_final = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)
        
        datos_tabla = []
        radio_punto = max(10, int(min(img_alto, img_ancho) * 0.022))
        
        for rank, fruta in enumerate(lista_ordenada):
            cx, cy = fruta['centroid']
            
            color_index = min(rank, len(COLOR_PALETTE) - 1)
            color_info = COLOR_PALETTE[color_index]
            color_bgr = color_info["bgr"]

            labels = ["Consumir 1ero", "Consumir 2do", "Consumir 3ero", "Consumir 4to", "Consumir 5to", "Consumir 6to"]
            label_dibujo = labels[min(rank, len(labels)-1)] 
            
            # DIBUJO DE PUNTO INDICADOR INDIVIDUAL EN EL CENTROIDE
            cv2.circle(img_final, (cx, cy), radio_punto + 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img_final, (cx, cy), radio_punto + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img_final, (cx, cy), radio_punto, color_bgr, -1, cv2.LINE_AA)
            
            datos_tabla.append({
                "Orden": label_dibujo,
                "Color": color_info["nombre"],
                "Estado": fruta['estado_legible'],
                "Confianza": f"{fruta['confianza']*100:.1f}%"
            })
            
        dibujar_leyenda_en_imagen(img_final)
        
        st.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if datos_tabla:
            st.dataframe(pd.DataFrame(datos_tabla), hide_index=True, use_container_width=True)

    # ==========================================
    # VISUALIZACIÓN EN DOS COLUMNAS
    # ==========================================
    col_naranja, col_platano = st.columns(2)
    
    with col_naranja:
        st.subheader("🍊 Naranja / Mandarina")
        dibujar_seccion_fruta(naranjas, "Naranjas / Mandarinas")
        
    with col_platano:
        st.subheader("🍌 Plátano")
        dibujar_seccion_fruta(platanos, "Plátanos")