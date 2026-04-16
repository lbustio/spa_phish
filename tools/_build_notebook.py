"""
One-shot script: reads spaphish.ipynb, produces processing/spaphish.ipynb with:
  - All markdown cells translated to English
  - All Spanish inline comments / docstrings translated to English
  - Cells 16-20 removed (first-draft duplicate pipeline)
  - Paths updated: Path("data") -> Path("../data"), etc.
  - All cell outputs cleared
  - Cell IDs regenerated (uuid4)
"""

import json, re, uuid
from pathlib import Path

SRC  = Path(__file__).parent / "spaphish.ipynb"
DST  = Path(__file__).parent / "processing" / "spaphish.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))

# ── helpers ──────────────────────────────────────────────────────────────────

def new_id():
    return uuid.uuid4().hex[:8] + "-" + uuid.uuid4().hex[:4] + "-" + uuid.uuid4().hex[:4] + "-" + uuid.uuid4().hex[:4] + "-" + uuid.uuid4().hex[:12]

def translate_markdown(src: str) -> str:
    lines = src.splitlines(keepends=True)
    out   = []
    for ln in lines:
        # section titles / headings (match key phrases)
        ln = ln.replace("Para que el código sea autocontenido, deben instalarse las librerías que se van a usar.",
                        "To keep the code self-contained, install the required libraries.")
        ln = ln.replace("## Librerías de uso",          "## Libraries")
        ln = ln.replace("Se deben indicar las librerías que se van a importar para su uso en este notebook.",
                        "Import the libraries used throughout this notebook.")
        ln = ln.replace("## Configuraciones de trabajo.", "## Configuration")
        ln = ln.replace("Primero se deben definir, y crear si es que no existen, las carpetas de trabajo que se usarán en este notebook.",
                        "Define (and create if absent) the working directories used by this notebook.")
        ln = ln.replace("## Manejo de datos",            "## Data handling")
        ln = ln.replace(
            "En la carpeta de datos `RAW_DATA_DIR` están los mensajes de correo electrónico de phishing en español, y puede que hayan repetidos. Se deben obtener los mensajes únicos y eso se hace usando una hash (`SHA256`) para identificar los repetidos y separarlos a la carpeta correspondiente. Para este fin se define la función `file_hash`.",
            "The `RAW_DATA_DIR` folder contains the raw Spanish phishing email messages; some may be duplicates. "
            "We use a **SHA-256 hash** to identify duplicates and copy only unique messages to `PROCESSED_DATA_DIR`. "
            "The helper function `file_hash` computes that hash.")
        ln = ln.replace(
            "Se debe recorrer la carpeta `RAW_DATA_DIR` para obtener todos los mensajes con extensión `.eml` y eliminar los duplicados. Los datos listos para procesarse irán en la carpeta `PROCESSED_DATA_DIR`.",
            "Walk `RAW_DATA_DIR`, compute the SHA-256 hash of every `.eml` file, and copy only the unique ones to `PROCESSED_DATA_DIR`.")
        ln = ln.replace(
            "Una  vez realizado el filtrado de duplicados, se deben verificar los duplicados para ver que todo este ok.",
            "After deduplication, print the duplicate log to verify the results.")
        # pipeline explanation cell (cell 21)
        ln = ln.replace("## 📌 Explicación del pipeline de procesamiento de correos `.eml`",
                        "## Pipeline overview — `.eml` processing")
        ln = ln.replace(
            "Este notebook implementa un flujo completo para transformar mensajes de correo electrónico (`.eml`) en un **dataset estructurado** que luego puede usarse para análisis o problemas de clasificación.",
            "This notebook implements a complete pipeline that transforms raw `.eml` email files into a **structured dataset** suitable for analysis and classification.")
        ln = ln.replace("El procesamiento incluye varias etapas y genera distintos archivos de salida en la carpeta `reports/`.",
                        "The pipeline runs in several stages and writes its output files to the `../output/` directory.")
        ln = ln.replace("### 1. Preparación de carpetas y rutas",   "### 1. Directory setup")
        ln = ln.replace("- Se crean/verifican las carpetas:",        "- Creates/verifies the working directories:")
        ln = ln.replace("  - `data/raw/`: correos originales.",      "  - `../data/raw/`: raw email files.")
        ln = ln.replace("  - `data/processed/`: correos únicos (sin duplicados).", "  - `../data/processed/`: deduplicated email files.")
        ln = ln.replace("  - `models/`: espacio reservado para modelos ML.", "  - `../models/`: reserved for trained ML models.")
        ln = ln.replace("  - `reports/`: reportes, logs y dataset generado.", "  - `../output/`: generated dataset, reports, and logs.")
        ln = ln.replace("### 2. Manejo de duplicados",               "### 2. Duplicate handling")
        ln = ln.replace("- Se calculan **hashes SHA256** de cada `.eml` en `raw/`.",
                        "- Computes **SHA-256 hashes** for every `.eml` in `raw/`.")
        ln = ln.replace("- Solo los archivos únicos se copian a `processed/`.",
                        "- Only unique files are copied to `processed/`.")
        ln = ln.replace("- Archivos auxiliares generados:",          "- Auxiliary files produced:")
        ln = ln.replace("  - `reports/duplicates.txt`: lista de duplicados ignorados.",
                        "  - `../output/duplicates.txt`: list of ignored duplicates.")
        ln = ln.replace("  - `reports/hashes.txt`: listado de hashes de todos los mensajes.",
                        "  - `../output/hashes.txt`: SHA-256 hash of every message.")
        ln = ln.replace("### 3. Construcción del dataset (`eml_dataset.csv`)",
                        "### 3. Dataset construction (`eml_dataset.csv`)")
        ln = ln.replace("A partir de los correos en `data/processed/` se extrae:",
                        "Each email in `data/processed/` is parsed and the following fields are extracted:")
        ln = ln.replace("- **Identificación**:",                     "- **Identification**:")
        ln = ln.replace("  - `relative_path`: ruta relativa a `data/`.", "  - `relative_path`: path relative to `data/`.")
        ln = ln.replace("  - `hash`: identificador único.",           "  - `hash`: unique SHA-256 identifier.")
        ln = ln.replace("- **Encabezado**:",                          "- **Header**:")
        ln = ln.replace("  - `subject`: asunto del mensaje (vacío si falta).",
                        "  - `subject`: message subject (empty string if missing).")
        ln = ln.replace("  - `date`: fecha/hora en formato `YYYY-MM-DD HH:MM:SS`.",
                        "  - `date`: timestamp in `YYYY-MM-DD HH:MM:SS` format.")
        ln = ln.replace("- **Cuerpo del mensaje**:",                  "- **Message body**:")
        ln = ln.replace("  - `body_raw`: cuerpo sin procesar (texto plano o HTML).",
                        "  - `body_raw`: raw body (plain text or HTML).")
        ln = ln.replace("  - `body_text`: versión legible (texto plano o scrape de HTML).",
                        "  - `body_text`: human-readable body (plain text, or BeautifulSoup scrape of HTML).")
        ln = ln.replace("  - `body_type`: `text/plain`, `text/html`, `html+image` o `unknown`.",
                        "  - `body_type`: one of `text/plain`, `text/html`, `html+image`, or `unknown`.")
        ln = ln.replace("  - `media_flag`: `\"IMAGEN\"` si el cuerpo contiene imágenes.",
                        "  - `media_flag`: `\"IMAGE\"` if the body contains any image.")
        ln = ln.replace("  - `body_media_flags`: detalle del tipo de imágenes detectadas:",
                        "  - `body_media_flags`: detail of the image types found:")
        ln = ln.replace("    - `INLINE_IMG_TAG`: HTML con `<img>`.",  "    - `INLINE_IMG_TAG`: HTML contains an `<img>` tag.")
        ln = ln.replace("    - `DATA_URI_IMG`: imagen en base64 embebida.", "    - `DATA_URI_IMG`: base64-encoded data URI image.")
        ln = ln.replace("    - `CID_IMG`: imagen referenciada con `cid:`.", "    - `CID_IMG`: image referenced via `cid:`.")
        ln = ln.replace("    - `IMG_ATTACHMENT_INLINE`: imágenes inline sin filename.",
                        "    - `IMG_ATTACHMENT_INLINE`: inline `image/*` part with no filename.")
        ln = ln.replace("- **URLs**:",                                "- **URLs**:")
        ln = ln.replace("  - `url_count`: número de URLs en el cuerpo.",
                        "  - `url_count`: number of URLs found in the body.")
        ln = ln.replace("  - `urls`: listado de URLs codificado como `[URL1],[URL2],...`.",
                        "  - `urls`: URL list encoded as `[URL1],[URL2],...`.")
        ln = ln.replace("- **Adjuntos**:",                            "- **Attachments**:")
        ln = ln.replace("  - `attachments_count`: cantidad de adjuntos.",
                        "  - `attachments_count`: number of attachments.")
        ln = ln.replace("  - `attachments_types`: extensiones (`pdf`, `zip`, etc.).",
                        "  - `attachments_types`: file extensions (`pdf`, `zip`, etc.).")
        ln = ln.replace("  - `attachments_total_size`: tamaño total (bytes).",
                        "  - `attachments_total_size`: total attachment size in bytes.")
        ln = ln.replace("  - `attachments_sizes`: lista de tamaños individuales (`[1024],[2048],...`).",
                        "  - `attachments_sizes`: per-attachment sizes (`[1024],[2048],...`).")
        ln = ln.replace("- **Ruteo**:",                               "- **Routing**:")
        ln = ln.replace("  - `hops_count`: número de saltos (cabeceras `Received`).",
                        "  - `hops_count`: hop count derived from `Received` headers.")
        ln = ln.replace("### 4. Manejo de errores",                   "### 4. Error handling")
        ln = ln.replace("- Si un mensaje no se puede procesar, se registra en:",
                        "- If a message cannot be parsed, the error is logged to:")
        ln = ln.replace("  - `reports/processing_errors.log`",        "  - `../output/processing_errors.log`")
        ln = ln.replace("- Cada línea incluye:",                      "- Each log entry includes:")
        ln = ln.replace("  - Timestamp de procesamiento.",            "  - Processing timestamp.")
        ln = ln.replace("  - Ruta relativa del archivo.",             "  - Relative file path.")
        ln = ln.replace("  - Hash (si pudo calcularse).",             "  - SHA-256 hash (when available).")
        ln = ln.replace("  - Tipo de error y mensaje de detalle.",    "  - Error type and detail message.")
        out.append(ln)
    return "".join(out)


def translate_code(src: str) -> str:
    """Translate Spanish comments and docstrings in code cells."""
    # ── inline comments ──────────────────────────────────────────────────────
    replacements = [
        # config cell
        ("# Definir las carpetas como objetos Path directamente",
         "# Define working directories as Path objects"),
        ("DATA_DIR = Path(\"../data\") # Carpeta principal de datos",
         "DATA_DIR = Path(\"../data\")  # Main data directory"),
        ("DATA_DIR = Path(\"data\") # Carpeta principal de datos",
         "DATA_DIR = Path(\"../data\")  # Main data directory"),
        ("RAW_DATA_DIR = DATA_DIR / \"raw\" # Carpeta de datos sin procesar",
         "RAW_DATA_DIR = DATA_DIR / \"raw\"           # Raw (unprocessed) emails"),
        ("PROCESSED_DATA_DIR = DATA_DIR / \"processed\" # Carpeta de datos procesados",
         "PROCESSED_DATA_DIR = DATA_DIR / \"processed\" # Deduplicated emails"),
        ("MODELS_DIR = Path(\"models\") # Carpeta de modelos",
         "MODELS_DIR = Path(\"../models\")  # ML models"),
        ("REPORTS_DIR = Path(\"reports\") # Carpeta de informes y reportes",
         "REPORTS_DIR = Path(\"../output\")  # Reports and output files"),
        ("MODELS_DIR = Path(\"../models\") # Carpeta de modelos",
         "MODELS_DIR = Path(\"../models\")  # ML models"),
        ("REPORTS_DIR = Path(\"../output\") # Carpeta de informes y reportes",
         "REPORTS_DIR = Path(\"../output\")  # Reports and output files"),
        ("LOG_FILE = REPORTS_DIR / \"duplicates.txt\"   # Uso de operador \"/\" en Path",
         "LOG_FILE = REPORTS_DIR / \"duplicates.txt\"   # Duplicate log (uses Path / operator)"),
        ("DATASET_CSV = REPORTS_DIR / \"eml_dataset.csv\" # Archivo CSV del dataset final",
         "DATASET_CSV = REPORTS_DIR / \"eml_dataset.csv\" # Final dataset CSV"),
        ("ERRORS_LOG  = REPORTS_DIR / \"processing_errors.log\" # Archivo de log de errores",
         "ERRORS_LOG  = REPORTS_DIR / \"processing_errors.log\" # Parsing error log"),
        ("ENABLE_OCR = True                 # bandera global para activar/desactivar OCR",
         "ENABLE_OCR = True                 # Global flag to enable/disable OCR"),
        ("OCR_LANG = \"eng+spa\"              # idiomas (Tesseract): inglés + español",
         "OCR_LANG = \"eng+spa\"              # Tesseract languages: English + Spanish"),
        ("MAX_IMAGES_PER_EMAIL = 20         # límite superior para evitar tiempos excesivos",
         "MAX_IMAGES_PER_EMAIL = 20         # Cap to avoid excessive processing time"),
        ("MAX_OCR_TEXT_CHARS = 20000        # truncar ocr_text concatenado si excede",
         "MAX_OCR_TEXT_CHARS = 20000        # Truncate concatenated OCR text beyond this limit"),
        ("ALLOW_REMOTE_IMAGES = False       # NO descargar imágenes http/https por defecto",
         "ALLOW_REMOTE_IMAGES = False       # Do NOT download remote http/https images by default"),
        ("# Log de errores OCR (texto plano), consistente con tus otros logs",
         "# OCR error log (plain text), consistent with other logs"),
        ("OCR_ERRORS_LOG = REPORTS_DIR / \"ocr_errors.log\"",
         "OCR_ERRORS_LOG = REPORTS_DIR / \"ocr_errors.log\""),
        ("# Lista de carpetas de trabajo",
         "# List of working directories"),
        ("working_folders = [RAW_DATA_DIR,    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR] # Lista de carpetas de trabajo",
         "working_folders = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]"),
        ("# Crear todas las carpetas si no existen y mostrar resultados",
         "# Create directories that do not exist yet and report results"),
        ("    if not path.exists(): # Verificar si la carpeta no existe",
         "    if not path.exists():"),
        ("        path.mkdir(parents=True, exist_ok=True) # Crear la carpeta (y padres si es necesario)",
         "        path.mkdir(parents=True, exist_ok=True)"),
        ("        print(f\"📂 Creada carpeta: {path}\")",
         "        print(f\"Created: {path}\")"),
        ("        print(f\"✅ Ya existe: {path}\")",
         "        print(f\"Already exists: {path}\")"),
        ("print(\"\\n✔️ Todas las carpetas ya se encuentran listas para el trabajo.\")",
         "print(\"\\nAll working directories are ready.\")"),
        # dedup loop comments
        ("# Archivo de salida donde se guardarán los hashes calculados de cada mensaje",
         "# Output file that records each message hash"),
        ("# Diccionario para registrar qué hashes ya se han encontrado",
         "# Dictionary tracking already-seen hashes"),
        ("# (clave = hash calculado, valor = nombre de archivo original)",
         "# (key = hash, value = original filename)"),
        ("# Contadores para las estadísticas finales",
         "# Counters for final statistics"),
        ("copiados = 0     # Archivos únicos copiados a PROCESSED_DATA_DIR",
         "copiados = 0     # Unique files copied to PROCESSED_DATA_DIR"),
        ("duplicados = 0   # Archivos ignorados por ser duplicados",
         "duplicados = 0   # Duplicate files ignored"),
        ("total = 0        # Total de archivos procesados",
         "total = 0        # Total files encountered"),
        ("# Abrimos dos archivos de log al mismo tiempo:",
         "# Open both log files simultaneously:"),
        ("# - LOG_FILE   → para registrar duplicados encontrados",
         "# - LOG_FILE     → record duplicates found"),
        ("# - HASHES_FILE → para listar el hash de cada archivo .eml",
         "# - HASHES_FILE  → list the hash of every .eml"),
        ("    # Encabezados en los archivos de salida",
         "    # Write headers to both output files"),
        ("    log.write(\"Listado de duplicados encontrados:\\n\\n\")",
         "    log.write(\"List of duplicates found:\\n\\n\")"),
        ("    hashes_file.write(\"Listado de hashes de cada archivo:\\n\\n\")",
         "    hashes_file.write(\"Hash listing for every file:\\n\\n\")"),
        ("    # Recorremos todos los archivos con extensión .eml en la carpeta RAW_DATA_DIR",
         "    # Walk all .eml files in RAW_DATA_DIR"),
        ("        h = file_hash(eml_file)  # Calculamos el hash del archivo",
         "        h = file_hash(eml_file)  # Compute the file hash"),
        ("        # Mostrar en pantalla el archivo y su hash",
         "        # Print file path and hash"),
        ("        # Guardar en el archivo de hashes",
         "        # Write to the hashes file"),
        ("        # Si el hash aún no fue registrado, es un archivo único",
         "        # If this hash has not been seen before, it is a unique file"),
        ("            shutil.copy2(eml_file, PROCESSED_DATA_DIR / eml_file.name)  # Copiar a carpeta de procesados",
         "            shutil.copy2(eml_file, PROCESSED_DATA_DIR / eml_file.name)  # Copy to processed folder"),
        ("            print(f\"✅ Copiado: {eml_file.name}\")",
         "            print(f\"Copied: {eml_file.name}\")"),
        ("        else:",
         "        else:"),
        ("            # Si el hash ya existe, se considera duplicado y no se copia",
         "            # Hash already seen — this is a duplicate; skip it"),
        ("            msg = f\"Duplicado ignorado: {eml_file} (igual a {seen_hashes[h]})\"",
         "            msg = f\"Duplicate ignored: {eml_file} (same as {seen_hashes[h]})\""),
        ("            log.write(msg + \"\\n\")   # Guardar en log de duplicados",
         "            log.write(msg + \"\\n\")"),
        ("            print(f\"⚠️ {msg}\")",
         "            print(f\"Duplicate: {msg}\")"),
        ("# --- Mostrar estadísticas finales ---",
         "# --- Print final statistics ---"),
        ("print(\"\\n📊 Estadísticas del procesamiento\")",
         "print(\"\\nDeduplication statistics\")"),
        ("print(f\"   Total de archivos encontrados: {total}\")",
         "print(f\"  Total .eml files found:    {total}\")"),
        ("print(f\"   Archivos copiados (únicos):    {copiados}\")",
         "print(f\"  Unique files copied:        {copiados}\")"),
        ("print(f\"   Archivos duplicados ignorados: {duplicados}\")",
         "print(f\"  Duplicate files ignored:    {duplicados}\")"),
        ("print(\"\\n✔️ Proceso completado. Archivos únicos en:\", PROCESSED_DATA_DIR)",
         "print(\"\\nDone. Unique files are in:\", PROCESSED_DATA_DIR)"),
        ("print(\"📄 Log de duplicados guardado en:\", LOG_FILE)",
         "print(\"Duplicate log saved to:\", LOG_FILE)"),
        ("print(\"📄 Listado de hashes guardado en:\", HASHES_FILE)",
         "print(\"Hash listing saved to:\", HASHES_FILE)"),
        # main processing loop (cell 27 / 20)
        ("# Métricas de resumen",
         "# Summary metrics"),
        ("# Inicializar/limpiar logs de errores",
         "# Initialise (or clear) error logs"),
        ("    fh.write(\"Listado de errores de procesamiento:\\n\\n\")",
         "    fh.write(\"Processing error log:\\n\\n\")"),
        ("    fh.write(\"Listado de errores de OCR:\\n\\n\")",
         "    fh.write(\"OCR error log:\\n\\n\")"),
        ("    # Ruta relativa a data/",
         "    # Path relative to data/"),
        ("    # Hash (por si falla el parsing poder rastrear)",
         "    # Compute hash for traceability even if parsing fails"),
        ("    try:",
         "    try:"),
        ("        # Parseo del mensaje",
         "        # Parse the email message"),
        ("        # body_raw / body_text",
         "        # Derive body_raw and body_text"),
        ("        # URLs",
         "        # URLs"),
        ("        # Hops",
         "        # Hop count"),
        ("        # Adjuntos (acumulado global)",
         "        # Attachments (global accumulator)"),
        ("        # === OCR ===",
         "        # === OCR ==="),
        ("# DataFrame y CSV",
         "# Build DataFrame and save to CSV"),
        ("print(f\"✅ Dataset generado con {len(df)} filas en: {DATASET_CSV}\")",
         "print(f\"Dataset saved: {len(df)} rows -> {DATASET_CSV}\")"),
        ("print(\"📄 Logs:\", ERRORS_LOG, \"|\", OCR_ERRORS_LOG)",
         "print(\"Logs:\", ERRORS_LOG, \"|\", OCR_ERRORS_LOG)"),
        # summary print cell
        ("print(\"\\n📊 Resumen del procesamiento\")",
         "print(\"\\nProcessing summary\")"),
        ("print(f\"   Total de mensajes encontrados:      {total_msgs}\")",
         "print(f\"  Total messages found:          {total_msgs}\")"),
        ("print(f\"   Mensajes procesados OK:             {processed_ok}\")",
         "print(f\"  Messages processed OK:         {processed_ok}\")"),
        ("print(f\"   Mensajes con error de parsing:      {errors_count}\")",
         "print(f\"  Messages with parse errors:    {errors_count}\")"),
        ("print(f\"   Total adjuntos detectados:          {total_attachments_detected}\")",
         "print(f\"  Total attachments detected:    {total_attachments_detected}\")"),
        ("print(f\"   Total URLs detectadas:              {total_urls_detected}\")",
         "print(f\"  Total URLs detected:           {total_urls_detected}\")"),
        ("print(f\"   Total imágenes con texto (OCR):     {total_ocr_images}\")",
         "print(f\"  Total images with OCR text:    {total_ocr_images}\")"),
        ("print(\"\\n📄 Archivos de salida\")",
         "print(\"\\nOutput files\")"),
        ("print(f\"   Dataset (CSV):          {DATASET_CSV}\")",
         "print(f\"  Dataset (CSV):      {DATASET_CSV}\")"),
        ("print(f\"   Errores (txt):          {ERRORS_LOG}\")",
         "print(f\"  Error log:          {ERRORS_LOG}\")"),
        ("print(f\"   Errores OCR (txt):      {OCR_ERRORS_LOG}\")",
         "print(f\"  OCR error log:      {OCR_ERRORS_LOG}\")"),
        ("    print(f\"   Duplicados (txt):       {REPORTS_DIR / 'duplicates.txt'}\")",
         "    print(f\"  Duplicates log:     {REPORTS_DIR / 'duplicates.txt'}\")"),
        ("    print(f\"   Hashes (txt):           {REPORTS_DIR / 'hashes.txt'}\")",
         "    print(f\"  Hash listing:       {REPORTS_DIR / 'hashes.txt'}\")"),
        # generic remaining Spanish
        ("# bandera global para activar/desactivar OCR",
         "# Global flag to enable/disable OCR"),
        ("# idiomas (Tesseract): inglés + español",
         "# Tesseract languages: English + Spanish"),
        ("# límite superior para evitar tiempos excesivos",
         "# Cap to avoid excessive processing time"),
        ("# truncar ocr_text concatenado si excede",
         "# Truncate concatenated OCR text beyond this limit"),
        ("# NO descargar imágenes http/https por defecto",
         "# Do NOT download remote http/https images by default"),
        # extract_bodies_and_media docstring + comments
        ("    has_inline_img_tag = False      # HTML contiene <img>",
         "    has_inline_img_tag = False      # HTML has an <img> tag"),
        ("    has_data_uri_img   = False      # <img src=\"data:image/...;base64,...\">",
         "    has_data_uri_img   = False      # <img src=\"data:image/...;base64,...\">"),
        ("    has_cid_img        = False      # <img src=\"cid:...\">",
         "    has_cid_img        = False      # <img src=\"cid:...\">"),
        ("    has_img_attach_inl = False      # partes image/* inline (o sin filename), tratadas como cuerpo",
         "    has_img_attach_inl = False      # inline image/* part (no filename) — treated as body"),
        ("        # cuerpos",
         "        # Body parts"),
        ("        # imágenes inline (aunque no haya <img> en HTML)",
         "        # Inline images (even if there is no <img> tag in HTML)"),
        ("        # imágenes inline (aunque no haya <img>)",
         "        # Inline images (even without an <img> tag)"),
        ("        # adjuntos (si tiene filename lo contamos)",
         "        # Attachments (anything with a filename)"),
        ("        # adjuntos si hay filename",
         "        # Attachments — count parts that have a filename"),
        ("    # Analizamos HTML para <img> / data-uri / cid",
         "    # Inspect HTML for <img> tags, data URIs, and cid references"),
        ("    # análisis de HTML para <img>, data-uri, cid",
         "    # Inspect HTML for <img> tags, data URIs, and cid references"),
        ("    # tipo de body",
         "    # Determine body type"),
        # build_cid_map docstring
        ("    Construye un mapeo Content-ID -> bytes de la parte correspondiente,\n    para resolver <img src=\"cid:...\"> en HTML.",
         "    Build a Content-ID -> bytes mapping so that <img src=\"cid:\"> references\n    in HTML can be resolved to their binary payloads."),
        # extract_images_from_html
        ("        # data URI",
         "        # data URI image"),
        ("            # formato: data:image/png;base64,XXXXX",
         "            # Format: data:image/png;base64,XXXXX"),
        ("        # cid",
         "        # cid reference"),
        ("        # http/https (por defecto NO descargamos)",
         "        # http/https — not downloaded by default"),
        ("                    # Aquí podrías descargar la imagen, si así lo decides.",
         "                    # Remote download could be added here."),
        ("                    # Aquí podrías descargar la imagen si decides habilitarlo.\n                    # Por ahora, lo omitimos por privacidad/reproducibilidad.",
         "                    # Remote download could be added here.\n                    # Skipped for privacy and reproducibility."),
        # extract_images_from_parts
        ("                # si falla, lo ignoramos y lo reportaremos en OCR si corresponde",
         "                # Failure is silently ignored; OCR will log it if relevant"),
        # preprocess_image_for_ocr
        ("        # Si es animado (GIF), tomar el primer frame",
         "        # For animated images (e.g. GIF), use the first frame"),
        ("        # Convertir a escala de grises",
         "        # Convert to grayscale"),
        ("        # Umbral simple (opcional, descomentar si te ayuda):",
         "        # Optional simple threshold (uncomment if it helps):"),
        # run_ocr_on_images
        ("        # el log se realiza en el sitio que llama (para tener relative_path/hash/img_idx)",
         "        # Error logging is done by the caller (to have relative_path/hash/img_idx)"),
        ("    # concatenar y truncar si excede",
         "    # Concatenate and truncate if over the character limit"),
        # URL helpers
        ("    # Dedup preservando orden",
         "    # Deduplicate while preserving insertion order"),
        ("    # <a href=...>",
         "    # Anchor tags"),
        ("    # recursos comunes",
         "    # Common resource tags"),
        # collect_urls / count_hops docstrings
        ("\"\"\"Retorna (url_count, urls_codificadas) desde texto y HTML, dedup preservando orden.\"\"\"",
         "\"\"\"Return (url_count, encoded_urls) extracted from plain text and HTML, deduplicated.\"\"\""),
        ("\"\"\"Cantidad de cabeceras 'Received' (aprox. número de saltos).\"\"\"",
         "\"\"\"Return the number of 'Received' headers (approximate hop count).\"\"\""),
        # encode_list / encode_sizes docstrings (one-liners)
        ("\"\"\"Codifica una lista como: [item1],[item2],... (sin espacios).\"\"\"",
         "\"\"\"Encode a list as [item1],[item2],... (no spaces).\"\"\""),
        ("\"\"\"Codifica tamaños en bytes como: [1024],[2048],...\"\"\"",
         "\"\"\"Encode a list of byte-sizes as [1024],[2048],...\"\"\""),
        # parse helpers
        ("\"\"\"Parsea un .eml y retorna EmailMessage (lanza excepción si falla).\"\"\"",
         "\"\"\"Parse a .eml file and return an EmailMessage (raises on failure).\"\"\""),
        ("\"\"\"Convierte Date a 'YYYY-MM-DD HH:MM:SS' o '' si no es posible.\"\"\"",
         "\"\"\"Parse the Date header into 'YYYY-MM-DD HH:MM:SS', or '' on failure.\"\"\""),
        # remaining log messages in the main loop
        ("    fh_errors.write(\"Listado de errores de procesamiento:\\n\\n\")",
         "    fh_errors.write(\"Processing error log:\\n\\n\")"),
        ("    f.write(\"Listado de errores de OCR:\\n\\n\")",
         "    f.write(\"OCR error log:\\n\\n\")"),
        # cell 27 extra comments
        ("        # Parseo de mensaje",
         "        # Parse the email message"),
        ("        # Extraer cuerpos + medios/adjuntos (ya lo tenías en Celda C)",
         "        # Extract bodies + media/attachments"),
        ("        # body_raw y body_text (prioridad acordada)",
         "        # Derive body_raw and body_text"),
        ("        # URLs (desde texto plano y HTML)",
         "        # URLs (from plain text and HTML)"),
        ("        # Hops (Received)",
         "        # Hop count (Received headers)"),
        ("        # Adjuntos (acumulador global)",
         "        # Attachment count (global accumulator)"),
        ("        # === OCR (nuevo bloque) ===",
         "        # === OCR ==="),
        ("            # Reunir imágenes desde HTML y desde partes image/*",
         "            # Collect images from HTML and from image/* parts"),
        ("            # Merge y recorte a máximo permitido",
         "            # Merge and cap at MAX_IMAGES_PER_EMAIL"),
        ("            # Ejecutar OCR por imagen con logging de errores por imagen",
         "            # Run OCR per image, logging individual failures"),
        ("            idx = 0",
         "            idx = 0"),
        ("                except Exception as e:",
         "                except Exception as e:"),
        ("                    texts.append(\"\")  # placeholder",
         "                    texts.append(\"\")  # keep position"),
        ("            # Concatenar y contabilizar",
         "            # Concatenate and count images that produced text"),
        ("            ocr_image_count = sum(1 for t in texts if t)  # solo las que dieron texto",
         "            ocr_image_count = sum(1 for t in texts if t)  # only images that yielded text"),
        ("        # Agregar registro a la tabla final",
         "        # Append record to the final table"),
        ("            # --- nuevas columnas OCR ---",
         "            # --- OCR columns ---"),
        ("# Construir DataFrame y guardar CSV",
         "# Build DataFrame and save to CSV"),
        # Path updates (catch any remaining ones not already replaced)
        ("Path(\"data\")", "Path(\"../data\")"),
        ("Path(\"models\")", "Path(\"../models\")"),
        ("Path(\"reports\")", "Path(\"../output\")"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    # translate the file_hash multi-line docstring
    src = src.replace(
        '    """\n    Calcula el hash SHA256 de un archivo.\n    \n    Este hash se utiliza para identificar de forma única el contenido del archivo, \n    lo cual resulta útil para detectar duplicados incluso si tienen nombres distintos.\n    \n    Parámetros\n    ----------\n    filepath : str o Path\n        Ruta del archivo que se desea procesar.\n    block_size : int, opcional (por defecto 65536)\n        Tamaño de bloque en bytes para leer el archivo por partes. \n        Se usa para no cargar archivos grandes en memoria de una sola vez.\n    \n    Retorna\n    -------\n    str\n        Cadena hexadecimal que representa el hash SHA256 del archivo.\n    """',
        '    """\n    Compute the SHA-256 hash of a file.\n\n    The hash uniquely identifies file content, making it possible to detect\n    duplicates regardless of filename.\n\n    Parameters\n    ----------\n    filepath : str or Path\n        Path to the file to hash.\n    block_size : int, optional (default 65536)\n        Read chunk size in bytes. Avoids loading large files entirely into memory.\n\n    Returns\n    -------\n    str\n        Hexadecimal SHA-256 digest of the file.\n    """'
    )

    # translate file_hash inline comments
    src = src.replace(
        "    # Crear un objeto de tipo hash SHA256\n    hasher = hashlib.sha256()\n    \n    # Abrir el archivo en modo binario (\"rb\" = read binary)\n    with open(filepath, \"rb\") as f:\n        # Leer el archivo en bloques de tamaño 'block_size'\n        # Se usa 'iter' con un lambda para ir leyendo hasta que no haya más datos (\"\" vacío)\n        for block in iter(lambda: f.read(block_size), b\"\"):\n            # Actualizar el hash con el bloque leído\n            hasher.update(block)\n    \n    # Retornar el hash calculado en formato hexadecimal\n    return hasher.hexdigest()",
        "    # Create a SHA-256 hash object\n    hasher = hashlib.sha256()\n\n    # Open the file in binary mode\n    with open(filepath, \"rb\") as f:\n        # Read the file in chunks until EOF (empty bytes sentinel)\n        for block in iter(lambda: f.read(block_size), b\"\"):\n            hasher.update(block)\n\n    # Return the hex digest\n    return hasher.hexdigest()"
    )

    # translate extract_bodies_and_media docstring
    src = src.replace(
        '    """\n    Retorna:\n      - text_plain, text_html\n      - body_type: text/plain | text/html | html+image | unknown\n      - media_flag: "IMAGEN" si hay cualquier imagen (HTML <img>, data URI, cid, inline image/*)\n      - body_media_flags: [INLINE_IMG_TAG],[DATA_URI_IMG],[CID_IMG],[IMG_ATTACHMENT_INLINE]\n      - attachments_count, attachments_types ([ext]), attachments_sizes ([bytes]), attachments_total_size (int)\n    """',
        '    """\n    Extract body text, media flags, and attachment metadata from an email message.\n\n    Returns a dict with:\n      - text_plain, text_html\n      - body_type : "text/plain" | "text/html" | "html+image" | "unknown"\n      - media_flag: "IMAGE" if any image is present (inline <img>, data URI, cid, or image/* part)\n      - body_media_flags: encoded flags [INLINE_IMG_TAG],[DATA_URI_IMG],[CID_IMG],[IMG_ATTACHMENT_INLINE]\n      - attachments_count, attachments_types ([ext]), attachments_sizes ([bytes]),\n        attachments_total_size (int)\n    """'
    )
    src = src.replace(
        '    """\n    Retorna:\n      - text_plain, text_html\n      - body_type: text/plain | text/html | html+image | unknown\n      - media_flag: "IMAGEN" si hay cualquier imagen en el cuerpo (HTML <img>, data URI, cid, inline)\n      - body_media_flags (codificado): [INLINE_IMG_TAG],[DATA_URI_IMG],[CID_IMG],[IMG_ATTACHMENT_INLINE]\n      - attachments_count, attachments_types ([ext]), attachments_sizes ([bytes]), attachments_total_size (int)\n    """',
        '    """\n    Extract body text, media flags, and attachment metadata from an email message.\n\n    Returns a dict with:\n      - text_plain, text_html\n      - body_type : "text/plain" | "text/html" | "html+image" | "unknown"\n      - media_flag: "IMAGE" if any image is present (inline <img>, data URI, cid, or image/* part)\n      - body_media_flags: encoded flags [INLINE_IMG_TAG],[DATA_URI_IMG],[CID_IMG],[IMG_ATTACHMENT_INLINE]\n      - attachments_count, attachments_types ([ext]), attachments_sizes ([bytes]),\n        attachments_total_size (int)\n    """'
    )

    # translate extract_images_from_html docstring (cell 19 version)
    src = src.replace(
        '    """\n    Devuelve lista de bytes de imágenes del HTML:\n      - data URI base64\n      - cid:... resuelto\n      - http/https: ignorado salvo ALLOW_REMOTE_IMAGES=True (no recomendado)\n    """',
        '    """\n    Return a list of image bytes extracted from HTML:\n      - base64 data URIs\n      - cid:... references resolved via the message CID map\n      - http/https: skipped unless ALLOW_REMOTE_IMAGES=True (not recommended)\n    """'
    )
    # cell 24 version
    src = src.replace(
        '    """\n    Devuelve lista de bytes de imágenes encontradas en el HTML:\n      - data URI base64\n      - cid:... resuelto usando el cid_map\n      - http/https: SOLO si ALLOW_REMOTE_IMAGES=True (por defecto False)\n    """',
        '    """\n    Return a list of image bytes extracted from HTML:\n      - base64 data URIs\n      - cid:... references resolved via the message CID map\n      - http/https: only if ALLOW_REMOTE_IMAGES=True (default is False)\n    """'
    )

    # translate extract_images_from_parts docstring
    src = src.replace(
        '    """Devuelve bytes de todas las partes image/* (inline o attachment)."""',
        '    """Return bytes for every image/* part (inline or attachment)."""'
    )
    src = src.replace(
        '    """\n    Devuelve lista de bytes de todas las partes image/* (inline o attachment).\n    Útil para firmas/inline sin <img> o adjuntos de imagen.\n    """',
        '    """\n    Return bytes for every image/* MIME part (inline or attachment).\n    Useful for signatures or inline images that have no corresponding <img> tag.\n    """'
    )

    # preprocess_image_for_ocr docstring
    src = src.replace(
        '    """\n    Carga bytes -> PIL.Image y preprocesa:\n    - primer frame si animado\n    - escala de grises\n    - (opcional) umbral\n    """',
        '    """\n    Load image bytes into a PIL.Image and apply pre-processing:\n    - Seek to first frame if the image is animated.\n    - Convert to grayscale.\n    - Optional simple threshold (commented out).\n    """'
    )
    src = src.replace(
        '    """\n    Carga bytes -> PIL.Image y realiza preprocesado simple:\n    - convertir a L (escala de grises)\n    - binarización básica opcional\n    - retornar PIL.Image listo para OCR\n    """',
        '    """\n    Load image bytes into a PIL.Image and apply simple pre-processing:\n    - Convert to grayscale (mode L).\n    - Optional basic binarisation (commented out).\n    Returns a PIL.Image ready for Tesseract.\n    """'
    )

    # run_ocr_on_images docstring
    src = src.replace(
        '    """\n    Ejecuta OCR sobre una lista de imágenes (bytes).\n    Retorna (ocr_text_concatenado, ocr_image_count)\n    """',
        '    """\n    Run OCR over a list of images (each given as bytes).\n    Returns (concatenated_ocr_text, image_count_with_text).\n    """'
    )

    # log_ocr_error — no docstring originally, leave as-is but translate the write line
    # (already handled above via f.write replacement)

    return src


# ── cell-ID → index map ──────────────────────────────────────────────────────
# First-draft cell IDs (cells 16-20 from the original numbering)
FIRST_DRAFT_IDS = {"557598ee", "114ea899", "5d28abe8", "ae57f157", "60da1ba5"}

# ── process cells ────────────────────────────────────────────────────────────
new_cells = []
for cell in nb["cells"]:
    cell_id = cell.get("id", "")
    # Strip the 8-char prefix from the full UUID if present
    short_id = cell_id.split("-")[0] if "-" in cell_id else cell_id[:8]

    if short_id in FIRST_DRAFT_IDS:
        continue  # remove first-draft cells

    # Clear outputs
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        # Translate code source
        src = "".join(cell["source"])
        src = translate_code(src)
        cell["source"] = src.splitlines(keepends=True)

    elif cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        src = translate_markdown(src)
        cell["source"] = src.splitlines(keepends=True)

    # Assign fresh cell ID
    cell["id"] = new_id()
    new_cells.append(cell)

nb["cells"] = new_cells

# ── write output ─────────────────────────────────────────────────────────────
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written {len(new_cells)} cells to {DST}")
