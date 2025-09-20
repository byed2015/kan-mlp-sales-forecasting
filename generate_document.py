"""
Script alternativo para generar el documento en múltiples formatos
Incluye conversión a HTML cuando LaTeX no está disponible

Características:
- Conversión Markdown → HTML con formato académico
- CSS styling profesional 
- Renderizado de ecuaciones matemáticas con MathJax
- Manejo de imágenes y tablas
- Exportación a PDF usando navegador (si disponible)
"""

import os
import subprocess
import sys
import shutil
import json
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile
import re

# Configuración
MARKDOWN_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.md"
HTML_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.html"
PDF_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.pdf"
CSS_FILE = "academic_style.css"

def create_academic_css():
    """Crea un archivo CSS con estilo académico profesional"""
    
    css_content = """
/* Estilo Académico Profesional para Proyecto KAN vs MLP */

/* Reset y configuración base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.6;
    color: #333;
    background-color: #fff;
    max-width: 8.5in;
    margin: 0 auto;
    padding: 1in;
    text-align: justify;
}

/* Título principal */
h1 {
    color: #1a472a;
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    margin: 0 0 0.5in 0;
    page-break-after: avoid;
}

/* Información del autor */
.author-info {
    text-align: center;
    margin-bottom: 0.5in;
    font-size: 11pt;
}

/* Abstract */
.abstract {
    background-color: #f8f9fa;
    padding: 0.3in;
    margin: 0.3in 0;
    border-left: 4px solid #007acc;
    font-style: italic;
}

.abstract h2 {
    font-size: 12pt;
    margin-bottom: 0.2in;
    color: #007acc;
}

/* Títulos de sección */
h2 {
    color: #1a472a;
    font-size: 14pt;
    font-weight: bold;
    margin: 0.4in 0 0.2in 0;
    page-break-after: avoid;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.1in;
}

h3 {
    color: #2d5a3d;
    font-size: 12pt;
    font-weight: bold;
    margin: 0.3in 0 0.15in 0;
    page-break-after: avoid;
}

h4 {
    color: #4a6741;
    font-size: 11pt;
    font-weight: bold;
    margin: 0.2in 0 0.1in 0;
    page-break-after: avoid;
}

/* Párrafos */
p {
    margin-bottom: 0.15in;
    text-indent: 0.2in;
}

/* Listas */
ul, ol {
    margin: 0.1in 0 0.15in 0.4in;
    padding-left: 0.2in;
}

li {
    margin-bottom: 0.05in;
}

/* Énfasis */
strong {
    font-weight: bold;
    color: #1a472a;
}

em {
    font-style: italic;
}

/* Tablas */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.2in 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th {
    background-color: #f1f3f4;
    border: 1px solid #ddd;
    padding: 0.1in;
    text-align: center;
    font-weight: bold;
    color: #1a472a;
}

td {
    border: 1px solid #ddd;
    padding: 0.08in;
    text-align: center;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Caption de tablas */
.table-caption {
    font-size: 10pt;
    font-weight: bold;
    text-align: center;
    margin: 0.1in 0;
    color: #1a472a;
}

/* Figuras */
figure {
    margin: 0.2in 0;
    text-align: center;
    page-break-inside: avoid;
}

img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

figcaption {
    font-size: 10pt;
    font-weight: bold;
    margin-top: 0.1in;
    color: #1a472a;
}

/* Código */
code {
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 0.02in 0.05in;
    border-radius: 2px;
    color: #d73a49;
}

pre {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 3px;
    padding: 0.15in;
    margin: 0.1in 0;
    overflow-x: auto;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
    color: inherit;
}

/* Ecuaciones matemáticas */
.math {
    font-family: 'Times New Roman', serif;
    font-style: italic;
}

/* Enlaces */
a {
    color: #007acc;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Notas al pie */
.footnote {
    font-size: 9pt;
    margin-top: 0.3in;
    border-top: 1px solid #ccc;
    padding-top: 0.1in;
}

/* Márgenes para impresión */
@media print {
    body {
        margin: 0;
        padding: 0.5in;
        font-size: 11pt;
    }
    
    h1 { page-break-after: avoid; }
    h2, h3, h4 { page-break-after: avoid; }
    table, img, figure { page-break-inside: avoid; }
    
    a {
        color: black;
        text-decoration: none;
    }
    
    .no-print {
        display: none;
    }
}

/* Estilos específicos del proyecto */
.metrics-highlight {
    background-color: #e8f5e8;
    padding: 0.05in 0.1in;
    border-radius: 3px;
    font-weight: bold;
    color: #1a472a;
}

.hypothesis-box {
    border: 1px solid #007acc;
    background-color: #f0f8ff;
    padding: 0.15in;
    margin: 0.1in 0;
    border-radius: 3px;
}

.conclusion-box {
    border: 2px solid #1a472a;
    background-color: #f8fff8;
    padding: 0.2in;
    margin: 0.2in 0;
    border-radius: 5px;
}

/* Tabla de contenidos */
.toc {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 0.2in;
    margin: 0.2in 0;
    border-radius: 3px;
}

.toc h2 {
    margin-top: 0;
    color: #1a472a;
    border-bottom: none;
}

.toc ul {
    list-style-type: none;
    margin-left: 0;
    padding-left: 0;
}

.toc li {
    margin: 0.05in 0;
}

.toc a {
    color: #007acc;
    text-decoration: none;
}

.toc a:hover {
    text-decoration: underline;
}
"""
    
    with open(CSS_FILE, 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    print(f"✓ CSS académico creado: {CSS_FILE}")

def convert_markdown_to_html():
    """Convierte el archivo Markdown a HTML con formato académico"""
    
    if not os.path.exists(MARKDOWN_FILE):
        print(f"❌ Archivo {MARKDOWN_FILE} no encontrado")
        return False
    
    try:
        # Leer markdown
        with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convertir usando pandoc si está disponible
        try:
            result = subprocess.run([
                'pandoc', 
                MARKDOWN_FILE, 
                '-o', HTML_FILE,
                '--standalone',
                '--css', CSS_FILE,
                '--mathjax',
                '--toc',
                '--toc-depth=3',
                '--metadata', 'title="Comparación KAN vs MLP en Predicción de Ventas"',
                '--metadata', 'author="Edgar Alberto Morales Gutiérrez"'
            ], check=True, capture_output=True, text=True)
            
            print("✓ Conversión exitosa usando Pandoc")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Pandoc no disponible, usando conversión básica...")
            return convert_markdown_basic(content)
            
    except Exception as e:
        print(f"❌ Error convirtiendo Markdown: {e}")
        return False

def convert_markdown_basic(content: str) -> bool:
    """Conversión básica de Markdown a HTML"""
    
    try:
        # Procesar contenido básico
        html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparación KAN vs MLP en Predicción de Ventas - Edgar Alberto Morales Gutiérrez</title>
    <meta name="author" content="Edgar Alberto Morales Gutiérrez">
    <link rel="stylesheet" href="{CSS_FILE}">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }}
    }};
    </script>
</head>
<body>
"""
        
        # Procesar el contenido Markdown línea por línea
        lines = content.split('\n')
        in_code_block = False
        in_table = False
        
        for line in lines:
            line = line.rstrip()
            
            # Bloques de código
            if line.startswith('```'):
                if in_code_block:
                    html_content += "</pre>\n"
                    in_code_block = False
                else:
                    html_content += "<pre><code>\n"
                    in_code_block = True
                continue
            
            if in_code_block:
                html_content += line + "\n"
                continue
            
            # Títulos
            if line.startswith('# '):
                html_content += f"<h1>{line[2:]}</h1>\n"
            elif line.startswith('## '):
                html_content += f"<h2>{line[3:]}</h2>\n"
            elif line.startswith('### '):
                html_content += f"<h3>{line[4:]}</h3>\n"
            elif line.startswith('#### '):
                html_content += f"<h4>{line[5:]}</h4>\n"
            
            # Tablas (detectar por |)
            elif '|' in line and line.strip():
                if not in_table:
                    html_content += "<table>\n"
                    in_table = True
                
                # Procesar fila de tabla
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if all(cell.replace('-', '').replace(' ', '') == '' for cell in cells):
                    # Línea separadora de encabezado
                    continue
                
                # Determinar si es encabezado (primera fila)
                tag = 'th' if '**' in line else 'td'
                html_content += "<tr>\n"
                for cell in cells:
                    # Limpiar markdown básico
                    cell = cell.replace('**', '').replace('*', '')
                    html_content += f"  <{tag}>{cell}</{tag}>\n"
                html_content += "</tr>\n"
            
            # Fin de tabla
            elif in_table and not line.strip():
                html_content += "</table>\n"
                in_table = False
                html_content += "<p></p>\n"
            
            # Listas
            elif line.startswith('- ') or line.startswith('* '):
                html_content += f"<li>{line[2:]}</li>\n"
            elif line.startswith('1. ') or re.match(r'^\d+\. ', line):
                content_part = re.sub(r'^\d+\. ', '', line)
                html_content += f"<li>{content_part}</li>\n"
            
            # Párrafos normales
            elif line.strip():
                # Procesar markdown básico en línea
                processed_line = line
                processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_line)
                processed_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_line)
                processed_line = re.sub(r'`(.*?)`', r'<code>\1</code>', processed_line)
                
                html_content += f"<p>{processed_line}</p>\n"
            else:
                html_content += "<br>\n"
        
        # Cerrar tabla si quedó abierta
        if in_table:
            html_content += "</table>\n"
        
        html_content += """
</body>
</html>"""
        
        # Guardar HTML
        with open(HTML_FILE, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML generado: {HTML_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ Error en conversión básica: {e}")
        return False

def try_pdf_conversion():
    """Intenta convertir HTML a PDF usando diferentes métodos"""
    
    methods = [
        ("wkhtmltopdf", lambda: subprocess.run([
            'wkhtmltopdf', 
            '--page-size', 'A4',
            '--margin-top', '1in',
            '--margin-bottom', '1in', 
            '--margin-left', '1in',
            '--margin-right', '1in',
            '--enable-local-file-access',
            HTML_FILE, PDF_FILE
        ], check=True)),
        
        ("weasyprint", lambda: subprocess.run([
            'weasyprint', HTML_FILE, PDF_FILE
        ], check=True)),
        
        ("chrome/chromium", lambda: subprocess.run([
            'chrome', '--headless', '--disable-gpu', '--print-to-pdf=' + PDF_FILE,
            'file://' + os.path.abspath(HTML_FILE)
        ], check=True))
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"⚙ Intentando conversión PDF con {method_name}...")
            method_func()
            if os.path.exists(PDF_FILE):
                print(f"✓ PDF generado con {method_name}: {PDF_FILE}")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠ {method_name} no disponible")
            continue
    
    print("⚠ No se pudo generar PDF automáticamente")
    print("💡 Puedes abrir el HTML en un navegador e imprimir como PDF")
    return False

def validate_files():
    """Valida los archivos generados"""
    results = {}
    
    # Validar HTML
    if os.path.exists(HTML_FILE):
        size = os.path.getsize(HTML_FILE)
        results['html'] = f"✓ HTML válido ({size:,} bytes)"
    else:
        results['html'] = "❌ HTML no encontrado"
    
    # Validar CSS
    if os.path.exists(CSS_FILE):
        size = os.path.getsize(CSS_FILE)
        results['css'] = f"✓ CSS creado ({size:,} bytes)"
    else:
        results['css'] = "❌ CSS no encontrado"
    
    # Validar PDF
    if os.path.exists(PDF_FILE):
        size = os.path.getsize(PDF_FILE)
        results['pdf'] = f"✓ PDF generado ({size:,} bytes)"
    else:
        results['pdf'] = "⚠ PDF no generado (usar navegador para crear)"
    
    return results

def main():
    """Función principal del generador alternativo"""
    print("🚀 GENERADOR ALTERNATIVO - Proyecto KAN vs MLP")
    print("=" * 55)
    
    # Verificar archivo fuente
    if not os.path.exists(MARKDOWN_FILE):
        print(f"❌ Archivo {MARKDOWN_FILE} no encontrado")
        return False
    
    success_count = 0
    
    # 1. Crear CSS académico
    try:
        create_academic_css()
        success_count += 1
    except Exception as e:
        print(f"❌ Error creando CSS: {e}")
    
    # 2. Convertir a HTML
    try:
        if convert_markdown_to_html():
            success_count += 1
            print("✓ Conversión HTML completada")
        else:
            print("❌ Falló conversión HTML")
    except Exception as e:
        print(f"❌ Error en conversión HTML: {e}")
    
    # 3. Intentar conversión a PDF
    try:
        if try_pdf_conversion():
            success_count += 1
    except Exception as e:
        print(f"❌ Error en conversión PDF: {e}")
    
    # 4. Validar resultados
    results = validate_files()
    
    print("\n📊 RESULTADOS FINALES:")
    print("=" * 30)
    for file_type, status in results.items():
        print(f"{file_type.upper()}: {status}")
    
    # Mostrar rutas absolutas
    if os.path.exists(HTML_FILE):
        print(f"\n📍 HTML: {os.path.abspath(HTML_FILE)}")
    if os.path.exists(PDF_FILE):
        print(f"📍 PDF: {os.path.abspath(PDF_FILE)}")
    
    print(f"\n✅ Procesos exitosos: {success_count}/3")
    
    if success_count >= 2:
        print("\n🎉 ¡GENERACIÓN EXITOSA!")
        print("💡 Abre el archivo HTML en tu navegador para ver el documento")
        if not os.path.exists(PDF_FILE):
            print("💡 Para PDF: abre HTML en navegador → Imprimir → Guardar como PDF")
        return True
    else:
        print("\n⚠ Generación parcialmente exitosa")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹ Generación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)