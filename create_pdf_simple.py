"""
Generador de PDF mejorado usando ReportLab
Convierte el contenido del documento a PDF con formato académico completo

Características:
- Conversión directa de texto a PDF
- Formato académico profesional
- Manejo completo de texto, títulos, tablas e imágenes
- Procesamiento correcto de markdown
- Manejo de imágenes y figuras
"""

import os
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas

def clean_markdown_text(text):
    """Limpia el markdown de un texto para ReportLab"""
    # Corregir los escapes de regex - usar r'\1' en lugar de '\\1'
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
    return text

def process_table_data(lines, start_idx):
    """Procesa datos de tabla de manera más robusta"""
    table_data = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Si no hay línea o no contiene |, terminar tabla
        if not line or '|' not in line:
            break
            
        # Ignorar líneas separadoras (solo contienen -, |, y espacios)
        if all(c in '-| ' for c in line):
            i += 1
            continue
        
        # Procesar fila de datos
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if cells:
            # Limpiar markdown en cada celda
            clean_cells = [clean_markdown_text(cell) for cell in cells]
            table_data.append(clean_cells)
        
        i += 1
    
    return table_data, i

def add_image_if_exists(story, image_path, caption=""):
    """Agrega imagen al documento si existe"""
    if os.path.exists(image_path):
        try:
            # Crear imagen
            img = Image(image_path, width=6*inch, height=4*inch)
            story.append(img)
            
            # Agregar caption si se proporciona
            if caption:
                caption_style = ParagraphStyle(
                    'ImageCaption',
                    fontSize=9,
                    alignment=TA_CENTER,
                    spaceAfter=12,
                    textColor=colors.darkblue
                )
                story.append(Paragraph(f"<b>{caption}</b>", caption_style))
            
            story.append(Spacer(1, 12))
            print(f"✅ Imagen agregada: {image_path}")
            return True
        except Exception as e:
            print(f"⚠ Error cargando imagen {image_path}: {e}")
    else:
        print(f"⚠ Imagen no encontrada: {image_path}")
    return False

def create_pdf_document():
    """Crear documento PDF usando ReportLab con procesamiento mejorado"""
    
    MARKDOWN_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.md"
    PDF_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.pdf"
    
    if not os.path.exists(MARKDOWN_FILE):
        print(f"❌ Archivo {MARKDOWN_FILE} no encontrado")
        return False
    
    try:
        # Leer contenido markdown
        with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Crear documento PDF
        doc = SimpleDocTemplate(
            PDF_FILE,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Estilos
        styles = getSampleStyleSheet()
        
        # Estilos personalizados mejorados
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        author_style = ParagraphStyle(
            'AuthorStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=15,
            alignment=TA_CENTER,
            textColor=colors.darkgreen,
            fontName='Helvetica'
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0
        )
        
        # Lista para almacenar elementos del documento
        story = []
        
        # Procesar contenido línea por línea
        lines = content.split('\n')
        in_code_block = False
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Líneas vacías
            if not line:
                if not in_code_block:
                    story.append(Spacer(1, 6))
                i += 1
                continue
            
            # Bloques de código
            if line.startswith('```'):
                in_code_block = not in_code_block
                i += 1
                continue
            
            if in_code_block:
                i += 1
                continue
            
            # Procesar imágenes (detectar por patrones comunes)
            if ('![' in line or 'figura_' in line.lower() or '.png' in line.lower() or '.jpg' in line.lower()):
                # Intentar extraer ruta de imagen
                img_match = re.search(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
                if img_match:
                    caption = img_match.group(1)
                    img_path = img_match.group(2)
                    add_image_if_exists(story, img_path, caption)
                else:
                    # Buscar rutas de imagen conocidas
                    if 'figura_1' in line.lower():
                        add_image_if_exists(story, 'reports/figura_1_arquitectura_comparativa.png', 
                                          'Figura 1: Arquitectura Comparativa KAN vs MLP')
                    elif 'figura_3' in line.lower():
                        add_image_if_exists(story, 'reports/figura_3_comparacion_rendimiento.png',
                                          'Figura 3: Comparación de Rendimiento')
                i += 1
                continue
            
            # Procesar tablas
            if '|' in line and not line.startswith('#'):
                table_data, next_i = process_table_data(lines, i)
                if table_data:
                    # Crear tabla con mejor formato
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 12))
                    print(f"✅ Tabla procesada con {len(table_data)} filas")
                i = next_i
                continue
            
            # Títulos
            if line.startswith('# '):
                text = clean_markdown_text(line[2:])
                story.append(Paragraph(text, title_style))
                
            elif line.startswith('## '):
                text = clean_markdown_text(line[3:])
                story.append(Paragraph(text, heading1_style))
                
            elif line.startswith('### '):
                text = clean_markdown_text(line[4:])
                story.append(Paragraph(text, heading2_style))
                
            elif line.startswith('#### '):
                text = clean_markdown_text(line[5:])
                story.append(Paragraph(text, heading2_style))
            
            # Información de autor
            elif line.startswith('**Autor**:'):
                text = clean_markdown_text(line)
                story.append(Paragraph(text, author_style))
                
            # Listas
            elif line.startswith('- ') or line.startswith('* '):
                text = clean_markdown_text(line[2:])
                story.append(Paragraph(f"• {text}", body_style))
                
            elif re.match(r'^\d+\. ', line):
                text = clean_markdown_text(re.sub(r'^\d+\. ', '', line))
                story.append(Paragraph(f"1. {text}", body_style))
                
            # Párrafos normales
            else:
                text = clean_markdown_text(line)
                if text:
                    story.append(Paragraph(text, body_style))
            
            i += 1
        
        # Construir PDF
        print("🔨 Generando PDF mejorado...")
        doc.build(story)
        
        # Verificar resultado
        if os.path.exists(PDF_FILE):
            size = os.path.getsize(PDF_FILE)
            print(f"✅ PDF generado exitosamente: {PDF_FILE}")
            print(f"📊 Tamaño: {size:,} bytes")
            print(f"📍 Ruta: {os.path.abspath(PDF_FILE)}")
            return True
        else:
            print("❌ Error: PDF no fue creado")
            return False
            
    except Exception as e:
        print(f"❌ Error generando PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 GENERADOR PDF - ReportLab")
    print("=" * 35)
    
    success = create_pdf_document()
    
    if success:
        print("\n🎉 ¡PDF GENERADO EXITOSAMENTE!")
        print("💡 El documento está listo para uso académico")
    else:
        print("\n❌ Falló la generación del PDF")
    
    return success

if __name__ == "__main__":
    main()