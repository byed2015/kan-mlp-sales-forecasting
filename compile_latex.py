"""
Script para compilar documento LaTeX a PDF
Automatiza la compilación del proyecto final KAN vs MLP

Características:
- Detección automática de instalación LaTeX
- Creación de imágenes placeholder si es necesario
- Manejo de múltiples pasadas de compilación
- Limpieza automática de archivos temporales
- Validación de PDF generado
"""

import os
import subprocess
import sys
import shutil
import json
from pathlib import Path
from typing import Optional, List, Tuple
import tempfile

# Configuración
LATEX_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.tex"
PDF_FILE = "PROYECTO_FINAL_KAN_MLP_SALES.pdf"
REPORTS_DIR = "reports"

# Imágenes requeridas (con fallbacks)
REQUIRED_IMAGES = [
    ("reports/figura_1_arquitectura_comparativa.png", "Arquitectura Comparativa KAN vs MLP"),
    ("reports/figura_3_comparacion_rendimiento.png", "Comparación de Rendimiento")
]

# Dependencias LaTeX requeridas
LATEX_PACKAGES = [
    "inputenc", "babel", "geometry", "graphicx", "float", 
    "amsmath", "amssymb", "amsthm", "booktabs", "longtable",
    "array", "multirow", "multicol", "xcolor", "listings",
    "hyperref", "url", "fancyhdr", "titlesec", "enumitem",
    "caption", "subcaption"
]

def check_latex_installation() -> Tuple[bool, Optional[str]]:
    """
    Verifica si LaTeX está instalado y disponible
    
    Returns:
        (bool, str): (disponible, comando_encontrado)
    """
    latex_commands = ['pdflatex', 'xelatex', 'lualatex']
    
    for cmd in latex_commands:
        try:
            result = subprocess.run(
                [cmd, '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ LaTeX encontrado: {cmd}")
                return True, cmd
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    return False, None

def suggest_latex_installation():
    """Sugiere cómo instalar LaTeX según el sistema operativo"""
    import platform
    
    system = platform.system().lower()
    
    suggestions = {
        'windows': [
            "Instalar MiKTeX: https://miktex.org/download",
            "O TeX Live: https://tug.org/texlive/windows.html",
            "Comando: winget install MiKTeX.MiKTeX"
        ],
        'darwin': [  # macOS
            "Instalar MacTeX: https://tug.org/mactex/",
            "O con Homebrew: brew install --cask mactex"
        ],
        'linux': [
            "Ubuntu/Debian: sudo apt-get install texlive-full",
            "Fedora: sudo dnf install texlive-scheme-full",
            "Arch: sudo pacman -S texlive-most"
        ]
    }
    
    print(f"\n🔧 Para instalar LaTeX en {system.title()}:")
    for suggestion in suggestions.get(system, ["Consultar: https://www.latex-project.org/get/"]):
        print(f"   • {suggestion}")

def create_placeholder_image(filepath: str, title: str, size: Tuple[int, int] = (800, 600)):
    """
    Crea una imagen placeholder si no existe la original
    
    Args:
        filepath: Ruta donde crear la imagen
        title: Título para mostrar en la imagen
        size: Dimensiones (ancho, alto)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(size[0]/100, size[1]/100), dpi=100)
        
        # Fondo gris claro
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, 
                                     facecolor='lightgray', 
                                     edgecolor='darkgray', 
                                     linewidth=2))
        
        # Texto principal
        ax.text(0.5, 0.6, title, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, fontweight='bold',
                wrap=True)
        
        # Subtexto
        ax.text(0.5, 0.4, "[Imagen Placeholder]\nGenerar figura desde notebooks", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10, style='italic',
                color='gray')
        
        # Configuración
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Guardar
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Placeholder creado: {filepath}")
        return True
        
    except ImportError:
        print(f"⚠ matplotlib no disponible, creando archivo vacío: {filepath}")
        # Crear archivo vacío para evitar errores LaTeX
        Path(filepath).touch()
        return False
    except Exception as e:
        print(f"❌ Error creando placeholder {filepath}: {e}")
        return False

def check_and_create_images():
    """Verifica y crea imágenes necesarias"""
    print("\n📸 Verificando imágenes...")
    
    missing_images = []
    
    for img_path, img_title in REQUIRED_IMAGES:
        if not os.path.exists(img_path):
            print(f"⚠ Imagen faltante: {img_path}")
            missing_images.append((img_path, img_title))
        else:
            print(f"✓ Imagen encontrada: {img_path}")
    
    # Crear placeholders para imágenes faltantes
    if missing_images:
        print(f"\n🎨 Creando {len(missing_images)} placeholders...")
        for img_path, img_title in missing_images:
            create_placeholder_image(img_path, img_title)
    
    return len(missing_images) == 0

def compile_latex(latex_cmd: str, max_runs: int = 3) -> bool:
    """
    Compila el documento LaTeX
    
    Args:
        latex_cmd: Comando LaTeX a usar (pdflatex, xelatex, etc.)
        max_runs: Máximo número de pasadas de compilación
        
    Returns:
        bool: True si la compilación fue exitosa
    """
    print(f"\n🔨 Compilando con {latex_cmd}...")
    
    # Configurar entorno
    env = os.environ.copy()
    env['TEXINPUTS'] = '.:' + env.get('TEXINPUTS', '')
    
    success = False
    
    for run in range(1, max_runs + 1):
        print(f"   Pasada {run}/{max_runs}...")
        
        try:
            # Ejecutar compilación
            result = subprocess.run(
                [latex_cmd, '-interaction=nonstopmode', LATEX_FILE],
                cwd=os.getcwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutos timeout
            )
            
            # Verificar resultado
            if result.returncode == 0:
                if os.path.exists(PDF_FILE):
                    success = True
                    print(f"   ✓ Pasada {run} exitosa")
                    
                    # Si es la última pasada o ya no hay referencias pendientes
                    if run == max_runs or not needs_rerun(result.stdout):
                        break
                else:
                    print(f"   ⚠ Pasada {run} completó pero PDF no generado")
            else:
                print(f"   ❌ Pasada {run} falló (código: {result.returncode})")
                # En la primera pasada, mostrar errores para debug
                if run == 1:
                    print_latex_errors(result.stdout)
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Pasada {run} timeout (>2 min)")
        except Exception as e:
            print(f"   ❌ Error en pasada {run}: {e}")
    
    return success

def needs_rerun(latex_output: str) -> bool:
    """
    Determina si se necesita otra pasada de compilación
    """
    rerun_indicators = [
        "Rerun to get cross-references right",
        "Label(s) may have changed",
        "There were undefined references"
    ]
    
    return any(indicator in latex_output for indicator in rerun_indicators)

def print_latex_errors(latex_output: str):
    """
    Extrae y muestra errores de LaTeX de forma legible
    """
    lines = latex_output.split('\n')
    in_error = False
    error_count = 0
    
    print("\n📋 Errores de LaTeX detectados:")
    
    for line in lines:
        if line.startswith('!'):
            in_error = True
            error_count += 1
            print(f"\n❌ Error {error_count}: {line}")
        elif in_error and line.startswith('l.'):
            print(f"   📍 Línea: {line}")
            in_error = False
        elif "Missing" in line or "Undefined" in line:
            print(f"⚠ Advertencia: {line}")
    
    if error_count == 0:
        print("   (No se detectaron errores específicos)")

def cleanup_temp_files():
    """
    Limpia archivos temporales de LaTeX
    """
    temp_extensions = ['.aux', '.log', '.out', '.toc', '.fdb_latexmk', '.fls', '.synctex.gz']
    base_name = LATEX_FILE.rsplit('.', 1)[0]
    
    cleaned = 0
    for ext in temp_extensions:
        temp_file = base_name + ext
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                cleaned += 1
            except Exception as e:
                print(f"⚠ No se pudo eliminar {temp_file}: {e}")
    
    if cleaned > 0:
        print(f"🧹 Limpiados {cleaned} archivos temporales")

def validate_pdf():
    """
    Valida que el PDF generado sea válido
    """
    if not os.path.exists(PDF_FILE):
        return False, "PDF no encontrado"
    
    # Verificar tamaño
    size = os.path.getsize(PDF_FILE)
    if size < 1024:  # Menos de 1KB es sospechoso
        return False, f"PDF muy pequeño ({size} bytes)"
    
    # Verificar que sea un PDF válido (header básico)
    try:
        with open(PDF_FILE, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False, "No es un archivo PDF válido"
    except Exception as e:
        return False, f"Error leyendo PDF: {e}"
    
    return True, f"PDF válido ({size:,} bytes)"

def main():
    """Función principal del script"""
    print("🚀 COMPILADOR LaTeX - Proyecto KAN vs MLP")
    print("=" * 50)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists(LATEX_FILE):
        print(f"❌ Archivo {LATEX_FILE} no encontrado en directorio actual")
        print(f"📂 Directorio actual: {os.getcwd()}")
        print("\n💡 Asegúrate de ejecutar desde la carpeta del proyecto")
        return False
    
    # 1. Verificar LaTeX
    latex_available, latex_cmd = check_latex_installation()
    if not latex_available:
        print("❌ LaTeX no encontrado en el sistema")
        suggest_latex_installation()
        return False
    
    # 2. Verificar/crear imágenes
    all_images_found = check_and_create_images()
    if not all_images_found:
        print("⚠ Algunas imágenes fueron reemplazadas por placeholders")
    
    # 3. Compilar LaTeX
    compilation_success = compile_latex(latex_cmd)
    
    # 4. Limpiar archivos temporales
    cleanup_temp_files()
    
    # 5. Validar resultado
    if compilation_success:
        is_valid, message = validate_pdf()
        if is_valid:
            print(f"\n🎉 ¡COMPILACIÓN EXITOSA!")
            print(f"📄 PDF generado: {PDF_FILE}")
            print(f"📊 {message}")
            
            # Mostrar información adicional
            abs_path = os.path.abspath(PDF_FILE)
            print(f"📍 Ruta completa: {abs_path}")
            
            return True
        else:
            print(f"\n❌ PDF generado pero inválido: {message}")
            return False
    else:
        print(f"\n❌ COMPILACIÓN FALLÓ")
        print("\n🔍 Pasos para solucionar:")
        print("   1. Verificar errores LaTeX mostrados arriba")
        print("   2. Instalar paquetes LaTeX faltantes")
        print("   3. Verificar sintaxis del archivo .tex")
        print("   4. Ejecutar manualmente: pdflatex PROYECTO_FINAL_KAN_MLP_SALES.tex")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹ Compilación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)