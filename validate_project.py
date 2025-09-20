"""
Resumen final del proyecto - Verificación de documentos generados
Valida todos los archivos creados y proporciona estadísticas finales

Este script verifica la integridad y completitud de todos los documentos
generados para el proyecto final de comparación KAN vs MLP.
"""

import os
import json
from datetime import datetime
from pathlib import Path

def get_file_info(filepath):
    """Obtiene información de un archivo"""
    if os.path.exists(filepath):
        stat = os.stat(filepath)
        return {
            'exists': True,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / 1024 / 1024, 2),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        return {'exists': False}

def validate_project_structure():
    """Valida la estructura completa del proyecto"""
    
    print("🔍 VALIDACIÓN FINAL DEL PROYECTO")
    print("=" * 50)
    
    # Documentos principales
    main_docs = {
        'Markdown Original': 'PROYECTO_FINAL_KAN_MLP_SALES.md',
        'Documento HTML': 'PROYECTO_FINAL_KAN_MLP_SALES.html', 
        'Documento PDF': 'PROYECTO_FINAL_KAN_MLP_SALES.pdf',
        'Documento LaTeX': 'PROYECTO_FINAL_KAN_MLP_SALES.tex',
        'CSS Académico': 'academic_style.css'
    }
    
    # Scripts generados
    scripts = {
        'Compilador LaTeX': 'compile_latex.py',
        'Generador HTML': 'generate_document.py',
        'Generador PDF Simple': 'create_pdf_simple.py'
    }
    
    # Validar documentos principales
    print("\n📄 DOCUMENTOS PRINCIPALES:")
    print("-" * 30)
    
    total_size = 0
    docs_found = 0
    
    for name, filepath in main_docs.items():
        info = get_file_info(filepath)
        if info['exists']:
            print(f"✅ {name}")
            print(f"   📁 {filepath}")
            print(f"   📊 Tamaño: {info['size']:,} bytes ({info['size_mb']} MB)")
            print(f"   🕒 Modificado: {info['modified']}")
            total_size += info['size']
            docs_found += 1
        else:
            print(f"❌ {name} - NO ENCONTRADO")
        print()
    
    # Validar scripts
    print("🛠️ SCRIPTS GENERADOS:")
    print("-" * 25)
    
    scripts_found = 0
    for name, filepath in scripts.items():
        info = get_file_info(filepath)
        if info['exists']:
            print(f"✅ {name} ({info['size']:,} bytes)")
            scripts_found += 1
        else:
            print(f"❌ {name} - NO ENCONTRADO")
    
    # Validar estructura de datos
    print("\n📊 ESTRUCTURA DE DATOS:")
    print("-" * 28)
    
    data_dirs = ['data/raw', 'data/processed', 'models', 'reports', 'notebooks']
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_path}: {files} archivos")
        else:
            print(f"❌ {dir_path}: NO ENCONTRADO")
    
    # Validar notebooks específicos
    print("\n📓 NOTEBOOKS DEL PROYECTO:")
    print("-" * 30)
    
    expected_notebooks = [
        '00_env_check.ipynb',
        '01_preprocessing_eda_v2_fixed.ipynb', 
        '02_baseline_mlp_improved_fixed.ipynb',
        '03_kan_model_fixed.ipynb',
        '04_eval_compare_fixed.ipynb',
        '05_kan_deep_analysis.ipynb'
    ]
    
    notebooks_found = 0
    for nb in expected_notebooks:
        nb_path = f"notebooks/{nb}"
        if os.path.exists(nb_path):
            info = get_file_info(nb_path)
            print(f"✅ {nb} ({info['size_mb']} MB)")
            notebooks_found += 1
        else:
            print(f"❌ {nb} - NO ENCONTRADO")
    
    # Validar modelos entrenados
    print("\n🧠 MODELOS ENTRENADOS:")
    print("-" * 24)
    
    expected_models = [
        'baseline_mlp.pt',
        'baseline_mlp_improved.pt', 
        'kan_model.pt',
        'simplified_kan_model.pt'
    ]
    
    models_found = 0
    for model in expected_models:
        model_path = f"models/{model}"
        if os.path.exists(model_path):
            info = get_file_info(model_path)
            print(f"✅ {model} ({info['size_mb']} MB)")
            models_found += 1
        else:
            print(f"❌ {model} - NO ENCONTRADO")
    
    # Validar reportes y figuras
    print("\n📈 REPORTES Y FIGURAS:")
    print("-" * 26)
    
    if os.path.exists('reports'):
        report_files = os.listdir('reports')
        json_files = [f for f in report_files if f.endswith('.json')]
        png_files = [f for f in report_files if f.endswith('.png')]
        csv_files = [f for f in report_files if f.endswith('.csv')]
        
        print(f"✅ Métricas JSON: {len(json_files)} archivos")
        print(f"✅ Figuras PNG: {len(png_files)} archivos") 
        print(f"✅ Predicciones CSV: {len(csv_files)} archivos")
        
        # Mostrar archivos importantes
        important_files = [
            'kan_metrics.json',
            'baseline_mlp_improved_metrics.json',
            'figura_1_arquitectura_comparativa.png',
            'figura_3_comparacion_rendimiento.png'
        ]
        
        for file in important_files:
            if file in report_files:
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file}")
    
    # Generar reporte de completitud
    print("\n" + "=" * 50)
    print("📋 REPORTE DE COMPLETITUD")
    print("=" * 50)
    
    print(f"📄 Documentos principales: {docs_found}/5 ({docs_found/5*100:.0f}%)")
    print(f"🛠️ Scripts generados: {scripts_found}/3 ({scripts_found/3*100:.0f}%)")
    print(f"📓 Notebooks: {notebooks_found}/6 ({notebooks_found/6*100:.0f}%)")
    print(f"🧠 Modelos: {models_found}/4 ({models_found/4*100:.0f}%)")
    print(f"💾 Tamaño total documentos: {total_size:,} bytes ({round(total_size/1024/1024, 2)} MB)")
    
    overall_completion = (docs_found + scripts_found + notebooks_found + models_found) / 18 * 100
    print(f"\n🎯 COMPLETITUD GENERAL: {overall_completion:.1f}%")
    
    if overall_completion >= 90:
        print("🎉 ¡PROYECTO COMPLETADO CON ÉXITO!")
        print("✨ Todos los componentes esenciales están presentes")
    elif overall_completion >= 75:
        print("✅ Proyecto mayormente completo")
        print("💡 Algunos componentes opcionales pueden faltar")
    else:
        print("⚠️ Proyecto parcialmente completo")
        print("🔧 Revisa los componentes faltantes")
    
    # Información de entrega
    print("\n" + "=" * 50)
    print("📦 INFORMACIÓN DE ENTREGA")
    print("=" * 50)
    
    deliverables = []
    
    if os.path.exists('PROYECTO_FINAL_KAN_MLP_SALES.pdf'):
        deliverables.append("📄 Documento PDF (Principal)")
    
    if os.path.exists('PROYECTO_FINAL_KAN_MLP_SALES.html'):
        deliverables.append("🌐 Documento HTML (Alternativo)")
        
    if os.path.exists('PROYECTO_FINAL_KAN_MLP_SALES.tex'):
        deliverables.append("📝 Código LaTeX (Fuente)")
    
    if notebooks_found >= 5:
        deliverables.append("📓 Notebooks Jupyter (Código fuente)")
        
    if models_found >= 2:
        deliverables.append("🧠 Modelos entrenados (.pt)")
    
    print("🎁 ENTREGABLES LISTOS:")
    for item in deliverables:
        print(f"   {item}")
    
    print(f"\n📍 Ubicación del proyecto:")
    print(f"   {os.path.abspath('.')}")
    
    print(f"\n🕒 Validación completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_completion

def main():
    """Función principal"""
    try:
        completion = validate_project_structure()
        return completion >= 75  # Considera exitoso si >= 75%
    except Exception as e:
        print(f"❌ Error durante validación: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)