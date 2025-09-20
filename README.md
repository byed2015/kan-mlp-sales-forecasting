# Comparación de Redes Kolmogorov-Arnold (KAN) vs. Perceptrones Multicapa (MLP)
## Estudio Empírico Comprensivo en Predicción de Ventas

**Autor:** Edgar Alberto Morales Gutiérrez  
**Proyecto Final - Aprendizaje Profundo**  
**Universidad Panamericana**  
**Fecha:** Septiembre 2025

---

## 🎯 Resumen del Proyecto

Este proyecto presenta una comparación empírica comprensiva entre las Redes Kolmogorov-Arnold (KAN) y los Perceptrones Multicapa (MLP) tradicionales para la predicción de ventas semanales usando el dataset completo de Walmart Sales. Las KAN, introducidas por Liu et al. (2024), reemplazan las funciones de activación fijas con funciones spline aprendibles en las conexiones, prometiendo interpretabilidad y eficiencia mejoradas.

### 🏆 Resultados Principales

- **KAN supera a MLP:** R² = 0.9790 vs R² = 0.9549 (+2.4 puntos porcentuales)
- **Mejor eficiencia:** RMSE 31.8% menor, MAE 35.4% menor
- **Interpretabilidad:** Funciones spline revelan patrones económicos interpretables
- **Eficiencia paramétrica:** Rendimiento superior con parámetros similares (13,889 vs 13,441)

---

## 📋 Estructura del Proyecto

```
kan_mlp_sales/
├── data/
│   ├── raw/                    # Datos originales Kaggle
│   └── processed/              # Datos procesados + metadata
├── notebooks/                  # Análisis completo reproducible
│   ├── 00_env_check.ipynb     # Verificación entorno
│   ├── 01_preprocessing_eda_v2_fixed.ipynb
│   ├── 02_baseline_mlp_improved_fixed.ipynb  
│   ├── 03_kan_model_fixed.ipynb
│   ├── 04_eval_compare_fixed.ipynb
│   └── 05_kan_deep_analysis.ipynb  # ⭐ Análisis avanzado
├── models/                     # Modelos entrenados (.pt)
├── reports/                    # Resultados y figuras
├── utils/                      # Utilidades comunes
├── PROYECTO_FINAL_KAN_MLP_SALES.md    # 📄 Documento principal
├── PROYECTO_FINAL_KAN_MLP_SALES.pdf   # 📄 Documento académico
├── PROYECTO_FINAL_KAN_MLP_SALES.html  # 🌐 Versión web
├── compile_latex.py            # 🛠️ Generador LaTeX
├── create_pdf_simple.py        # 🛠️ Generador PDF
└── generate_document.py        # 🛠️ Generador HTML
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python:** 3.13.1 o superior
- **PyTorch:** 2.0+ con soporte CUDA (recomendado)
- **Hardware:** GPU CUDA compatible, 16GB+ RAM recomendado

### Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd kan_mlp_sales
   ```

2. **Crear entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install torch torchvision torchaudio
   pip install pandas numpy scikit-learn  
   pip install matplotlib seaborn plotly
   pip install jupyter notebook
   pip install reportlab  # Para generación PDF
   ```

---

## 📊 Uso del Proyecto

### 1. Ejecución de Notebooks

Los notebooks están diseñados para ejecutarse en orden secuencial:

```bash
jupyter notebook notebooks/00_env_check.ipynb
```

**Secuencia recomendada:**
1. `00_env_check.ipynb` - Verificación del entorno
2. `01_preprocessing_eda_v2_fixed.ipynb` - Preprocesamiento y EDA
3. `02_baseline_mlp_improved_fixed.ipynb` - Modelo MLP baseline
4. `03_kan_model_fixed.ipynb` - Implementación KAN
5. `04_eval_compare_fixed.ipynb` - Comparación de resultados
6. `05_kan_deep_analysis.ipynb` - Análisis profundo de KAN

### 2. Generación de Documentos

**PDF Académico:**
```bash
python create_pdf_simple.py
```

**HTML Web:**
```bash
python generate_document.py
```

**LaTeX:**
```bash
python compile_latex.py
```

---

## 🧠 Arquitecturas Implementadas

### KAN (Kolmogorov-Arnold Network)
- **Capa:** FastKANLayer con funciones spline lineales
- **Parámetros:** 13,889 (1,152 funciones spline individuales)
- **Knots:** 12 puntos uniformemente espaciados en [-2, 2]
- **Interpretabilidad:** Cada función φᵢ,ⱼ es visualizable

### MLP Mejorado
- **Arquitectura:** 36 → 128 → 64 → 1
- **Regularización:** Dropout (0.3, 0.2) + BatchNorm
- **Activación:** ReLU
- **Parámetros:** 13,441

---

## 📈 Resultados Experimentales

### Métricas de Rendimiento

| Modelo | Conjunto | MAE | RMSE | R² | MAPE |
|--------|----------|-----|------|----|----|
| **KAN** | **Test** | **1,508** | **3,164** | **0.979** | **20.3%** |
| MLP | Test | 2,334 | 4,641 | 0.955 | 29.8% |

### Análisis de Interpretabilidad

- **50%** de las funciones KAN son no-monótonas
- **20%** son altamente no-lineales
- Patrones económicos revelados:
  - Efectos no-monotónicos del precio del combustible
  - Dependencias estacionales complejas
  - Comportamientos de umbral en variables categóricas

---

## 🔬 Metodología

### Dataset
- **Fuente:** Walmart Sales Forecasting (Kaggle)
- **Registros:** 421,570 observaciones
- **Periodo:** Febrero 2010 - Julio 2012 (130 semanas)
- **Features:** 36 variables después de feature engineering
- **División:** 80.4% entrenamiento, 9.8% validación, 9.8% prueba

### Técnicas Utilizadas
- **Feature Engineering:** 35+ features temporales y económicas
- **Regularización:** Early stopping, learning rate scheduling
- **Validación:** División temporal estricta
- **Métricas:** MAE, RMSE, R², MAPE

---

## 📚 Referencias Principales

1. Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks." *arXiv:2404.19756v5*
2. Kolmogorov, A. N. (1957). "On the representation of continuous functions..."
3. Hornik, K., et al. (1989). "Multilayer feedforward networks are universal approximators."

**Ver documento completo para 24 referencias académicas completas.**

---

## 🤝 Contribuciones y Uso Académico

Este proyecto es de **uso académico**. Si utilizas este código o metodología, por favor cita apropiadamente:

```bibtex
@misc{morales2025kan,
  title={Comparación de Redes Kolmogorov-Arnold vs. Perceptrones Multicapa: Estudio Empírico en Predicción de Ventas},
  author={Edgar Alberto Morales Gutiérrez},
  year={2025},
  school={Universidad Panamericana}
}
```

---

## 📄 Archivos Principales

- **`PROYECTO_FINAL_KAN_MLP_SALES.pdf`** - Documento académico completo (1.09 MB)
- **`notebooks/05_kan_deep_analysis.ipynb`** - Análisis interpretabilidad (3.77 MB)
- **`models/simplified_kan_model.pt`** - Modelo KAN entrenado
- **`reports/figura_*`** - Visualizaciones de resultados

---

## 🐛 Troubleshooting

### Problemas Comunes

**Error CUDA:**
```bash
# Instalar versión CPU si no tienes GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Error de memoria:**
```python
# Reducir batch size en los notebooks
batch_size = 64  # en lugar de 256
```

**Problemas con PDF:**
```bash
pip install --upgrade reportlab
```

---

## 📧 Contacto

**Edgar Alberto Morales Gutiérrez**  
Universidad Panamericana - Aprendizaje Profundo  
Proyecto Final - Septiembre 2025

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella!**