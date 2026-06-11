<div align="center">

# 🔮 PathForge

### *Career Universe Explorer — AI-Powered Professional Trajectory Optimization*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black?logo=threedotjs)](https://threejs.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**PathForge** is an intelligent career trajectory exploration engine that combines **Beam Search multiobjetivo**, **dominancia de Pareto (NSGA-II)**, **restricciones de satisfacción**, y **Modelos de Lenguaje (LLM)** para generar y evaluar trayectorias profesionales alternativas bajo distintos objetivos y restricciones del mundo real.

</div>

---

## 📋 Tabla de Contenidos

- [Descripción del Problema](#-descripción-del-problema)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Configuración del LLM](#-configuración-del-llm)
- [Instrucciones de Ejecución](#-instrucciones-de-ejecución)
- [Dataset](#-dataset)
- [Rol del LLM](#-rol-del-llm)
- [Diseño Experimental](#-diseño-experimental)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tests](#-tests)

---

## 🎯 Descripción del Problema

**Tema 7 — Exploración de trayectorias profesionales alternativas**

Dado un conjunto de decisiones posibles en una carrera profesional, PathForge genera y evalúa múltiples trayectorias válidas bajo distintos criterios simultáneos:

| Objetivo | Tipo | Descripción |
|---|---|---|
| `salary_growth` | Maximizar | Crecimiento salarial total de la trayectoria |
| `final_salary` | Maximizar | Salario del rol final alcanzado |
| `avg_demand` | Maximizar | Demanda laboral promedio de los roles |
| `avg_satisfaction` | Maximizar | Satisfacción profesional media |
| `total_years` | Minimizar | Años totales de la trayectoria |
| `avg_risk` | Minimizar | Riesgo promedio de las transiciones |
| `avg_difficulty` | Minimizar | Dificultad media de los cambios de rol |

El sistema utiliza **dominancia de Pareto (NSGA-II vectorizado)** para encontrar trayectorias no dominadas — aquellas donde no existe ninguna alternativa mejor en todos los objetivos simultáneamente — y un **LLM** para análisis cualitativo y ranking por criterio en lenguaje natural.

---

## 🏛️ Arquitectura del Sistema

```
PathForge/
│
├── backend/
│   ├── core/               ← Núcleo algorítmico
│   │   ├── graph.py        # CareerGraph: abstracción del grafo + scoring multiobjetivo
│   │   ├── generator.py    # Beam Search multiobjetivo con poda por restricciones
│   │   ├── evaluator.py    # NSGA-II vectorizado con numpy + crowding distance
│   │   ├── constraints.py  # Sistema de restricciones componible (AND/OR)
│   │   ├── scorer.py       # GradientBoosting ML para scoring de transiciones
│   │   └── simulation.py   # Monte Carlo con eventos estocásticos (Welford online)
│   │
│   ├── data/               ← Gestión de datos
│   │   ├── loader.py       # Carga y validación del grafo (Pydantic v2)
│   │   ├── careers.json    # Grafo default (12 roles tech, 26 transiciones)
│   │   ├── input_manager.py# CRUD de configuraciones usuario (SQLite)
│   │   ├── download_data.py# Descarga dataset Karrierewege_plus de HuggingFace
│   │   ├── transform_data.py# Transforma CSV → domain graphs por sector ISCO-08
│   │   └── problems/       # Domain graphs generados (banking, hospitals, etc.)
│   │
│   ├── llm/                ← Integración LLM
│   │   ├── client.py       # Cliente multi-proveedor (7 LLMs) con cola circular
│   │   ├── analyzer.py     # Análisis cualitativo sync + async
│   │   └── prompts.py      # Prompts en inglés, respuesta en español
│   │
│   ├── experiments/        ← Diseño experimental
│   │   ├── metrics.py      # Métricas del algoritmo (diversidad vectorizada)
│   │   ├── runner.py       # 64 experimentos: 4 configs × 4 perfiles × 4 fuentes
│   │   ├── run_experiments.py # Experimento por instancias con timeout LLM
│   │   └── visualizer.py   # 6 gráficas matplotlib del análisis experimental
│   │
│   ├── tests/              ← Suite de tests
│   │   ├── test_graph.py   # 25 tests del grafo y scoring
│   │   ├── test_generator.py # 22 tests del Beam Search y Pareto
│   │   └── test_llm.py     # 24 tests del cliente LLM y prompts
│   │
│   └── main_api.py         ← FastAPI + WebSocket (lifespan, async endpoints)
│
└── frontend/               ← Interfaz visual
    ├── index.html          # SPA con 3 vistas: Setup / Universe 3D / Analysis
    ├── css/style.css       # Tema espacial cyberpunk (1400 líneas)
    └── js/
        ├── universe.js     # Fondo espacial 2D animado (SpaceBG)
        ├── graph3d.js      # Universo 3D interactivo (Three.js, U3D)
        ├── animation.js    # Animaciones del beam search en tiempo real
        ├── websocket.js    # API REST + WebSocket con heartbeat (API module)
        └── ui.js           # Estado global + gestión de vistas (AppState)
```

### Flujo de datos

```
Usuario
  │
  ▼
Frontend (Setup)
  → Configura nodos, aristas, perfil de restricciones, beam_width
  │
  ▼
WebSocket /ws/explore
  → TrajectoryGenerator.generate()
      ├── Beam Search (expand → filter → Pareto select)
      ├── Emite steps en tiempo real → animación 3D Universe view
      └── Retorna lista de EvaluatedTrajectory ordenada por Pareto rank
  │
  ▼
Frontend (Analysis)
  → Muestra trayectorias con métricas
  → POST /api/analyze → TrajectoryAnalyzer.rank_by_async()
      └── LLM analiza y rankea por criterio en lenguaje natural
```

---

## 🚀 Instalación

### Requisitos

- Python 3.10 o superior
- pip

### Paso 1 — Clonar e instalar dependencias

```bash
git clone https://github.com/JosuSC/PathForge.git
cd PathForge
pip install -r requirements.txt
```

### Paso 2 — Verificar la instalación

```bash
python verify_installation.py
```

---

## 🔑 Configuración del LLM

PathForge soporta **7 proveedores LLM** con cola circular automática. Si una key se agota, rota a la siguiente sin interrumpir la sesión.

### Crear el archivo `.env`

```bash
cp .env.example .env
```

### Editar `.env` con tus API keys

```env
# Formato: LLM_KEY_N=proveedor:api_key
# Puedes combinar múltiples proveedores

LLM_KEY_1=gemini:AIzaSy...
LLM_KEY_2=claude:sk-ant-...
LLM_KEY_3=openai:sk-...
LLM_KEY_4=groq:gsk_...
LLM_KEY_5=deepseek:sk-...
LLM_KEY_6=mistral:...
LLM_KEY_7=openrouter:sk-or-...

# Opcional: personalizar modelo por proveedor
GEMINI_MODEL=gemini-1.5-flash
CLAUDE_MODEL=claude-haiku-4-5-20251001
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant
DEEPSEEK_MODEL=deepseek-chat
MISTRAL_MODEL=mistral-small-latest
```

> **Nota:** Con una sola key funciona correctamente. Más keys = mayor disponibilidad y menor riesgo de rate-limit.

### Obtener API keys gratuitas

| Proveedor | URL | Plan gratuito |
|---|---|---|
| Gemini | [aistudio.google.com](https://aistudio.google.com) | ✅ 1M tokens/min |
| Groq | [console.groq.com](https://console.groq.com) | ✅ 30 req/min |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | ✅ Créditos iniciales |
| Claude | [console.anthropic.com](https://console.anthropic.com) | 💳 Pago por uso |
| OpenAI | [platform.openai.com](https://platform.openai.com) | 💳 Pago por uso |

---

## ▶️ Instrucciones de Ejecución

### Opción A — Ejecución estándar (recomendada)

```bash
# Desde la raíz del proyecto
python -m backend.main_api
```

Abre en el navegador: **http://localhost:8000**

### Opción B — Con recarga automática (desarrollo)

```bash
uvicorn backend.main_api:app --reload --port 8000
```

### Opción C — Solo el backend (para usar con cliente propio)

```bash
python -m backend.main_api
# API disponible en http://localhost:8000
# Documentación Swagger: http://localhost:8000/docs
```

### Verificar que el sistema está funcionando

```bash
# Estado del LLM
curl http://localhost:8000/api/llm/status

# Grafo default
curl http://localhost:8000/api/graph

# Dominios disponibles (tras ejecutar transform_data.py)
curl http://localhost:8000/api/domains
```

---

## 📊 Dataset

PathForge utiliza dos fuentes de datos:

### 1. Grafo Default (`backend/data/careers.json`)

Grafo curado manualmente de **12 roles del sector tecnológico** con **26 transiciones reales**:

| Campo | Descripción |
|---|---|
| `avg_salary` | Salario medio en USD/año (25k–180k) |
| `demand` | Demanda laboral normalizada [0–1] |
| `satisfaction` | Satisfacción profesional [0–1] |
| `type` | Nivel del rol: `entry/mid/senior/leadership` |
| `transition_probability` | Probabilidad observada de esa transición |
| `salary_growth` | Crecimiento salarial típico en esa transición |

### 2. Domain Graphs (generados desde datos reales)

Derivados del dataset **Karrierewege_plus** (HuggingFace: `ElenaSenger/Karrierewege_plus`):

- **80,000 trayectorias reales** de profesionales flamencos (VDAB, Bélgica)
- **1,162 ocupaciones ESCO** clasificadas con taxonomía **ISCO-08**
- Genera automáticamente grafos por sector: `banking`, `hospitals`, `software_development`, `engineering`, `medical_doctors`, y más

#### Generar los domain graphs

```bash
# 1. Descargar los datos crudos (primera vez, ~5 min)
python backend/data/download_data.py

# 2. Transformar a domain graphs (1-3 min)
python backend/data/transform_data.py

# 3. Verificar los dominios generados
curl http://localhost:8000/api/domains
```

> Los archivos generados se guardan en `backend/data/problems/` como `graph.json`, `metadata.json` e `instances.json` por dominio.

---

## 🤖 Rol del LLM

El LLM es un **componente funcional del sistema**, no un adorno. Tiene tres roles distintos:

### 1. Comparación de trayectorias (`/api/analyze`)

El LLM recibe las top-5 trayectorias del frente de Pareto con sus métricas numéricas y produce un análisis estructurado en 6 puntos:

- Tipo de éxito que representa cada destino final
- Perfil de riesgo relativo
- Relevancia en el mercado 2025-2026
- Satisfacción a largo plazo
- Recomendación fundamentada
- Riesgos ocultos

### 2. Ranking por criterio natural (`rank_by`)

El usuario expresa su objetivo en lenguaje libre ("quiero maximizar mi salario en 5 años con bajo riesgo") y el LLM rankea las trayectorias disponibles hacia el destino final que mejor encaja con ese objetivo.

### 3. Análisis de destinos alternativos (`compare_terminals`)

Dado que el sistema puede encontrar trayectorias hacia múltiples destinos finales (CTO, Founder, ML Engineer, etc.), el LLM compara el mejor camino hacia cada uno y recomienda cuál perseguir según el perfil del usuario.

### Arquitectura del cliente LLM

```python
# Ejemplo: 3 keys con fallback automático
LLM_KEY_1=gemini:AIza...   # activa por defecto
LLM_KEY_2=groq:gsk_...     # si gemini llega al rate-limit
LLM_KEY_3=deepseek:sk-...  # backup final
```

- **Cola circular**: rota automáticamente al llegar al límite de cuota
- **Backoff exponencial**: 60s → 120s → 240s → máx 600s por key fallida
- **Thread-safe**: múltiples requests concurrentes sin condiciones de carrera
- **Timeout LLM**: 30s por llamada en los experimentos (configurable)

---

## 🧪 Diseño Experimental

### Ejecutar el experimento completo

```bash
# 64 combinaciones: 4 configs × 4 perfiles × 4 nodos fuente
python backend/experiments/runner.py

# Con un domain graph real
python backend/experiments/runner.py --domain software_development

# Experimento por instancias de test_instances.json
python backend/experiments/run_experiments.py

# Generar gráficas de análisis
python backend/experiments/visualizer.py
# → backend/experiments/results/plots/experiment_summary.png
```

### Configuraciones comparadas

| Config | beam_width | max_depth | Característica |
|---|---|---|---|
| `narrow_shallow` | 4 | 3 | Rápido, pocas rutas |
| `narrow_deep` | 4 | 6 | Profundo pero selectivo |
| `wide_shallow` | 12 | 3 | Amplio pero corto |
| `wide_deep` | 12 | 6 | Máxima exploración |

### Métricas medidas

- **Diversidad** — distancia euclidiana media entre pares en espacio de scores normalizados
- **Frente de Pareto** — trayectorias no dominadas encontradas
- **Tasa de terminales** — fracción que alcanza un nodo sin sucesores
- **Probabilidad de transición** — promedio de probabilidades observadas en el dataset real
- **Tiempo de ejecución** — ms por configuración

### Gráficas generadas (6 paneles)

1. Diversidad promedio por configuración del generador
2. Tamaño del frente de Pareto por perfil de restricciones
3. Trade-off crecimiento salarial vs riesgo
4. Tiempo de ejecución por configuración
5. Tasa de terminales alcanzados por configuración
6. Probabilidad de transición media por perfil

---

## 📁 Estructura del Proyecto

```
PathForge/
│
├── .env.example                    ← Plantilla de configuración LLM
├── requirements.txt                ← Dependencias Python
├── verify_installation.py          ← Script de verificación del entorno
├── README.md
│
├── backend/
│   ├── main_api.py                 ← FastAPI (lifespan, REST + WebSocket)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py                ← CareerGraph + Trajectory + scoring
│   │   ├── generator.py            ← Beam Search + selección Pareto
│   │   ├── evaluator.py            ← NSGA-II vectorizado numpy
│   │   ├── constraints.py          ← Sistema de restricciones composable
│   │   ├── scorer.py               ← GradientBoostingClassifier (sklearn)
│   │   └── simulation.py           ← Monte Carlo + Welford online stats
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               ← Pydantic v2 schema + validación
│   │   ├── careers.json            ← Grafo default 12 nodos (tech sector)
│   │   ├── test_instances.json     ← 12 instancias de prueba
│   │   ├── input_manager.py        ← SQLite CRUD de configuraciones
│   │   ├── download_data.py        ← HuggingFace dataset downloader
│   │   ├── transform_data.py       ← CSV → domain graphs (ISCO-08)
│   │   └── problems/               ← Domain graphs generados
│   │       └── {domain}/
│   │           ├── graph.json
│   │           ├── metadata.json
│   │           └── instances.json
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py               ← Multi-provider client (7 LLMs)
│   │   ├── analyzer.py             ← Sync + async analysis methods
│   │   └── prompts.py              ← EN prompts → ES responses
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── metrics.py              ← ExperimentMetrics + diversity
│   │   ├── runner.py               ← 64-experiment grid search
│   │   ├── run_experiments.py      ← Instance-based experiment runner
│   │   ├── visualizer.py           ← 6-panel matplotlib report
│   │   └── results/                ← Generado automáticamente
│   │       ├── experiment_results.json
│   │       ├── results.csv
│   │       └── plots/
│   │           └── experiment_summary.png
│   └── tests/
│       ├── __init__.py
│       ├── test_graph.py           ← 25 tests (CareerGraph, Trajectory, scoring)
│       ├── test_generator.py       ← 22 tests (Beam Search, Pareto, constraints)
│       └── test_llm.py             ← 24 tests (client, analyzer, prompts)
│
└── frontend/
    ├── index.html                  ← SPA: Setup / Universe 3D / Analysis
    ├── css/
    │   └── style.css               ← Cyberpunk space theme
    └── js/
        ├── universe.js             ← SpaceBG: nebulas, stars, shooting stars
        ├── graph3d.js              ← U3D: Three.js interactive 3D graph
        ├── animation.js            ← Beam search step-by-step animation
        ├── websocket.js            ← API module: REST + WebSocket + heartbeat
        ├── ui.js                   ← AppState, views, LLM analysis
        └── three.min.js            ← Three.js r128
```

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest backend/tests/ -v

# Un módulo específico
pytest backend/tests/test_graph.py -v
pytest backend/tests/test_generator.py -v
pytest backend/tests/test_llm.py -v

# Con reporte de cobertura
pytest backend/tests/ --cov=backend --cov-report=term-missing

# Solo tests rápidos (sin LLM)
pytest backend/tests/ -v -k "not llm_call"
```

### Resumen de la suite (71 tests)

| Archivo | Tests | Cubre |
|---|---|---|
| `test_graph.py` | 25 | Carga del grafo, scoring multiobjetivo, campos v2 (is_terminal_end, transition_probability_score), iter_paths iterativo, validación, Trajectory dataclass |
| `test_generator.py` | 22 | Generación básica, perfiles de restricciones, deduplicación [GEN2], max_salary dinámico [GEN3], Pareto quality, step_callback, configuraciones extremas |
| `test_llm.py` | 24 | Cliente sin keys [CL6], todos los proveedores registrados [CL1], empty list guard [AN2], prompt no expuesto en API [AN3], métodos async [AN1], ml_success_prob eliminado [PR1], sim_section condicional [PR2] |

---

## 🔧 API Reference

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/graph` | Grafo default (careers.json) |
| `GET` | `/api/domains` | Listado de domain graphs disponibles |
| `GET` | `/api/domains/{id}/graph` | Grafo de un dominio específico |
| `GET` | `/api/terminals` | Nodos terminales del grafo default |
| `GET` | `/api/model/info` | Info del modelo ML (CV AUC, feature importances) |
| `GET` | `/api/llm/status` | Estado de las API keys LLM |
| `POST` | `/api/generate` | Generar trayectorias (REST, sin streaming) |
| `POST` | `/api/analyze` | Análisis LLM de trayectorias |
| `POST` | `/api/simulate` | Simulación Monte Carlo de una trayectoria |
| `POST` | `/api/inputs/create` | Guardar configuración de usuario (SQLite) |
| `GET` | `/api/inputs/list` | Listar configuraciones guardadas |
| `GET` | `/api/inputs/{id}` | Obtener configuración por ID |
| `DELETE` | `/api/inputs/{id}` | Eliminar configuración |
| `WS` | `/ws/explore` | Beam Search con streaming en tiempo real |

---

## 📦 Dependencias principales

```
fastapi>=0.104          # API framework con lifespan
uvicorn                 # ASGI server
networkx                # Grafo dirigido
numpy                   # Vectorización (NSGA-II, diversidad)
scikit-learn            # GradientBoostingClassifier
pydantic>=2.0           # Validación de datos
loguru                  # Logging estructurado
python-dotenv           # Carga de .env
anthropic               # Claude API
google-genai            # Gemini API (nuevo SDK)
openai                  # OpenAI, Groq, DeepSeek, Mistral (OpenAI-compatible)
matplotlib              # Gráficas del experimento
rich                    # Progreso en terminal (opcional)
pytest                  # Suite de tests
```
