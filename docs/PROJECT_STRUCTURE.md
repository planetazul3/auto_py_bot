# Estructura profesional del proyecto

## Árbol principal

```text
auto_py_bot/
├── src/auto_py_bot/        # Código fuente del bot y módulos de negocio
├── scripts/                # Entry points ejecutables
├── docs/                   # Documentación técnica y de operación
├── tests/                  # Pruebas automatizadas
├── deploy/                 # Orquestación de despliegue (Docker/systemd)
├── .github/workflows/      # CI
├── config.json             # Configuración runtime
├── requirements.txt        # Dependencias Python
└── Makefile                # Tareas estándar
```

## Convenciones

- `src/auto_py_bot/`: lógica reusable y módulos importables.
- `scripts/`: wrappers mínimos para ejecutar cada flujo (`train`, `run`, `backtest`, etc.).
- `docs/`: guías y roadmap de evolución.
- `deploy/`: archivos para operar el bot en entorno productivo.
- `tests/`: pruebas rápidas de estructura e importación.
