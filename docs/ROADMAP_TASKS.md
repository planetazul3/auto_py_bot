# Roadmap de tareas del proyecto

## 1) Actualización de código

- [ ] Migrar configuración a variables de entorno + validación fuerte (Pydantic o dataclasses).
- [ ] Separar capa de estrategias, capa ML y capa de ejecución de órdenes en servicios independientes.
- [ ] Implementar manejo consistente de errores de exchange (retries exponenciales + circuit breaker).
- [ ] Añadir tipado estático en todos los módulos críticos y ejecutar `mypy` en CI.

## 2) Documentación

- [ ] Escribir `docs/OPERATIONS.md` con runbook de incidentes (caídas API, datos corruptos, latencia).
- [ ] Añadir `docs/CONFIG_REFERENCE.md` detallando cada parámetro de `config.json`.
- [ ] Publicar arquitectura lógica con diagramas de componentes y flujo de datos.
- [ ] Generar changelog semántico de versiones.

## 3) Pruebas

- [ ] Crear tests unitarios para `risk_manager.py` y `sol_strategies.py`.
- [ ] Incorporar tests de integración con datos históricos mockeados.
- [ ] Añadir prueba de regresión para señales ML (dataset fijo + salida esperada).
- [ ] Medir cobertura y exigir umbral mínimo del 70% en CI.

## 4) Orquestación de despliegue

- [ ] Parametrizar `docker-compose.yml` con perfiles (`paper`, `live`).
- [ ] Añadir healthcheck activo y auto-restart.
- [ ] Integrar envío de logs a stack centralizado (Loki/ELK).
- [ ] Automatizar despliegue con pipeline CD (tag -> build -> release).

## 5) Gobierno operativo

- [ ] Definir política de gestión de secretos (Vault/GitHub Secrets).
- [ ] Establecer alertas por drawdown, error rate y desconexión de exchange.
- [ ] Crear checklist de preproducción y postmortem template.
