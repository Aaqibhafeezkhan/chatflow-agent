## Description / Descripción

<!--
Please include a summary of the changes and the related issue.
Por favor incluye un resumen de los cambios y el issue relacionado.
-->

Closes / Fixes: #

---

## Type of Change / Tipo de Cambio

- [ ] Bug fix / Corrección de error (non-breaking change fixing an issue)
- [ ] New feature / Nueva funcionalidad (non-breaking change adding capability)
- [ ] Breaking change (fix or feature changing existing behavior/API)
- [ ] Documentation / Documentación (updates, examples, guides)
- [ ] Tests / Pruebas (adding or improving test coverage)
- [ ] Refactoring or Chore (internal improvements, CI, dependencies)

---

## Motivation & Context / Motivación y Contexto

<!--
Why is this change required? What problem does it solve?
¿Por qué se requiere este cambio? ¿Qué problema resuelve?
-->

---

## Affected Components / Componentes Afectados

- [ ] `chatflow_agent.core.agent` (Agent, handoffs)
- [ ] `chatflow_agent.core.runner` (Runner loop, tool execution)
- [ ] `chatflow_agent.core.engines` (Gemini, OpenAI, Anthropic, Ollama)
- [ ] `chatflow_agent.core.memory` (SessionStore, SQLiteSessionStore, pruning)
- [ ] `chatflow_agent.core.tools` (Tool decorator, introspection)
- [ ] `chatflow_agent.channels` (WhatsApp, Telegram, CLI, base)
- [ ] Documentation or Examples / Documentación o Ejemplos

---

## Testing Performed / Pruebas Realizadas

<!--
Describe the tests you ran to verify your changes. Include commands used.
Describe las pruebas que ejecutaste para verificar tus cambios.
-->

- [ ] Added new unit or integration test(s) in `tests/`
- [ ] Ran full test suite locally: `pytest -v`

---

## Checklist / Lista de Verificación

Please verify before requesting a review:
*Por favor verifica antes de solicitar una revisión:*

- [ ] My code follows the project's code style and conventions (`target-version = "py310"`).
- [ ] I ran `ruff check .` with zero errors or warnings.
- [ ] I ran `mypy` with zero type check errors (`strict = true`).
- [ ] All existing and new tests pass with `pytest`.
- [ ] I did **NOT** commit any private secrets, API keys, tokens, or personal phone numbers.
- [ ] I did not introduce unnecessary third-party dependencies into the core package.
- [ ] Documentation and/or docstrings have been updated if applicable.
- [ ] Backward compatibility of public APIs is preserved (or explicitly discussed in the issue).
