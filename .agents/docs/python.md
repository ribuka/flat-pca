# Python coding

## Rules

- Make changes only where necessary, minimize impact.
- Add docstrings for all functions and classes with NumPy style.
- Add type hints for all function parameters and return types.

### Architecture and maintainability

- Keep one primary responsibility per Python module.
- NEVER append a distinct responsibility to an existing module; create a focused module or package instead.
- Split independently testable stages such as input, validation, preprocessing, transformation, and orchestration into cohesive modules when appropriate.
- Keep public API modules and `__init__.py` files thin and explicitly re-export public symbols. When refactoring changes an import path, update every reference instead of keeping a re-export for the old path.
- NEVER introduce circular imports or generic catch-all modules such as `utils.py`.
- Organize tests by responsibility and share setup through narrowly scoped pytest fixtures. NEVER split modules based on line count alone.
