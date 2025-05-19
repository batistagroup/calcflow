---
trigger: always_on
description: 
globs: 
---
## General Guidelines

- All functions and methods must have type annotations for parameters and return values.
- Use Python 3.11+ typing features where applicable.
- All code must pass mypy validation with strict mode enabled.
- Type annotations should be as precise as possible while remaining maintainable.

## Type Annotations

- Use built-in collection types with their generic parameters: `list[str]`, `dict[str, int]`, etc.
- Use `None` as a return type for functions that don't return a value.
- Use `|` for union types (Python 3.10+): `str | None` instead of `Optional[str]`.
- Use `Literal` types for constrained string or numeric values.
- Use type aliases for complex types that are reused throughout the codebase.

## Function Annotations

- Annotate all function parameters and return types.
- Use `-> None` for functions that don't return a value.
- For async functions, use `-> Awaitable[T]` or more specifically `-> AsyncGenerator[T, None]` if applicable.

```python
def process_data(data: list[str]) -> dict[str, int]:
    # Implementation...
    
async def fetch_data(url: str) -> list[dict[str, any]]:
    # Implementation...
```

## Variable Annotations

- Annotate class attributes in `__init__` or as class variables.
- Annotate complex local variables when their type is not obvious.
- Use Final for constants and values that should not be reassigned.

```python
class User:
    name: str
    age: int
    settings: dict[str, bool] = {"notifications": True}
    
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
```

## Type Imports

- Import from `typing` module only what is needed.
- Python 3.11+ built-in types don't need to be imported from typing.
- For special types, import explicitly: `from typing import TypeVar, Generic, Protocol`.

## Generic Types

- Use TypeVar for generic functions and classes.
- Define proper bounds for TypeVars when appropriate.
- Use meaningful names for type variables (T, K, V for simple cases, descriptive names for complex cases).

```python
from typing import TypeVar, Generic

T = TypeVar('T')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

class Cache(Generic[K, V]):
    # Implementation...
```

## Type Aliases

- Create type aliases for complex or frequently used types.
- Place type aliases in a dedicated `types.py` file or at the top of the module.

```python
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
UserData: TypeAlias = dict[str, str | int | list[str]]
```

## Optional and Union Types

- Use `|` syntax for union types: `str | None` instead of `Optional[str]`.
- Be explicit about None: use `str | None` instead of just `Optional`.
- For more complex unions, consider creating a type alias.

## Special Types

- Use `Any` sparingly and only when absolutely necessary.
- Prefer `object` over `Any` when you just need a common base class.
- Use `Protocol` for structural typing instead of requiring inheritance.
- Use `TypedDict` for dictionaries with specific key/value requirements.

```python
from typing import Protocol, TypedDict

class Renderer(Protocol):
    def render(self, data: dict[str, any]) -> str:
        ...

class UserDict(TypedDict):
    name: str
    age: int
    email: str | None
```

## Collections

- Annotate all collections with their element types.
- Use specific collection types from `collections.abc` for clearer intentions: `Mapping` instead of `dict` when appropriate.
- For mutable/immutable distinctions, use the appropriate type: `Sequence` vs `list`.

## Documentation

- Document complex type usage with docstrings.
- Explain the purpose of type aliases, especially if they represent domain concepts.
- Use type documentation in docstrings that follows the same conventions as code.

## Mypy Configuration

- Run mypy with `--strict` mode.
- Document exceptions to strict typing with `# type: ignore` comments that include explanations.
- Use inline `# type: ignore[error-code]` rather than blanket ignores.

## Error Handling

- Define and use proper Exception subclasses with type annotations.
- Annotate exception handling appropriately in try/except blocks.

## Testing

- Ensure test functions also have proper type annotations.
- Use appropriate types for mocks and test fixtures. 