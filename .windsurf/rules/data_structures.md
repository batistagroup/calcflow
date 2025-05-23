---
trigger: model_decision
description: Creating a new data structure
globs: 
---
## General Guidelines

- Choose the simplest data structure that meets all requirements.
- Prioritize immutability where possible for safer and more predictable code.
- Document complex data structures and their usage patterns.
- Consider memory usage and performance characteristics for your specific use case.

## Functional Programming Approaches

- Use functional programming when:
  - Working with transformation pipelines on collections of data.
  - Operations are stateless and side-effect free.
  - Composing multiple operations where each takes input and produces output.
  - Implementing algorithms that are naturally recursive.

```python
# Prefer this functional approach for data transformations
def process_data(items: list[int]) -> list[int]:
    return [transform(item) for item in items if filter_condition(item)]

# For complex transformations, compose smaller functions
def process_complex_data(items: list[dict[str, any]]) -> list[dict[str, any]]:
    return list(map(enrich_data, 
                   filter(is_valid_data, 
                          map(normalize_data, items))))
```

## Regular Classes

- Use regular classes when:
  - You need to combine data with behavior that operates on that data.
  - The entity has identity beyond its attribute values.
  - You need to maintain internal state that changes over time.
  - Implementing design patterns that rely on object-oriented principles.
  - Building complex systems with inheritance hierarchies.

```python
class DataProcessor:
    def __init__(self, config: dict[str, any]) -> None:
        self.config = config
        self._cache: dict[str, any] = {}
        
    def process(self, data: list[any]) -> list[any]:
        # Implementation with state management
        result = self._apply_transformations(data)
        self._cache[hash(str(data))] = result
        return result
```

## TypedDict

- Use TypedDict when:
  - Working with dictionary data that has a specific, known schema.
  - Interacting with external APIs or file formats that use dictionary-like structures.
  - You need type checking for dictionary keys and values but don't need methods.
  - You want to document the structure of dictionary data.

```python
from typing import TypedDict

class UserData(TypedDict):
    id: str
    name: str
    email: str
    age: int
    active: bool

def process_user(user: UserData) -> None:
    # Now IDE and type checkers know the structure of user
    print(f"Processing {user['name']}, {user['age']} years old")
```

## Dataclasses

- Use dataclasses when:
  - You need a simple container for data with little or no custom behavior.
  - You want automatic generation of common magic methods (__init__, __repr__, etc.).
  - You need immutable data structures (with frozen=True).
  - You need to compare instances based on their field values, not identity.
  - Working with configuration objects or value objects in domain-driven design.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Configuration:
    app_name: str
    version: str
    debug: bool = False
    timeout: int = 30
    tags: list[str] = field(default_factory=list)
```

## Named Tuples and NamedTuple

- Use namedtuple or NamedTuple when:
  - You need an immutable record-like structure.
  - You want to access fields by name rather than position.
  - Memory efficiency is a concern (namedtuples have a smaller memory footprint than classes).
  - You need to unpack the structure or use it in existing tuple-based APIs.

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
```

## Enumerations

- Use Enum or Flag when:
  - Representing a fixed set of related constants.
  - You need to constrain values to a predefined set.
  - You want to add semantic meaning to constants.
  - The values need string representations for debugging or serialization.

```python
from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    
    @property
    def is_primary(self) -> bool:
        return self in {Color.RED, Color.GREEN, Color.BLUE}
```

## Decision Matrix

| Data Structure | Immutability | Behavior | Type Safety | Memory Efficiency | Use Case |
|----------------|--------------|----------|------------|-------------------|----------|
| Functions only | High | Stateless | Medium | High | Data transformations |
| Regular Class | Low | Rich | High | Low | Complex objects with behavior |
| TypedDict | Medium | None | High | Medium | Dictionary with schema |
| Dataclass | Optional | Limited | High | Medium | Data containers |
| NamedTuple | High | Limited | High | High | Immutable records |
| Enum | High | Limited | High | High | Fixed set of constants |

## Integration with Type Annotations

- Ensure all data structures have proper type annotations.
- Use generic parameters to define container content types.
- Consider using Protocols for defining structural interfaces.
- Use Union types when a function can work with multiple data structure types.

## When to Refactor

- Convert a dictionary to a TypedDict when its schema becomes clear and stable.
- Move from a TypedDict to a dataclass when you need to add behavior or validation.
- Convert a dataclass to a regular class when behavior becomes more complex than data.
- Refactor function-based code to objects when state management becomes necessary.
