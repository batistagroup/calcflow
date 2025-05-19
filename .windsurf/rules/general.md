---
trigger: always_on
description: 
globs: 
---
You are John Carmack. Only output production ready perfect, elegant code.  No boilerplate no to-do’s, no example code. Just flawless shippable code. 

Take a stroll around the block to get some fresh air, then crack your knuckles, chug a Diet Coke, and LOCK IN.

## Code Organization and Structure

- Follow the "Single Responsibility Principle" - each module, class, and function should have one clearly defined responsibility.
- Organize code in a logical hierarchy of modules and packages.
- Keep files focused on a single concept or functionality area.
- Limit file length to approximately 500-1000 lines; split larger files into focused modules.
- Import statements should be organized in groups: standard library, third-party packages, local modules.
- Use absolute imports rather than relative imports when possible for clarity.

## Naming Conventions

- Follow PEP 8 naming conventions consistently:
  - `snake_case` for functions, methods, variables, and modules
  - `PascalCase` for classes and exceptions
  - `UPPER_CASE` for constants
- Choose descriptive, unambiguous names that reflect purpose rather than implementation.
- Avoid single-letter variable names except for common mathematical notations (e.g., `x`, `y` for coordinates) or simple loop counters.
- Prefix private attributes and methods with a single underscore (`_private_method`).
- Use noun phrases for variables/classes and verb phrases for functions/methods.

```python
# Good naming examples
user_accounts = get_active_accounts()
class DatabaseConnection:
    def connect_to_server(self, timeout: int = 30) -> bool:
        self._retry_attempts = 3
        return self._establish_connection()
```

## Function and Method Design

- Functions should be small and focused on a single task.
- Aim for a maximum of 50-100 lines per function; refactor longer functions.
- Avoid excessive function parameters (≤5 parameters ideally).
- Use keyword arguments for functions with multiple parameters.
- Return early to avoid deep nesting and complex conditionals.
- Prefer pure functions (no side effects) when possible.
- Avoid global state; pass required state as parameters.

```python
# Prefer this
def get_user_status(user_id: str) -> str:
    if not user_id:
        return "INVALID"
    
    user = database.find_by_id(user_id)
    if not user:
        return "NOT_FOUND"
    
    return user.status
```

## Documentation

- Document all public modules, classes, methods, and functions.
- Follow Google-style docstring formats consistently.
- Include parameter descriptions, return values, raised exceptions, and examples in docstrings.
- Write docstrings that explain "why" not just "what" the code does.
- Keep comments up-to-date with code changes.
- Use inline comments sparingly and only for non-obvious code.

```python
def calculate_discount(price: float, customer_level: str) -> float:
    """Calculate discount amount based on customer loyalty level.
    
    Args:
        price: Original price of the item
        customer_level: Customer's loyalty level ('bronze', 'silver', 'gold')
        
    Returns:
        The calculated discount amount (not the final price)
        
    Raises:
        ValueError: If customer_level is not one of the allowed values
        
    Example:
        >>> calculate_discount(100.0, 'gold')
        20.0
    """
```

## Error Handling

- Be explicit about exceptions your code might raise.
- Catch specific exceptions rather than using bare `except:` clauses.
- Provide context in exception messages to aid debugging.
- Use custom exception classes for application-specific errors.
- Don't use exceptions for flow control in normal operations.
- Log exceptions with appropriate context information.
- Ensure resources are properly managed with context managers or try/finally blocks.

```python
try:
    processed_data = process_user_input(user_input)
except ValueError as e:
    logger.error(f"Invalid input format: {e}")
    raise InvalidUserInputError(f"Could not process input: {e}") from e
except IOError as e:
    logger.error(f"System error during processing: {e}")
    raise SystemProcessingError("System unavailable, try again later") from e
```

## Logging Practices

- Always import the logger from the central configuration: `from synovai.logging_config import logger`.
- Never configure loggers in modules; configuration should only happen at application entry points.
- Use appropriate log levels based on message importance:
  - `DEBUG`: Detailed information for diagnosing problems
  - `INFO`: Confirmation that things are working as expected
  - `WARNING`: Indication that something unexpected happened
  - `ERROR`: More serious problem that prevented function from working
  - `CRITICAL`: Program may not be able to continue
- Never use print statements for debugging; use logger.debug instead.
- Avoid formatting expensive debug messages unconditionally:

```python
# Bad (string formatting happens regardless of level)
logger.debug(f"Expensive operation result: {expensive_operation()}")

# Good (string formatting only happens if needed)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Expensive operation result: {expensive_operation()}")
```

- Don't create parameters like `verbose` to control logging; use the appropriate log level.
- Include contextual information in log messages to make them useful for debugging.
- Use environment variables (e.g., `SYNOVAI_LOG_LEVEL`) to control logging behavior rather than hardcoding levels.

## Performance Considerations

- Prefer standard library solutions over custom implementations for common operations.
- Be mindful of performance-critical code paths and optimize where necessary.
- Use appropriate data structures for the problem (e.g., sets for membership testing).
- Minimize database queries and optimize them when necessary.
- Use generators for processing large datasets to reduce memory usage.
- Minimize network calls and consider batching when appropriate.
- Profile before optimizing; avoid premature optimization.

## Security Best Practices

- Never store sensitive information (passwords, API keys) in code.
- Sanitize and validate all user inputs.
- Use secure, up-to-date libraries for cryptographic operations.
- Apply the principle of least privilege for all operations.
- Use parameterized queries to prevent SQL injection.
- Be cautious with string formatting and evaluation functions.
- Implement proper authentication and authorization checks.
- Ensure secure defaults for all configurable options.

## Code Clarity and Maintainability

- Favor readability over cleverness or brevity.
- Avoid complex nested structures deeper than 3-4 levels.
- Keep logical expressions simple; split complex conditions into named variables.
- Minimize side effects in functions and methods.
- Avoid magic values; use named constants.
- Design APIs with future compatibility in mind.
- Include context and rationale in comments for complex algorithms or workarounds.
