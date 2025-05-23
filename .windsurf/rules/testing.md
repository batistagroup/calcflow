---
trigger: model_decision
description: Writing tests for our code
globs: 
---
# Python Testing Standards

This document outlines the standards and best practices for writing effective and meaningful tests for Python codebases. The goal is not merely to achieve high coverage percentages but to build a robust test suite that provides genuine confidence in the correctness and reliability of the software. We aim for tests that are clear, focused, maintainable, and truly validate the intended behavior.

Inspired by principles of rigorous engineering, these guidelines draw upon the philosophies outlined in `general.mdc`, `typing.mdc`, and `data_structures.mdc`.

## Core Philosophy: Meaningful Validation

- **Test Behavior, Not Implementation:** Tests should verify *what* the code does (its public contract and observable behavior), not *how* it does it internally. Avoid tests that are tightly coupled to implementation details, as they become brittle and hinder refactoring.
- **Maximize Confidence:** Each test should significantly increase confidence in the correctness of the tested code path. Avoid superficial tests that cover lines but don't validate meaningful logic or edge cases.
- **Clarity and Readability:** Tests are documentation. They should be easy to understand, clearly stating the scenario being tested and the expected outcome.
- **Speed and Efficiency:** The test suite should run quickly to provide fast feedback during development. Optimize slow tests and ensure efficient setup/teardown.

## Test Structure and Organization

- **Arrange, Act, Assert (AAA):** Structure tests clearly:
    1.  **Arrange:** Set up the preconditions (instantiate objects, prepare data, mock dependencies).
    2.  **Act:** Execute the code under test.
    3.  **Assert:** Verify the outcome (check return values, state changes, exceptions raised).
- **Test Isolation:** Each test should run independently without relying on the state or outcome of other tests. Use fixtures and proper setup/teardown mechanisms (e.g., `pytest` fixtures) to ensure isolation.
- **Descriptive Naming:** Test function names should clearly describe the scenario being tested (e.g., `test_calculate_discount_for_gold_customer`, `test_connect_raises_timeout_error_on_unresponsive_server`).
- **File Organization:** Group related tests within the same file. Use subdirectories within the `tests/` folder to mirror the structure of the source code (`src/`).

## Test Types

- **Unit Tests:**
    - Focus on testing individual functions, methods, or classes in isolation.
    - Mock external dependencies (databases, network services, filesystem) to keep tests fast and focused.
    - Form the majority of the test suite.
- **Integration Tests:**
    - Verify the interaction between multiple components or modules.
    - May involve real dependencies (e.g., a test database, a local service) but should still be relatively fast.
    - Use sparingly compared to unit tests, focusing on critical integration points.
- **End-to-End (E2E) Tests:**
    - Test the entire application flow from start to finish, simulating real user scenarios.
    - Often slower and more complex to set up and maintain.
    - Use judiciously for validating critical user journeys.

## Assertions and Verification

- **Be Specific:** Use the most specific assertion possible (e.g., `assertEqual`, `assertTrue`, `assertRaises`). Avoid generic `assert True`.
- **Test Edge Cases:** Explicitly test boundary conditions, invalid inputs, empty collections, zero values, and potential error states.
- **Test Error Handling:** Verify that the code correctly raises expected exceptions under specific error conditions. Use `pytest.raises` or equivalent.
- **Avoid Asserting Mocks Directly:** Focus on verifying the *outcome* or *side effects* caused by interactions with mocks, not just that a mock method was called (unless the call itself *is* the primary outcome).

## Test Data and Fixtures

- **Use Realistic Data:** Employ data that resembles real-world scenarios but keep it as simple as possible to test the specific case.
- **Fixtures for Reusability:** Use testing framework fixtures (e.g., `pytest` fixtures) to manage setup and teardown logic, share test data, and manage dependencies.
- **Avoid Magic Values:** Use named constants or clearly defined variables for test data inputs and expected outputs.

## Mocking and Stubbing

- **Purposeful Mocking:** Use mocks/stubs primarily to isolate the code under test from external dependencies (I/O, network, time) or complex internal components *not* being tested directly.
- **Avoid Over-Mocking:** Do not mock collaborators within the same logical unit unless absolutely necessary. Excessive mocking can hide integration issues and make tests brittle.
- **Focus on Contracts:** When mocking, ensure the mock adheres to the contract (interface/protocol) of the real object.

## Coverage

- **Meaningful Coverage:** Aim for high *branch* and *condition* coverage, not just line coverage. Ensure critical logic paths and decision points are tested.
- **Coverage as a Guide:** Use coverage reports to identify *untested* code, but do not treat 100% coverage as the ultimate goal if it involves writing trivial or meaningless tests. Prioritize testing complex and critical sections.

## Typing and Tests

- **Leverage Type Hints:** Use type hints in test code for clarity and to allow static analysis tools to catch potential issues.
- **Test Type Contracts:** Ensure tests validate behavior consistent with the type annotations of the functions/methods being tested.
- **Type Safety in Fixtures:** Annotate fixtures correctly.

## Maintainability

- **DRY (Don't Repeat Yourself) Principle:** Use helper functions or fixtures to avoid duplicating test setup or assertion logic, but don't abstract so much that test clarity suffers.
- **Keep Tests Updated:** Tests must evolve alongside the code they test. Outdated or failing tests should be fixed or removed promptly.
- **Refactor Tests:** Just like production code, refactor tests when they become complex, slow, or difficult to understand. 