# Contributing to Dark_Circle

## 1. Welcome
Thank you for your interest in contributing to Dark_Circle. Given the technical depth of this document and the LVC Toolkit, we highly encourage contributions to improve the codebase.

## 2. Table of Contents
* [Folder Structure](#3-folder-structure-and-code-organization)
* [Style Information](#4-style-information)
* [Git Branching Plan](#5-git-branching-plan)
* [Testing Information](#6-testing-information)
* [Project Tracker](#7-project-management-tracker)

## 3. Folder Structure and Code Organization
*Use this section to help newcomers find code quickly.*

* `/src`: Main source code for the toolkit.
* `/tests`: Unit and integration tests.
* `/docs`: Documentation and reference materials.
* `/scripts`: Deployment and utility scripts.

## 4. Style Information
*We enforce the following style standards to ensure consistency.*

* **Linter:** [e.g., Flake8, Pylint]
* **Formatter:** [e.g., Black]
* **Commit Messages:** Imperative mood (e.g., "Add feature" not "Added feature").

Run the style check locally before pushing:
```bash
# Example command
make lint
