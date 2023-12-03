#!/bin/bash
set -e

poetry run ruff check .
poetry run mypy spark_frame
