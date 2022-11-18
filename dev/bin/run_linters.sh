#!/bin/bash
set -e

poetry run black .
poetry run isort spark_frame tests
poetry run flake8 spark_frame tests
poetry run mypy spark_frame

