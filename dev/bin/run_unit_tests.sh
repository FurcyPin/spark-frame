#!/bin/bash
set -e

poetry run coverage run -m pytest -n 4 "$@"

poetry run coverage xml
