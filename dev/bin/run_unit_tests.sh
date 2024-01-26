#!/bin/bash
set -e

poetry run pytest --cov -n 4 "$@"
