#!/bin/bash
set -e

poetry run coverage run -m pytest "$@"

poetry run coverage xml
