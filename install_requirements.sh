#!/bin/bash

echo "Uninstalling all packages..."
pip freeze | xargs pip uninstall -y

echo "Installing requirements from requirements.txt..."
pip install -r requirement.txt
