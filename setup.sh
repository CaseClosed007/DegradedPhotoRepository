#!/bin/bash
set -e

echo "Setting up Degraded Photo Detection..."

# Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Node dependencies
npm install

# Build React frontend into frontend/dist
cd frontend
npm install
npm run build
cd ..

echo ""
echo "Setup complete."
echo "Run:  npm start"
