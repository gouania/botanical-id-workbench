#!/bin/bash
# setup.sh - Setup script for Botanical Workbench

echo "🌿 Setting up Botanical Workbench..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Prepare data
echo "📊 Preparing data..."
python prepare_data.py

# Create necessary directories
mkdir -p cache gbif_cache

echo "✅ Setup complete!"
echo "Run 'streamlit run botanical_workbench.py' to start the application"
