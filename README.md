# 🌿 Botanical Identification Workbench

Advanced species identification using GBIF data and local flora databases.

## Features

- 🔍 GBIF database integration for species occurrence data
- 📍 Geographic search with customizable radius
- 🗺️ Interactive maps showing species distribution
- 📊 Phenological analysis (seasonal observation patterns)
- 🖼️ Species imagery integration
- 📄 Multiple export formats (Markdown, JSON)
- 🌱 Local e-Flora database integration

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure data files are in the `data/` directory
4. Run: `streamlit run botanical_app.py`

## Data Requirements

Place these files in the `data/` directory:
- `taxon.txt` - Taxa information with columns: id, scientificName
- `description.txt` - Species descriptions with columns: id, description, type  
- `vernacularname.txt` - Common names with columns: id, vernacularName

All files should be tab-separated format.

## Usage

1. Set your search location (latitude/longitude)
2. Choose search radius and target taxon
3. Click "Search GBIF" to find species
4. Select species of interest for detailed analysis
5. View results in interactive tabs
6. Export data in your preferred format

## Deployment

This app is designed to run on Streamlit Cloud. Make sure your data files are included in the repository.
