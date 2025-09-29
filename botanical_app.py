import streamlit as st
import pandas as pd
import pygbif.species as gbif_species
import pygbif.occurrences as gbif_occ
import math
import time
import os
import json
import requests
from datetime import datetime
from tqdm import tqdm
import warnings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from rapidfuzz import fuzz, process
import joblib
import functools
from concurrent.futures import ThreadPoolExecutor
import folium
from streamlit_folium import st_folium
import tiktoken
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Botanical ID Workbench",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'species_data' not in st.session_state:
    st.session_state.species_data = None
if 'selected_species' not in st.session_state:
    st.session_state.selected_species = {}
if 'eflora_data' not in st.session_state:
    st.session_state.eflora_data = None

# Caching decorator
def file_cache(cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create consistent cache key
            if 'latitude' in kwargs:
                kwargs['latitude'] = round(kwargs['latitude'], 2)
            if 'longitude' in kwargs:
                kwargs['longitude'] = round(kwargs['longitude'], 2)
            arg_str = "_".join(map(str, args))
            kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
            cache_key = f"{func.__name__}_{arg_str}_{kwarg_str}".replace('/', '_').replace('.', '_')
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                return joblib.load(cache_file)
            
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result
        return wrapper
    return decorator

# Core functions from your notebook
@st.cache_data
def download_and_load_eflora():
    """Loads e-Flora data from local files with Streamlit caching."""
    try:
        # Check if required files exist
        required_files = ['data/taxon.txt', 'data/vernacularname.txt', 'data/description.txt']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"Missing required files: {missing_files}")
            st.info("Please ensure the following files are in your 'data' directory:")
            for f in required_files:
                st.write(f"- {f}")
            return None
        
        st.info("Loading e-Flora database from local files...")
        
        # Load the three main files
        taxa_df = pd.read_csv('data/taxon.txt', sep='\t', header=0, 
                            usecols=['id', 'scientificName'], dtype={'id': str})
        
        desc_df = pd.read_csv('data/description.txt', sep='\t', header=0, 
                            usecols=['id', 'description', 'type'], dtype={'id': str})
        
        vernacular_df = pd.read_csv('data/vernacularname.txt', sep='\t', header=0, 
                                  usecols=['id', 'vernacularName'], dtype={'id': str})
        
        # Rename id columns for consistency
        for df in [taxa_df, desc_df]:
            df.rename(columns={'id': 'taxonID'}, inplace=True)
        vernacular_df.rename(columns={'id': 'taxonID'}, inplace=True)
        
        # Create clean scientific names (genus + species only)
        taxa_df['cleanScientificName'] = taxa_df['scientificName'].apply(
            lambda x: ' '.join(str(x).split()[:2]) if pd.notna(x) else ''
        )
        
        # Aggregate descriptions by type
        desc_agg = desc_df.groupby('taxonID').apply(
            lambda x: x.set_index('type')['description'].to_dict()
        ).reset_index(name='descriptions')
        
        # Aggregate vernacular names
        vernacular_agg = vernacular_df.groupby('taxonID')['vernacularName'].apply(
            lambda x: list(set(x.dropna()))
        ).reset_index()
        
        # Merge all data
        eflora_data = pd.merge(taxa_df, desc_agg, on='taxonID', how='left')
        eflora_data = pd.merge(eflora_data, vernacular_agg, on='taxonID', how='left')
        
        # Set index and remove duplicates
        eflora_data.set_index('cleanScientificName', inplace=True)
        eflora_data = eflora_data[~eflora_data.index.duplicated(keep='first')]
        
        # Remove empty entries
        eflora_data = eflora_data[eflora_data.index != '']
        
        st.success(f"Loaded {len(eflora_data)} taxa from e-Flora database")
        return eflora_data
        
    except Exception as e:
        st.error(f"Failed to load e-Flora data: {e}")
        st.info("Make sure your data files are properly formatted with tab separation and required columns.")
        return None

def format_species_name(name):
    if not name: return None
    parts = name.split()
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else name

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def safe_gbif_backbone(name, kingdom='Plantae'):
    return gbif_species.name_backbone(name=name, kingdom=kingdom, verbose=False)

@st.cache_data
def get_species_image(species_name, limit=1):
    """Fetch images for a species from GBIF occurrences with images."""
    try:
        taxon_info = safe_gbif_backbone(species_name)
        if 'usageKey' not in taxon_info:
            return None
        response = gbif_occ.search(
            taxonKey=taxon_info['usageKey'], 
            hasImage=True, 
            limit=limit
        )
        if response and 'results' in response and response['results']:
            media = response['results'][0].get('media', [])
            if media:
                return media[0].get('identifier')
        return None
    except:
        return None

def extract_month_from_date(date_str):
    """Extract month from GBIF eventDate."""
    if not date_str:
        return None
    try:
        if '-' in date_str:
            return int(date_str.split('-')[1])
        elif '/' in date_str:
            return int(date_str.split('/')[0])
        else:
            return None
    except:
        return None

def calculate_phenology(records):
    """Aggregate phenology from records."""
    months = [extract_month_from_date(r.get('eventDate')) for r in records if extract_month_from_date(r.get('eventDate'))]
    if not months:
        return {m: 0 for m in range(1, 13)}
    month_counts = {m: months.count(m) for m in range(1, 13)}
    return month_counts

@file_cache(cache_dir="gbif_cache")
def get_species_list_from_gbif(latitude, longitude, radius_km, taxon_name, record_limit=50000):
    """Queries GBIF with caching."""
    try:
        # Backbone match
        taxon_info = safe_gbif_backbone(taxon_name)
        if 'usageKey' not in taxon_info or taxon_info.get('matchType') == 'NONE':
            st.error(f"Taxon '{taxon_name}' not found in GBIF")
            return [], []
        
        search_taxon_key = taxon_info['usageKey']
        status = taxon_info.get('status', 'unknown').upper()
        synonym = taxon_info.get('synonym', False)
        status_flag = f" ({status}{' - Synonym' if synonym else ''})" if status != 'ACCEPTED' else ""

        # Bounding box
        lat_offset = radius_km / 111.32
        lon_offset = radius_km / (111.32 * abs(math.cos(math.radians(latitude))))
        params = {
            'taxonKey': search_taxon_key,
            'decimalLatitude': f'{latitude - lat_offset},{latitude + lat_offset}',
            'decimalLongitude': f'{longitude - lon_offset},{longitude + lon_offset}',
            'hasCoordinate': True,
            'hasGeospatialIssue': False,
            'limit': 300
        }

        all_records = []
        offset = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while offset < record_limit:
            params['offset'] = offset
            try:
                response = gbif_occ.search(**params)
                batch = response.get('results', [])
                if not batch:
                    break
                all_records.extend(batch)
                
                progress = min(len(all_records) / min(record_limit, 10000), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Fetched {len(all_records)} records...")
                
                if len(batch) < 300:
                    break
                offset += len(batch)
                time.sleep(0.05)
                
                if len(all_records) >= 100000:
                    break
            except Exception as e:
                st.error(f"Error fetching batch: {e}")
                break

        progress_bar.empty()
        status_text.empty()

        # Aggregate unique species
        species_dict = {}
        for record in all_records:
            species_name = record.get('species')
            species_key = record.get('speciesKey')
            if species_name and species_key:
                if species_name not in species_dict:
                    species_dict[species_name] = {
                        'name': species_name, 'count': 0, 'family': record.get('family', 'Unknown'),
                        'taxon_key': species_key, 'status_flag': status_flag, 'phenology': {}, 'records': []
                    }
                species_dict[species_name]['count'] += 1
                species_dict[species_name]['records'].append(record)
        
        # Compute phenology per species
        for sp in species_dict.values():
            sp['phenology'] = calculate_phenology(sp['records'])
        
        species_list = sorted(species_dict.values(), key=lambda x: x['count'], reverse=True)
        return species_list, all_records
        
    except Exception as e:
        st.error(f"GBIF search failed: {e}")
        return [], []

def get_local_eflora_description(scientific_name, eflora_data):
    """Get description from local e-Flora data with fuzzy matching."""
    if eflora_data is None:
        return False, "e-Flora data not available"
    
    clean_name = format_species_name(scientific_name)
    
    # Try exact match first
    if clean_name in eflora_data.index:
        matched_name = clean_name
    else:
        # Try fuzzy matching
        matches = process.extractOne(clean_name, eflora_data.index, scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 90:  # 90% similarity threshold
            matched_name = matches[0]
        else:
            return False, f"Species {clean_name} not found in local database (even with fuzzy matching)"
    
    try:
        # Retrieve the matched row
        row = eflora_data.loc[matched_name]
        descriptions = row['descriptions']
        vernacular_raw = row['vernacularName']
        full_scientific_name = row['scientificName']
        
        # Handle vernacular names safely
        if isinstance(vernacular_raw, list):
            vernacular_names = [str(n).strip() for n in vernacular_raw if pd.notna(n) and str(n).strip()]
        elif isinstance(vernacular_raw, str):
            vernacular_names = [vernacular_raw.strip()] if vernacular_raw.strip() else []
        else:
            vernacular_names = []
        
        # Check if descriptions is a valid dictionary
        if not isinstance(descriptions, dict) or not descriptions:
            return False, f"No description available for {clean_name}"

        # Build description with priority sections
        extracted_data = [f"**Scientific Name:** {full_scientific_name}"]
        
        if vernacular_names:
            extracted_data.append(f"**Common Names:** {', '.join(vernacular_names[:5])}")  # Limit to 5 names

        # Priority sections in order of importance
        priority_sections = [
            "Morphological description", 
            "Diagnostic characters", 
            "Habitat", 
            "Distribution",
            "Morphology",
            "Diagnostic",
            "Description",
            "Characters"
        ]
        
        sections_added = 0
        for section in priority_sections:
            if section in descriptions and pd.notna(descriptions[section]):
                desc_text = str(descriptions[section]).strip()
                if desc_text and len(desc_text) > 10:  # Ignore very short descriptions
                    extracted_data.append(f"**{section}:**\n{desc_text}")
                    sections_added += 1
                    if sections_added >= 4:  # Limit to 4 sections to avoid overwhelming
                        break
        
        # If no priority sections found, add any available sections
        if sections_added == 0:
            for section, desc in descriptions.items():
                if pd.notna(desc):
                    desc_text = str(desc).strip()
                    if desc_text and len(desc_text) > 10:
                        extracted_data.append(f"**{section}:**\n{desc_text}")
                        sections_added += 1
                        if sections_added >= 2:  # Limit to 2 if using fallback sections
                            break
        
        if sections_added == 0:
            return False, f"No detailed descriptions available for {clean_name}"
            
        return True, "\n\n".join(extracted_data)
        
    except Exception as e:
        st.error(f"Error processing {clean_name}: {e}")
        return False, f"Error retrieving data for {clean_name}"

def create_phenology_chart(phenology_data, species_name):
    """Create a chart using Plotly."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    counts = [phenology_data.get(i, 0) for i in range(1, 13)]
    
    fig = go.Figure(data=go.Bar(x=months, y=counts))
    fig.update_layout(
        title=f"Observations by Month: {species_name}",
        xaxis_title="Month",
        yaxis_title="Observation Count",
        height=400
    )
    return fig

def create_species_map(records, species_list, center_lat, center_lon):
    """Create an interactive map with species observations."""
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add center point
    folium.Marker(
        [center_lat, center_lon],
        popup="Search Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Color mapping for top species
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'gray']
    
    # Add species points (sample for performance)
    for i, species in enumerate(species_list[:10]):
        color = colors[i % len(colors)]
        species_records = [r for r in records if r.get('species') == species['name']]
        
        for record in species_records[:50]:  # Limit points per species
            lat = record.get('decimalLatitude')
            lon = record.get('decimalLongitude')
            if lat and lon:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    popup=f"<b>{species['name']}</b><br>Family: {species['family']}<br>Date: {record.get('eventDate', 'Unknown')}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    weight=2
                ).add_to(m)
    
    # Add legend with better visibility
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: auto; 
                background-color: rgba(255, 255, 255, 0.98); border:2px solid #333; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); color: #333; font-weight: bold;">
    <b>Species Legend</b><br>
    '''
    for i, species in enumerate(species_list[:10]):
        color = colors[i % len(colors)]
        legend_html += f'<i style="background:{color}; width:16px; height:16px; display: inline-block; margin-right: 6px; border:1px solid #333;"></i> {species["name"]}<br>'
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Main Streamlit App
def main():
    st.title("🌿 Botanical Identification Workbench")
    st.markdown("*Advanced species identification using GBIF data and local flora databases*")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Search Parameters")
        
        # Location settings
        st.subheader("📍 Location")
        latitude = st.number_input("Latitude", value=-33.92, format="%.6f")
        longitude = st.number_input("Longitude", value=18.42, format="%.6f")
        radius_km = st.slider("Search Radius (km)", 1, 200, 25)
        
        # Search parameters  
        st.subheader("🌱 Taxon Search")
        taxon_name = st.text_input("Taxon Name", value="Protea")
        user_features = st.text_area("Observed Features", 
                                   placeholder="e.g., yellow flowers, serrated leaves...")
        
        # Options
        st.subheader("⚙️ Options")
        include_images = st.checkbox("Include Images (slower)")
        export_format = st.selectbox("Output Format", ["Markdown", "JSON"])
        
        # Actions
        if st.button("🔍 Search GBIF", type="primary"):
            with st.spinner("Searching GBIF database..."):
                # Load e-Flora data if not already loaded
                if st.session_state.eflora_data is None:
                    st.session_state.eflora_data = download_and_load_eflora()
                
                if st.session_state.eflora_data is not None:
                    species_data, all_records = get_species_list_from_gbif(
                        latitude, longitude, radius_km, taxon_name
                    )
                    st.session_state.species_data = species_data
                    st.session_state.all_records = all_records
                    st.session_state.selected_species = {}
                    if 'species_selector' in st.session_state:
                        del st.session_state.species_selector
                    st.rerun()
                else:
                    st.error("Cannot proceed without e-Flora data. Please check your data files.")
        
        # Data diagnostics section
        with st.expander("🔧 Data Diagnostics"):
            if st.button("Test e-Flora Data Loading"):
                eflora_data = download_and_load_eflora()
                if eflora_data is not None:
                    st.success(f"Successfully loaded {len(eflora_data)} taxa")
                    
                    # Show sample data
                    st.subheader("Sample Taxa (first 5):")
                    sample_df = pd.DataFrame({
                        'Scientific Name': eflora_data['scientificName'].head(),
                        'Has Descriptions': [bool(desc) for desc in eflora_data['descriptions'].head()],
                        'Vernacular Names': [len(vn) if isinstance(vn, list) else 0 for vn in eflora_data['vernacularName'].head()]
                    })
                    st.dataframe(sample_df)
                    
                    # Show available description types
                    all_desc_types = set()
                    for descriptions in eflora_data['descriptions'].dropna():
                        if isinstance(descriptions, dict):
                            all_desc_types.update(descriptions.keys())
                    
                    st.subheader("Available Description Types:")
                    st.write(sorted(list(all_desc_types)))
                else:
                    st.error("Failed to load e-Flora data")
            
            st.markdown("**File Requirements:**")
            st.write("- `data/taxon.txt`: Must have columns 'id' and 'scientificName'")
            st.write("- `data/description.txt`: Must have columns 'id', 'description', and 'type'")
            st.write("- `data/vernacularname.txt`: Must have columns 'id' and 'vernacularName'")
            st.write("- All files should be tab-separated (.txt) format")
        
        if st.button("🗑️ Clear Cache"):
            # Clear cache directories
            for cache_dir in ['gbif_cache', 'cache']:
                if os.path.exists(cache_dir):
                    import shutil
                    shutil.rmtree(cache_dir)
            st.success("Cache cleared!")
    
    # Main content area
    if st.session_state.species_data is None:
        st.info("👆 Configure search parameters in the sidebar and click 'Search GBIF' to begin")
        
        # Show example location on map
        st.subheader("📍 Search Location Preview")
        preview_map = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker(
            [latitude, longitude], 
            popup="Search Center",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(preview_map)
        folium.Circle(
            location=[latitude, longitude],
            radius=radius_km * 1000,  # Convert to meters
            popup=f"Search radius: {radius_km} km",
            color="blue",
            fill=False
        ).add_to(preview_map)
        st_folium(preview_map, height=400, width=700)
        
    else:
        # Display search results
        st.success(f"Found {len(st.session_state.species_data)} species!")
        
        # Prepare species options for multiselect
        species_list_limited = st.session_state.species_data[:50]
        species_options = [
            f"{sp['name']} - *{sp['family']}* ({sp['count']} record{'s' if sp['count'] != 1 else ''}){sp['status_flag']}"
            for sp in species_list_limited
        ]
        st.session_state.species_options = species_options
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Species List", "🗺️ Map View", "📈 Analysis", "📄 Export"])
        
        with tab1:
            st.subheader("Select Species for Detailed Analysis")
            
            # Create DataFrame for display
            df = pd.DataFrame(st.session_state.species_data)
            df_display = df[['name', 'family', 'count', 'status_flag']].copy()
            df_display.columns = ['Species', 'Family', 'Records', 'Status']
            st.dataframe(df_display)
            
            # Define callback functions
            def select_all():
                st.session_state.species_selector = st.session_state.species_options[:]
                st.rerun()
            
            def deselect_all():
                st.session_state.species_selector = []
                st.rerun()
            
            def top_ten():
                st.session_state.species_selector = st.session_state.species_options[:10]
                st.rerun()
            
            # Selection control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("Select All", on_click=select_all)
            with col2:
                st.button("Deselect All", on_click=deselect_all)
            with col3:
                st.button("Top 10 Only", on_click=top_ten)
            
            # Multiselect for species selection
            selected_labels = st.multiselect(
                "Select species for analysis:",
                st.session_state.species_options,
                key="species_selector",
                format_func=lambda x: x
            )
            
            # Map selected labels back to species names
            selected_names = []
            for label in selected_labels:
                # Extract name from label (first part before ' - ')
                name = label.split(' - ')[0]
                selected_names.append(name)
            
            # Update session state for compatibility
            st.session_state.selected_species = {name: True for name in selected_names}
            
            # Process selected species
            selected_count = len(selected_names)
            if selected_count > 0:
                st.info(f"Selected {selected_count} species for processing")
                
                if st.button("📊 Generate Analysis", type="primary"):
                    selected_species_data = [
                        sp for sp in st.session_state.species_data 
                        if sp['name'] in selected_names
                    ]
                    
                    with st.spinner("Processing selected species..."):
                        # Generate detailed analysis
                        st.subheader("🔍 Detailed Species Analysis")
                        
                        for species in selected_species_data[:10]:  # Limit for demo
                            with st.expander(f"📋 {species['name']} ({species['count']} records)", expanded=True):
                                if include_images:
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                else:
                                    col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    # Get description
                                    success, description = get_local_eflora_description(
                                        species['name'], st.session_state.eflora_data
                                    )
                                    
                                    if success:
                                        st.markdown(description)
                                    else:
                                        st.warning(f"No local description available: {description}")
                                        st.info(f"**Family:** {species['family']}")
                                        st.info(f"**GBIF Records:** {species['count']}")
                                
                                if include_images:
                                    with col2:
                                        # Image if enabled
                                        with st.spinner("Fetching image..."):
                                            image_url = get_species_image(species['name'])
                                            if image_url:
                                                try:
                                                    response = requests.get(image_url, timeout=10)
                                                    img = Image.open(io.BytesIO(response.content))
                                                    st.image(img, caption=f"Image of {species['name']}", use_container_width=True)
                                                except:
                                                    st.warning("Failed to load image")
                                            else:
                                                st.info("No image available")
                                with col3 if include_images else col2:
                                    # Observations by month chart
                                    if species.get('phenology'):
                                        fig = create_phenology_chart(species['phenology'], species['name'])
                                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🗺️ Species Distribution Map")
            if hasattr(st.session_state, 'all_records'):
                species_map = create_species_map(
                    st.session_state.all_records,
                    st.session_state.species_data,
                    latitude,
                    longitude
                )
                st_folium(species_map, height=600, width=700)
                st.info("🎯 Red info icon = search center. Colored dots = species observations (top 10 species shown). Check the legend for species-color mapping.")
            else:
                st.warning("Map data not available. Run search first.")
        
        with tab3:
            st.subheader("📈 Search Statistics")
            
            if st.session_state.species_data:
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                total_species = len(st.session_state.species_data)
                total_records = sum(sp['count'] for sp in st.session_state.species_data)
                families = set(sp['family'] for sp in st.session_state.species_data)
                avg_records = total_records / total_species if total_species > 0 else 0
                
                col1.metric("Species Found", total_species)
                col2.metric("Total Records", total_records)
                col3.metric("Families", len(families))
                col4.metric("Avg Records/Species", f"{avg_records:.1f}")
                
                # Top families chart
                family_counts = {}
                for species in st.session_state.species_data:
                    family = species['family']
                    family_counts[family] = family_counts.get(family, 0) + 1
                
                if family_counts:
                    top_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    families_df = pd.DataFrame(top_families, columns=['Family', 'Species Count'])
                    
                    fig = px.bar(families_df, x='Species Count', y='Family', orientation='h',
                               title="Top 10 Families by Species Count")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📄 Export Data")
            
            selected_species_data = [
                sp for sp in st.session_state.species_data 
                if sp['name'] in st.session_state.selected_species
            ]
            
            if selected_species_data:
                if export_format == "JSON":
                    # JSON export
                    export_data = {
                        "metadata": {
                            "location": {"latitude": latitude, "longitude": longitude},
                            "radius_km": radius_km,
                            "taxon_searched": taxon_name,
                            "selected_species_count": len(selected_species_data),
                            "user_features": user_features,
                            "timestamp": datetime.now().isoformat()
                        },
                        "species": []
                    }
                    
                    for species in selected_species_data:
                        success, description = get_local_eflora_description(
                            species['name'], st.session_state.eflora_data
                        )
                        
                        export_data["species"].append({
                            "name": species['name'],
                            "family": species['family'],
                            "gbif_count": species['count'],
                            "status_flag": species['status_flag'],
                            "description": description if success else "No description available",
                            "phenology": species.get('phenology', {})
                        })
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"botanical_data_{taxon_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                    
                    with st.expander("Preview JSON Output"):
                        st.json(export_data)
                
                else:
                    # Markdown export
                    markdown_parts = [
                        f"# Botanical Identification Report",
                        f"**Location:** {latitude}, {longitude}",
                        f"**Search Radius:** {radius_km} km",
                        f"**Target Taxon:** {taxon_name}",
                        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"**Selected Species:** {len(selected_species_data)}",
                        "",
                        "## Species Summary",
                        "| Species | Family | Records | Status |",
                        "|---------|--------|---------|--------|"
                    ]
                    
                    for species in selected_species_data:
                        markdown_parts.append(
                            f"| {species['name']} | {species['family']} | {species['count']} | {species['status_flag']} |"
                        )
                    
                    markdown_parts.extend(["", "## Detailed Descriptions", ""])
                    
                    for species in selected_species_data:
                        success, description = get_local_eflora_description(
                            species['name'], st.session_state.eflora_data
                        )
                        
                        markdown_parts.extend([
                            f"### {species['name']}",
                            f"**Family:** {species['family']} | **Records:** {species['count']}{species['status_flag']}",
                            "",
                            description if success else "No description available.",
                            "",
                            "---",
                            ""
                        ])
                    
                    markdown_str = "\n".join(markdown_parts)
                    
                    st.download_button(
                        label="📥 Download Markdown",
                        data=markdown_str,
                        file_name=f"botanical_report_{taxon_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown"
                    )
                    
                    with st.expander("Preview Markdown Output"):
                        st.markdown(markdown_str)
            
            else:
                st.warning("No species selected for export. Please select species in the Species List tab.")

if __name__ == "__main__":
    main()
