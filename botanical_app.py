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
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import urllib.parse
import textwrap
import streamlit.components.v1 as components

# Global headers for iNaturalist API requests
INAT_HEADERS = {
    'User-Agent': 'BotanicalWorkbench/1.0 (contact: daniel.cahen.substance@gmail.com)'
}

# Configure page with botanical theme
st.set_page_config(
    page_title="Botanical Identification Workbench (South Africa edition)",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- START: Bullet-proof CSS from Version 1 ---
st.markdown("""
<style>
    /* Theme-adaptive CSS variables for light/dark modes */
    :root {
        /* Light theme defaults (matching Streamlit light theme) */
        --text-primary: #1a1a1a;
        --text-secondary: #4a4a4a;
        --bg-primary: #ffffff;
        --bg-secondary: #f0f2f6;
        --input-bg: #ffffff;
        --table-bg: #f9f9f5;
        --table-header-bg: #e8e8e8;
        --sidebar-gradient: linear-gradient(180deg, #ffffff 0%, #f9f9f5 100%);
        --alert-bg: #ffffff;
        --border-color: #d0d0d0;
        --shadow-color: rgba(0,0,0,0.1);
        
        /* Botanical accents (kept consistent across themes) */
        --dark-green: #1a3d07;
        --forest-green: #2d5016;
        --sage-green: #5a7c3d;
        --light-sage: #a4b494;
        --earth-brown: #4a3c28;
        --light-background: #f9f9f5;
        --cream: #fffef9;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            /* Dark theme overrides (matching Streamlit dark theme) */
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --input-bg: #262730;
            --table-bg: #1a1a1a;
            --table-header-bg: #2a2a2a;
            --sidebar-gradient: linear-gradient(180deg, #262730 0%, #1a1a1a 100%);
            --alert-bg: #1a1a1a;
            --border-color: #404040;
            --shadow-color: rgba(0,0,0,0.3);
        }
    }
    
    /* Global app styling with theme vars for high contrast */
    .stApp {
        color: var(--text-primary);
        background-color: var(--bg-primary);
        font-family: 'Arial', sans-serif;
    }
    
    /* Main container background */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
    }
    
    /* Sidebar styling with theme-adaptive gradient and contrast */
    [data-testid="stSidebar"] {
        background: var(--sidebar-gradient);
        border-right: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    
    /* Ensure sidebar elements inherit high-contrast text */
    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: var(--text-primary) !important;
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        color-scheme: light dark; /* Helps with native input contrast */
    }
    
    /* Headers with theme-adaptive color */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 600;
        border-bottom: 2px solid var(--light-sage);
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    /* Metrics with improved contrast using theme vars */
    [data-testid="stMetricValue"] {
        color: var(--dark-green) !important;
        font-weight: 600;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-size: 0.9rem !important;
    }
    
    /* Professional button styling (accents consistent, text contrasts with bg) */
    .stButton > button {
        background-color: var(--forest-green);
        color: white !important; /* Force white text on green bg for high contrast */
        border-radius: 4px;
        border: 1px solid var(--dark-green);
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: 0 2px 4px var(--shadow-color);
    }
    
    .stButton > button:hover {
        background-color: var(--dark-green);
        box-shadow: 0 4px 8px var(--shadow-color);
        transform: translateY(-1px);
    }
    
    .stButton > button[kind="primary"] {
        background-color: var(--sage-green);
        border-color: var(--forest-green);
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: var(--forest-green);
    }
    
    /* Tab styling with theme-adaptive bg */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--bg-secondary);
        padding: 4px;
        border-radius: 4px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        background-color: transparent;
        border-radius: 3px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--forest-green) !important;
        color: white !important;
    }
    
    /* Expander with theme-adaptive contrast */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 4px 4px;
        padding: 1rem;
        color: var(--text-primary);
    }
    
    /* Alert boxes with theme-adaptive styling */
    .stAlert {
        background-color: var(--alert-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--forest-green);
        border-radius: 4px;
        color: var(--text-primary) !important;
    }
    
    /* Input fields with theme-adaptive styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        color: var(--text-primary) !important;
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 4px;
        padding: 8px 12px;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--forest-green) !important;
        box-shadow: 0 0 0 2px rgba(45, 80, 22, 0.1) !important;
    }
    
    /* Dataframe styling with theme-adaptive colors */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        background-color: var(--table-bg) !important;
        color: var(--text-primary) !important;
    }
    
    .dataframe thead th {
        background-color: var(--table-header-bg) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
        border-bottom: 2px solid var(--border-color) !important;
    }
    
    .dataframe tbody td {
        color: var(--text-primary) !important;
        background-color: var(--table-bg) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--sage-green) !important;
    }
    
    /* Footer styling with theme adaptation */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--bg-secondary);
        color: var(--text-secondary);
        text-align: center;
        padding: 8px;
        font-size: 0.75rem;
        border-top: 1px solid var(--border-color);
        z-index: 999;
    }
    
    .footer a {
        color: var(--forest-green);
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)
# --- END: Bullet-proof CSS ---

# Suppress warnings and configure logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state using a function for cleanliness
def init_session_state():
    defaults = {
        'species_data': None,
        'selected_species': {},
        'eflora_data': None,
        'analysis_data': None,
        'all_records': None,
        'image_cache': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Optimized caching with TTL from Version 2
def file_cache(cache_dir="cache", ttl_hours=24):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs_copy = kwargs.copy()
            if 'latitude' in kwargs_copy:
                kwargs_copy['latitude'] = round(kwargs_copy['latitude'], 2)
            if 'longitude' in kwargs_copy:
                kwargs_copy['longitude'] = round(kwargs_copy['longitude'], 2)
            
            cache_key = f"{func.__name__}_{'_'.join(map(str, args))}_{'_'.join(f'{k}_{v}' for k, v in sorted(kwargs_copy.items()))}"
            cache_key = cache_key.replace('/', '_').replace('.', '_')[:200]
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < ttl_hours * 3600:
                    return joblib.load(cache_file)
            
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result
        return wrapper
    return decorator

# Optimized e-Flora loader from Version 2
@st.cache_data(ttl=3600)
def load_eflora_data():
    """Loads e-Flora data with improved error handling and validation."""
    try:
        required_files = {
            'data/taxon.txt': ['id', 'scientificName'],
            'data/vernacularname.txt': ['id', 'vernacularName'],
            'data/description.txt': ['id', 'description', 'type']
        }
        
        for filepath in required_files:
            if not os.path.exists(filepath):
                st.error(f"Missing required data file: `{filepath}`")
                return None
        
        taxa_df = pd.read_csv('data/taxon.txt', sep='\t', usecols=['id', 'scientificName'], dtype={'id': str}, on_bad_lines='skip')
        desc_df = pd.read_csv('data/description.txt', sep='\t', usecols=['id', 'description', 'type'], dtype={'id': str}, on_bad_lines='skip')
        vernacular_df = pd.read_csv('data/vernacularname.txt', sep='\t', usecols=['id', 'vernacularName'], dtype={'id': str}, on_bad_lines='skip')
        
        for df, name in zip([taxa_df, desc_df, vernacular_df], ['taxa', 'descriptions', 'vernacular']):
            df.rename(columns={'id': 'taxonID'}, inplace=True)
        
        taxa_df['cleanScientificName'] = taxa_df['scientificName'].apply(lambda x: ' '.join(str(x).split()[:2]) if pd.notna(x) else '')
        
        desc_agg = desc_df.groupby('taxonID').apply(lambda x: x.set_index('type')['description'].to_dict(), include_groups=False).reset_index(name='descriptions')
        vernacular_agg = vernacular_df.groupby('taxonID')['vernacularName'].apply(lambda x: list(set(x.dropna())), include_groups=False).reset_index()
        
        eflora_data = taxa_df.merge(desc_agg, on='taxonID', how='left').merge(vernacular_agg, on='taxonID', how='left')
        
        eflora_data.set_index('cleanScientificName', inplace=True)
        eflora_data = eflora_data[~eflora_data.index.duplicated(keep='first')]
        eflora_data = eflora_data[eflora_data.index != '']
        
        return eflora_data
        
    except Exception as e:
        st.error(f"Failed to load e-Flora data: {str(e)}")
        return None

def format_species_name(name):
    """Extract genus and species from scientific name."""
    if not name: return None
    parts = str(name).split()
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else name

@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def safe_gbif_backbone(name, kingdom='Plantae'):
    """Cached GBIF backbone lookup with retry. Corrected to remove 'verbose' parameter."""
    return gbif_species.name_backbone(name=name, kingdom=kingdom)

# License mappings
INAT_LICENSE_MAP = {
    'cc-by': 'CC BY 4.0', 'cc-by-sa': 'CC BY-SA 4.0', 'cc-by-nd': 'CC BY-ND 4.0',
    'cc-by-nc': 'CC BY-NC 4.0', 'cc-by-nc-nd': 'CC BY-NC-ND 4.0',
    'cc-by-nc-sa': 'CC BY-NC-SA 4.0', 'cc0': 'CC0 1.0', 'pd': 'Public Domain'
}
ALLOWED_INAT_LICENSES = ['cc-by', 'cc-by-sa', 'cc0', 'pd']

@st.cache_data(ttl=1800)
def get_species_images(species_name, limit=5):
    """Fetch iNaturalist photos with caching and better error handling."""
    try:
        encoded_name = urllib.parse.quote(species_name)
        search_url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={encoded_name}&per_page=1"
        response = requests.get(search_url, headers=INAT_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'): return [], None
            
        taxon = data['results'][0]
        taxon_id = taxon['id']
        photos = []
        
        # Process default photo
        if default_photo := taxon.get('default_photo'):
            if photo_url := (default_photo.get('medium_url') or default_photo.get('square_url')):
                attribution = default_photo.get('attribution', '(c) Unknown')
                photographer = attribution.split('(c)')[-1].split(',')[0].strip() if '(c)' in attribution else 'Unknown'
                if license_code := default_photo.get('license_code', ''):
                    if license_code in ALLOWED_INAT_LICENSES:
                        license_name = INAT_LICENSE_MAP.get(license_code, 'Unknown')
                        photos.append({'url': photo_url, 'photographer': photographer, 'license': license_name, 'caption': f"© {photographer} · {license_name}"})
        
        # Fetch observation photos
        obs_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&photos=true&per_page={limit}&order_by=votes&order=desc"
        obs_response = requests.get(obs_url, headers=INAT_HEADERS, timeout=10)
        
        if obs_response.status_code == 200:
            for obs in obs_response.json().get('results', []):
                if len(photos) >= limit: break
                user = obs.get('user', {})
                photographer_name = user.get('name') or user.get('login', 'Unknown')
                for photo in obs.get('photos', []):
                    if photo_url := (photo.get('url', '').replace('square', 'medium') or photo.get('medium_url')):
                        if photo_url not in [p['url'] for p in photos]:
                            if license_code := photo.get('license_code', ''):
                                if license_code in ALLOWED_INAT_LICENSES:
                                    license_name = INAT_LICENSE_MAP.get(license_code, 'Unknown')
                                    photos.append({'url': photo_url, 'photographer': photographer_name, 'license': license_name, 'caption': f"© {photographer_name} · {license_name}"})
                                    if len(photos) >= limit: break
        
        return photos, taxon_id
        
    except requests.RequestException as e:
        logger.error(f"API error fetching iNat images for {species_name}: {e}")
        return [], None
    except Exception as e:
        logger.error(f"Unexpected error fetching iNat images for {species_name}: {e}")
        return [], None

@file_cache(cache_dir="gbif_cache", ttl_hours=48)
def get_species_list_from_gbif(latitude, longitude, radius_km, taxon_name, record_limit=50000):
    """Optimized GBIF query using bounding box."""
    try:
        taxon_info = safe_gbif_backbone(taxon_name)
        if 'usageKey' not in taxon_info or taxon_info.get('matchType') == 'NONE':
            return [], [], f"Taxon '{taxon_name}' not found in GBIF."
        
        search_taxon_key = taxon_info['usageKey']
        status = taxon_info.get('status', 'unknown').upper()
        synonym = taxon_info.get('synonym', False)
        status_flag = f" ({status}{' - Synonym' if synonym else ''})" if status != 'ACCEPTED' else ""

        lat_offset = radius_km / 111.32
        lon_offset = radius_km / (111.32 * abs(math.cos(math.radians(latitude))))
        
        params = {
            'taxonKey': search_taxon_key,
            'decimalLatitude': f'{latitude - lat_offset},{latitude + lat_offset}',
            'decimalLongitude': f'{longitude - lon_offset},{longitude + lon_offset}',
            'hasCoordinate': True, 'hasGeospatialIssue': False, 'limit': 300
        }

        all_records, offset = [], 0
        progress_bar = st.progress(0, text="Fetching GBIF records...")
        
        while offset < record_limit:
            params['offset'] = offset
            try:
                response = gbif_occ.search(**params)
                batch = response.get('results', [])
                if not batch: break
                all_records.extend(batch)
                progress = min(len(all_records) / record_limit, 1.0)
                progress_bar.progress(progress, text=f"Fetched {len(all_records)} records...")
                if len(batch) < 300: break
                offset += len(batch)
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error fetching GBIF batch: {e}")
                break
        
        progress_bar.empty()

        species_dict = {}
        for record in all_records:
            if (species_name := record.get('species')) and (species_key := record.get('speciesKey')):
                if species_name not in species_dict:
                    species_dict[species_name] = {
                        'name': species_name, 'count': 0, 'family': record.get('family', 'Unknown'),
                        'taxon_key': species_key, 'status_flag': status_flag, 'sample_records': []
                    }
                species_dict[species_name]['count'] += 1
                if len(species_dict[species_name]['sample_records']) < 50:
                    species_dict[species_name]['sample_records'].append({
                        'lat': record.get('decimalLatitude'), 'lon': record.get('decimalLongitude'),
                        'date': record.get('eventDate', 'Unknown')
                    })
        
        return sorted(species_dict.values(), key=lambda x: x['count'], reverse=True), all_records, None
        
    except Exception as e:
        return [], [], f"GBIF search failed: {str(e)}"

def get_local_eflora_description(scientific_name, eflora_data):
    """Optimized e-Flora description retrieval."""
    if eflora_data is None: return False, "e-Flora data not available."
    
    clean_name = format_species_name(scientific_name)
    
    if clean_name in eflora_data.index:
        matched_name = clean_name
    else:
        matches = process.extractOne(clean_name, eflora_data.index, scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 90:
            matched_name = matches[0]
        else:
            return False, f"Species '{clean_name}' not found in local database."
    
    try:
        row = eflora_data.loc[matched_name]
        descriptions = row.get('descriptions', {})
        vernacular_raw = row.get('vernacularName', [])
        
        vernacular_names = [str(n).strip() for n in vernacular_raw if pd.notna(n)] if isinstance(vernacular_raw, list) else []
        
        if not isinstance(descriptions, dict) or not descriptions:
            return False, f"No description available for '{clean_name}'."

        parts = [f"**Scientific Name:** {row['scientificName']}"]
        if vernacular_names:
            parts.append(f"**Common Names:** {', '.join(vernacular_names[:5])}")

        priority = ["Morphological description", "Diagnostic characters", "Habitat", "Distribution", "Description"]
        added_sections = []
        for section in priority:
            if (desc := descriptions.get(section)) and pd.notna(desc) and len(str(desc).strip()) > 10:
                parts.append(f"**{section}:**\n{str(desc).strip()}")
                added_sections.append(section)
        
        if not added_sections:
            return False, f"No detailed descriptions available for '{clean_name}'."

        parts.append("\n**Citation:** e-Flora of South Africa. v1.42. 2023. SANBI. (CC-BY 4.0)")
        return True, "\n\n".join(parts)
        
    except Exception as e:
        return False, f"Error retrieving data for '{clean_name}': {str(e)}"

def create_species_map(species_list, center_lat, center_lon):
    """Create optimized folium map."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='OpenStreetMap')
    folium.Marker([center_lat, center_lon], popup="Search Center", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    
    colors = ['#2d5016', '#4a7c24', '#6b8e23', '#8fbc8f', '#556b2f', '#697565', '#8b7355', '#6b4423', '#7c5e4c', '#a0826d']
    
    for i, species in enumerate(species_list[:10]):
        color = colors[i % len(colors)]
        for record in species.get('sample_records', []):
            if (lat := record.get('lat')) and (lon := record.get('lon')):
                folium.CircleMarker(location=[lat, lon], radius=4, popup=f"<b>{species['name']}</b>", color=color, fill=True, fillColor=color, weight=2, opacity=0.8, fillOpacity=0.6).add_to(m)
    return m

def add_footer():
    """Adds a professional footer with creator information."""
    st.markdown(f"""
    <div class="footer">
        Created by Daniel Cahen | © {datetime.now().year} | MIT License | 
        <a href="https://github.com/dcahen" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Main Streamlit App
def main():
    st.title("🌿 Botanical Identification Workbench (South Africa edition)")
    st.markdown("*Advanced species identification using GBIF data and local flora databases*")
    
    with st.sidebar:
        st.header("🔧 Search Parameters")
        st.subheader("📍 Location")
        latitude = st.number_input("Latitude", value=-33.92, format="%.6f")
        longitude = st.number_input("Longitude", value=18.42, format="%.6f")
        radius_km = st.slider("Search Radius (km)", 1, 200, 25)
        
        st.subheader("🌱 Taxon Search")
        taxon_name = st.text_input("Taxon Name", value="Protea")
        
        st.subheader("⚙️ Options")
        include_images = st.checkbox("Include Images", value=True)
        export_format = st.selectbox("Export Format", ["Markdown", "JSON"])
        
        if st.button("🔍 Search GBIF", type="primary", use_container_width=True):
            with st.spinner("Loading e-Flora database..."):
                st.session_state.eflora_data = load_eflora_data()
            
            if st.session_state.eflora_data is not None:
                species_data, all_records, error = get_species_list_from_gbif(latitude, longitude, radius_km, taxon_name)
                if error:
                    st.error(error)
                else:
                    st.session_state.species_data = species_data
                    st.session_state.all_records = all_records
                    st.session_state.selected_species = {}
                    st.session_state.analysis_data = None
                    st.success(f"Found {len(species_data)} species!")
                    st.rerun()
            else:
                st.error("Cannot proceed without e-Flora data. Please check file paths and format.")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            import shutil
            for cache_dir in ['gbif_cache', 'cache']:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            st.cache_data.clear()
            st.success("Cache cleared!")

    if st.session_state.species_data is None:
        st.info("👆 Configure search parameters in the sidebar and click 'Search GBIF' to begin.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Species List", "🗺️ Map View", "📄 Export"])
        
        with tab1:
            st.subheader("Species Found in Search Area")
            df = pd.DataFrame(st.session_state.species_data)
            df_display = df[['name', 'family', 'count']].rename(columns={'name': 'Species', 'family': 'Family', 'count': 'Records'})
            st.dataframe(df_display, use_container_width=True, height=300)
            
            st.subheader("Select Species for Detailed Analysis")
            species_options = [f"{sp['name']} - {sp['family']} ({sp['count']})" for sp in st.session_state.species_data[:50]]
            
            col1, col2, col3 = st.columns(3)
            if col1.button("Select All (Top 50)", use_container_width=True):
                st.session_state.selected_species = {sp['name']: True for sp in st.session_state.species_data[:50]}
                st.rerun()
            if col2.button("Clear Selection", use_container_width=True):
                st.session_state.selected_species = {}
                st.session_state.analysis_data = None
                st.rerun()
            if col3.button("Top 10", use_container_width=True):
                st.session_state.selected_species = {sp['name']: True for sp in st.session_state.species_data[:10]}
                st.rerun()
            
            selected_labels = st.multiselect("Select species:", species_options, default=[opt for opt in species_options if opt.split(' - ')[0] in st.session_state.selected_species])
            selected_names = [label.split(' - ')[0] for label in selected_labels]
            st.session_state.selected_species = {name: True for name in selected_names}
            
            if selected_names:
                if st.button("📊 Generate Detailed Analysis", type="primary", use_container_width=True):
                    st.session_state.analysis_data = [sp for sp in st.session_state.species_data if sp['name'] in selected_names]
                    st.rerun()
                
                if st.session_state.analysis_data:
                    st.subheader("🔍 Detailed Species Information")
                    for species in st.session_state.analysis_data:
                        with st.expander(f"📋 {species['name']} ({species['count']} records)", expanded=True):
                            cols = st.columns([3, 1]) if include_images else st.columns(1)
                            with cols[0]:
                                success, description = get_local_eflora_description(species['name'], st.session_state.eflora_data)
                                if success:
                                    st.markdown(description)
                                else:
                                    st.warning(description)
                            
                            if include_images and len(cols) > 1:
                                with cols[1]:
                                    with st.spinner("Loading images..."):
                                        images_data, taxon_id = get_species_images(species['name'])
                                        if images_data:
                                            for img_data in images_data[:3]:
                                                st.image(img_data['url'], caption=img_data['caption'], use_container_width=True)
                                            if taxon_id:
                                                st.markdown(f"[View on iNaturalist ↗](https://www.inaturalist.org/taxa/{taxon_id})")
                                        else:
                                            st.info("No suitable images found.")
        
        with tab2:
            st.subheader("🗺️ Species Distribution Map")
            if st.session_state.species_data:
                species_map = create_species_map(st.session_state.species_data, latitude, longitude)
                st_folium(species_map, height=600, width=700)
            else:
                st.warning("Map data not available.")
        
        with tab3:
            st.subheader("📄 Export Data")
            export_species = st.session_state.analysis_data or [sp for sp in st.session_state.species_data if sp['name'] in st.session_state.selected_species]
            
            if not export_species:
                st.warning("Please select species in the 'Species List' tab first.")
            else:
                st.info(f"Ready to export data for {len(export_species)} species.")
                
                if export_format == "JSON":
                    export_data = {"metadata": {"location": {"latitude": latitude, "longitude": longitude}, "radius_km": radius_km, "taxon_searched": taxon_name, "timestamp": datetime.now().isoformat()}, "species": []}
                    for species in export_species:
                        success, description = get_local_eflora_description(species['name'], st.session_state.eflora_data)
                        export_data["species"].append({"name": species['name'], "family": species['family'], "gbif_count": species['count'], "description": description if success else "No description available."})
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(label="📥 Download JSON", data=json_str, file_name=f"botanical_data_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json", use_container_width=True)
                    
                    with st.expander("Preview JSON", expanded=True):
                        components.html(f"""<button id="copyBtn" style="background-color: #2d5016; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; float: right;">Copy JSON</button><script>document.getElementById('copyBtn').addEventListener('click', function() {{navigator.clipboard.writeText({json.dumps(json_str)}).then(() => alert('Copied!'));}});</script>""", height=50)
                        st.json(export_data)
                else: # Markdown
                    markdown_parts = [f"# Botanical Report\n**Location:** {latitude}, {longitude}\n**Radius:** {radius_km} km\n**Taxon:** {taxon_name}\n---"]
                    for species in export_species:
                        success, description = get_local_eflora_description(species['name'], st.session_state.eflora_data)
                        markdown_parts.append(f"### {species['name']}\n**Family:** {species['family']} | **GBIF Records:** {species['count']}\n\n{description if success else 'No description available.'}\n\n---")
                    
                    markdown_str = "\n".join(markdown_parts)
                    st.download_button(label="📥 Download Markdown", data=markdown_str, file_name=f"botanical_report_{datetime.now().strftime('%Y%m%d')}.md", mime="text/markdown", use_container_width=True)
                    
                    with st.expander("Preview Markdown", expanded=True):
                        components.html(f"""<button id="copyBtnMd" style="background-color: #2d5016; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; float: right;">Copy Markdown</button><script>document.getElementById('copyBtnMd').addEventListener('click', function() {{navigator.clipboard.writeText({json.dumps(markdown_str)}).then(() => alert('Copied!'));}});</script>""", height=50)
                        st.markdown(markdown_str)

    add_footer()

if __name__ == "__main__":
    main()
