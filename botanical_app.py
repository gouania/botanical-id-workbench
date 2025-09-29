import streamlit as st
import pandas as pd
import pygbif.species as gbif_species
import pygbif.occurrences as gbif_occ
import math
import time
import os
import requests 
import zipfile   
from typing import Optional, List, Dict, Tuple
import json
from datetime import datetime
import warnings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from rapidfuzz import fuzz, process
import joblib
import functools
import folium
from streamlit_folium import st_folium
from PIL import Image
import io
import urllib.parse
import streamlit.components.v1 as components
import shutil

# Global headers for iNaturalist API requests
INAT_HEADERS = {
    'User-Agent': 'BotanicalWorkbench/1.0 (contact: daniel.cahen.substance@gmail.com)'
}

# Configure page with botanical theme
st.set_page_config(
    page_title="Botanical ID Workbench: South Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from the external styles.css file
st.html("styles.css")

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants for Data Loading ---
EFLORA_ZIP_URL = "https://ipt.sanbi.org.za/archive.do?r=flora_descriptions&v=1.42"
DATA_DIR = "data"
REQUIRED_FILES = ["taxon.txt", "description.txt", "vernacularname.txt"]

# iNaturalist license codes and their meanings
INAT_LICENSE_MAP = {
    'cc-by': 'CC BY 4.0',
    'cc-by-sa': 'CC BY-SA 4.0',
    'cc-by-nd': 'CC BY-ND 4.0',
    'cc-by-nc': 'CC BY-NC 4.0',
    'cc-by-nc-nd': 'CC BY-NC-ND 4.0',
    'cc-by-nc-sa': 'CC BY-NC-SA 4.0',
    'cc0': 'CC0 1.0',
    'pd': 'Public Domain'
}

ALLOWED_INAT_LICENSES = ['cc-by', 'cc-by-sa', 'cc0', 'pd']

# Initialize session state
def init_session_state():
    defaults = {
        'species_data': None,
        'selected_species': {},
        'eflora_data': None,
        'analysis_data': None,
        'all_records': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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

@st.cache_data(ttl=60*60*24*7) # Cache data for one week
def load_eflora_data() -> Optional[pd.DataFrame]:
    """
    Ensures e-Flora data is available locally, downloading and extracting if necessary.
    Then, it processes and merges the data into a single DataFrame.
    The final DataFrame is cached for performance.
    """
    # 1. Ensure the local data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Check if all required files are already present
    local_files_exist = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)

    # 3. If files are missing, download and extract them
    if not local_files_exist:
        zip_path = os.path.join(DATA_DIR, "eflora_data.zip")
        with st.spinner(f"Downloading e-Flora data from SANBI (~16MB)... This is a one-time setup."):
            try:
                # NEW: Define headers to mimic a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Download the ZIP file
                with requests.get(EFLORA_ZIP_URL, stream=True, headers=headers) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Extract the ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)

                # Clean up the downloaded ZIP file
                os.remove(zip_path)
                st.success("e-Flora data downloaded and prepared successfully!")

            except Exception as e:
                st.error(f"Failed to download or process e-Flora data: {e}")
                # Clean up failed download if it exists
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return None

    # 4. Load, process, and merge the data (this part is your existing logic)
    try:
        st.info("Loading e-Flora database from local files...")
        
        # Load the three main files (as before)
        taxa_df = pd.read_csv(os.path.join(DATA_DIR, 'taxon.txt'), sep='\t', header=0, 
                              usecols=['id', 'scientificName'], dtype={'id': str})
        
        desc_df = pd.read_csv(os.path.join(DATA_DIR, 'description.txt'), sep='\t', header=0, 
                              usecols=['id', 'description', 'type'], dtype={'id': str})
        
        vernacular_df = pd.read_csv(os.path.join(DATA_DIR, 'vernacularname.txt'), sep='\t', header=0, 
                                    usecols=['id', 'vernacularName'], dtype={'id': str})
        
        # --- Your existing merging and processing logic (unchanged) ---
        for df in [taxa_df, desc_df]:
            df.rename(columns={'id': 'taxonID'}, inplace=True)
        vernacular_df.rename(columns={'id': 'taxonID'}, inplace=True)
        
        taxa_df['cleanScientificName'] = taxa_df['scientificName'].apply(
            lambda x: ' '.join(str(x).split()[:2]) if pd.notna(x) else ''
        )
        
        desc_agg = desc_df.groupby('taxonID').apply(
            lambda x: x.set_index('type')['description'].to_dict()
        ).reset_index(name='descriptions')
        
        vernacular_agg = vernacular_df.groupby('taxonID')['vernacularName'].apply(
            lambda x: list(set(str(n).strip() for n in x.dropna() if pd.notna(n) and str(n).strip()))
        ).reset_index()
        
        eflora_data = pd.merge(taxa_df, desc_agg, on='taxonID', how='left')
        eflora_data = pd.merge(eflora_data, vernacular_agg, on='taxonID', how='left')
        
        eflora_data.set_index('cleanScientificName', inplace=True)
        eflora_data = eflora_data[~eflora_data.index.duplicated(keep='first')]
        eflora_data = eflora_data[eflora_data.index != '']
        # --- End of your existing logic ---

        st.success(f"Loaded {len(eflora_data)} taxa from e-Flora database.")
        return eflora_data
        
    except Exception as e:
        st.error(f"Failed to load e-Flora data from local files: {e}")
        return None

def format_species_name(name):
    if not name:
        return None
    parts = name.split()
    return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else name

@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def safe_gbif_backbone(name, kingdom='Plantae'):
    """Cached and retried GBIF backbone lookup."""
    return gbif_species.name_backbone(name=name, kingdom=kingdom, verbose=False)

@st.cache_data
def get_species_details(species_name: str, limit: int = 5, fetch_hierarchy: bool = True) -> Tuple[List[Dict], Optional[int], List[Dict]]:
    """
    Fetch iNat photos + taxonomic hierarchy.
    Returns: (photos, taxon_id, ancestors)
    """
    try:
        encoded_name = urllib.parse.quote(species_name)
        search_url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={encoded_name}&per_page=1"
        response = requests.get(search_url, headers=INAT_HEADERS, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"API error for taxon search ({response.status_code}): {response.text[:200]}")
            return [], None, []
            
        data = response.json()
        
        if not data.get('results'):
            return [], None, []
            
        taxon = data['results'][0]
        if taxon.get('rank') != 'species':
            for result in data.get('results', []):
                if result.get('rank') == 'species':
                    taxon = result
                    break
        
        taxon_id = taxon['id']
        photos = []
        
        # Existing default_photo logic (unchanged)
        default_photo = taxon.get('default_photo')
        if default_photo:
            photo_url = default_photo.get('medium_url') or default_photo.get('square_url')
            if photo_url:
                attribution = default_photo.get('attribution', '(c) Unknown photographer')
                photographer = attribution.split('(c)')[-1].split(',')[0].strip() if '(c)' in attribution else 'Unknown photographer'
                
                license_code = default_photo.get('license_code', '')
                if license_code in ALLOWED_INAT_LICENSES:
                    license_name = INAT_LICENSE_MAP.get(license_code, 'Unknown')
                    photos.append({
                        'url': photo_url,
                        'photographer': photographer,
                        'license': license_name,
                        'caption': f"© {photographer} · {license_name}"
                    })
        
        # Existing observations logic (unchanged)
        obs_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&photos=true&per_page={limit}&order_by=votes&order=desc"
        obs_response = requests.get(obs_url, headers=INAT_HEADERS, timeout=10)
        
        if obs_response.status_code != 200:
            logger.warning(f"API error for observations ({obs_response.status_code}): {obs_response.text[:200]}")
        
        obs_data = obs_response.json()
        
        for obs in obs_data.get('results', [])[:limit]:
            obs_user = obs.get('user', {})
            photographer_name = obs_user.get('name') or obs_user.get('login', 'Unknown')
            
            for photo in obs.get('photos', [])[:1]:
                photo_url = photo.get('url', '').replace('square', 'medium')
                if not photo_url:
                    photo_url = photo.get('medium_url') or photo.get('square_url')
                
                if photo_url and photo_url not in [p['url'] for p in photos]:
                    photo_attribution = photo.get('attribution', '')
                    photo_photographer = photo_attribution.split('(c)')[-1].split(',')[0].strip() if '(c)' in photo_attribution else photographer_name
                    
                    license_code = photo.get('license_code', '')
                    if license_code in ALLOWED_INAT_LICENSES:
                        license_name = INAT_LICENSE_MAP.get(license_code, 'Unknown')
                        
                        photos.append({
                            'url': photo_url,
                            'photographer': photo_photographer,
                            'license': license_name,
                            'caption': f"© {photo_photographer} · {license_name}"
                        })
                
                if len(photos) >= limit:
                    break
            
            if len(photos) >= limit:
                break
        
        # Fetch ancestors only if requested
        ancestors = []
        if fetch_hierarchy:
            taxon_url = f"https://api.inaturalist.org/v1/taxa/{taxon_id}"
            taxon_response = requests.get(taxon_url, headers=INAT_HEADERS, timeout=10)
            if taxon_response.status_code == 200:
                taxon_data = taxon_response.json().get('results', [{}])[0]
                raw_ancestors = taxon_data.get('ancestors', [])
                ancestors = sorted(
                    [anc for anc in raw_ancestors if anc.get('rank')],
                    key=lambda x: x.get('rank_level', 0), reverse=True  # Kingdom first
                )
                ancestors = [{'rank': anc['rank'], 'name': anc['name'], 'id': anc.get('id')} for anc in ancestors]  # Include ID for links
        
        return photos, taxon_id, ancestors
        
    except Exception as e:
        logger.error(f"Error fetching iNat details for {species_name}: {e}")
        return [], None, []

def filter_species_by_rank(species_data: List[Dict], rank: str, name: str) -> List[Dict]:
    """Filter species where hierarchy has matching rank and name (case-insensitive)."""
    return [sp for sp in species_data if any(
        anc['rank'] == rank and anc['name'].lower() == name.lower()
        for anc in sp.get('hierarchy', [])
    )]

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
                    st.warning("Search truncated at 100,000 records for performance.")
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
                        'taxon_key': species_key, 'status_flag': status_flag, 'records': []
                    }
                species_dict[species_name]['count'] += 1
                species_dict[species_name]['records'].append(record)
        
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
            return False, f"Species {clean_name} not found in local database"
    
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
            extracted_data.append(f"**Common Names:** {', '.join(vernacular_names[:5])}")

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
                if desc_text and len(desc_text) > 10:
                    extracted_data.append(f"**{section}:**\n{desc_text}")
                    sections_added += 1
                    if sections_added >= 4:
                        break
        
        # If no priority sections found, add any available sections
        if sections_added == 0:
            for section, desc in descriptions.items():
                if pd.notna(desc):
                    desc_text = str(desc).strip()
                    if desc_text and len(desc_text) > 10:
                        extracted_data.append(f"**{section}:**\n{desc_text}")
                        sections_added += 1
                        if sections_added >= 2:
                            break
        
        if sections_added == 0:
            return False, f"No detailed descriptions available for {clean_name}"

        # Add e-Flora citation
        extracted_data.append(
            "\n**Citation:** e-Flora of South Africa. v1.42. 2023. South African National Biodiversity Institute. http://ipt.sanbi.org.za/iptsanbi/resource?r=flora_descriptions&v=1.42"
        )
        extracted_data.append(
            "**License:** CC-BY 4.0"
        )
            
        return True, "\n\n".join(extracted_data)
        
    except Exception as e:
        st.error(f"Error processing {clean_name}: {e}")
        return False, f"Error retrieving data for {clean_name}"

def create_species_map(records, species_list, center_lat, center_lon):
    """Create an interactive map with species observations."""
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add center point
    folium.Marker(
        [center_lat, center_lon],
        popup="Search Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Professional color palette
    colors = ['#2d5016', '#4a7c24', '#6b8e23', '#8fbc8f', '#556b2f', 
              '#697565', '#8b7355', '#6b4423', '#7c5e4c', '#a0826d']
    
    # Add species points
    for i, species in enumerate(species_list[:10]):
        color = colors[i % len(colors)]
        species_records = [r for r in records if r.get('species') == species['name']]
        
        for record in species_records[:50]:
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
                    weight=2,
                    opacity=0.8,
                    fillOpacity=0.6
                ).add_to(m)
    
    # Create legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 60px; 
                left: 50px; 
                width: 280px; 
                background-color: rgba(255, 255, 255, 0.98); 
                border: 1px solid #d0d0d0; 
                border-radius: 4px;
                z-index: 9999; 
                font-size: 12px; 
                padding: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <div style="color: #1a3d07; font-weight: 600; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px;">
        Species Legend
    </div>
    '''
    
    for i, species in enumerate(species_list[:10]):
        color = colors[i % len(colors)]
        display_name = species['name'] if len(species['name']) <= 25 else species['name'][:22] + '...'
        legend_html += f'''
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <div style="background: {color}; 
                        width: 14px; 
                        height: 14px; 
                        border-radius: 50%; 
                        display: inline-block; 
                        margin-right: 8px; 
                        border: 1px solid #333;
                        flex-shrink: 0;">
            </div>
            <span style="color: #1a1a1a; font-size: 11px; line-height: 1.3;">{display_name}</span>
        </div>
        '''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def add_footer():
    """Adds a professional footer with creator information."""
    st.markdown(f"""
    <div class="footer">
        Created by Daniel Cahen | © {datetime.now().year} | MIT License | 
        <a href="https://github.com" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

def create_copy_button(copy_text, label):
    """Create a copy button component for export previews."""
    escaped_text = copy_text.replace('\\', '\\\\').replace('`', '\\`')
    components.html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid var(--border-color);">
        <span style="font-weight: 500;">Preview {label}</span>
        <button id="copyBtn" style="background-color: #2d5016; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px;">Copy {label}</button>
    </div>
    <script>
    document.getElementById('copyBtn').addEventListener('click', function() {{
        navigator.clipboard.writeText(`{escaped_text}`).then(() => {{
            alert('Copied to clipboard!');
        }});
    }});
    </script>
    """, height=80)

# Main Streamlit App
def main():
    st.title("🌿 Botanical ID Workbench: South Africa")
    st.markdown("*Advanced species identification using GBIF data and local flora databases*")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Search Parameters")
        
        # Location settings
        st.subheader("📍 Location")
        
        # Use session state to store the actual coordinates being used
        if 'current_latitude' not in st.session_state:
            st.session_state.current_latitude = -33.92
        if 'current_longitude' not in st.session_state:
            st.session_state.current_longitude = 18.42
            
        # 1. New field for pasting coordinates
        coord_paste = st.text_input(
            "Paste Coordinates (Lat, Lon)", 
            value=f"{st.session_state.current_latitude}, {st.session_state.current_longitude}",
            help="Paste a single string like '-26.8974527778, 27.476825' here."
        )
        
        # 2. Button to trigger parsing
        if st.button("Apply Coordinates", key="apply_coords", use_container_width=True):
            try:
                # Attempt to parse the pasted string
                lat_str, lon_str = coord_paste.split(',')
                new_lat = float(lat_str.strip())
                new_lon = float(lon_str.strip())
                
                # Update session state
                st.session_state.current_latitude = new_lat
                st.session_state.current_longitude = new_lon
                st.success(f"Coordinates updated to: {new_lat}, {new_lon}")
                # Rerun to update the map and search inputs immediately
                st.rerun() 
                
            except ValueError:
                st.error("Invalid coordinate format. Please use 'Latitude, Longitude' (e.g., -26.89, 27.47).")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

        # 3. Use the session state values for the rest of the app
        latitude = st.session_state.current_latitude
        longitude = st.session_state.current_longitude
        
        # Display the actual values being used (optional, but good for confirmation)
        st.markdown(f"**Current Lat:** `{latitude:.6f}`")
        st.markdown(f"**Current Lon:** `{longitude:.6f}`")
        
        radius_km = st.slider("Search Radius (km)", 1, 200, 25,
                              help="Radius around center point to search")
        
        # Search parameters  
        st.subheader("🌱 Taxon Search")
        taxon_name = st.text_input("Taxon Name", value="Protea",
                                   help="Scientific name of taxon to search (genus or higher)")
        
        st.subheader("🔍 Filter by Rank")
        rank_options = ["tribe", "subtribe", "subfamily", "genus", "species_group"]  # Common intermediates
        selected_rank = st.selectbox("Select Rank", rank_options)
        rank_name = st.text_input("Rank Name (e.g., Atripliceae)", placeholder="Leave blank to disable")

        # Options
        st.subheader("⚙️ Options")
        include_images = st.checkbox("Include Images", value=True,
                                     help="Fetch images from iNaturalist (slower but more informative)")
        include_hierarchy = st.checkbox("Include Taxonomic Hierarchy", value=True,
                                        help="Fetch hierarchy from iNaturalist (adds context but slower)")
        export_format = st.radio(
            "Export Format", 
            ["JSON", "Markdown"], # Set JSON as the default (first in list)
            horizontal=True,
            help="Format for downloading results (JSON is best for LLM analysis)"
        )
        
        # Actions
        if st.button("🔍 Search GBIF", type="primary", use_container_width=True):
            with st.spinner("Searching GBIF database..."):
                # Load e-Flora data if not already loaded
                if st.session_state.eflora_data is None:
                    st.session_state.eflora_data = load_eflora_data()
                
                if st.session_state.eflora_data is not None:
                    species_data, all_records = get_species_list_from_gbif(
                        latitude, longitude, radius_km, taxon_name
                    )
                    st.session_state.species_data = species_data
                    st.session_state.all_records = all_records
                    st.session_state.selected_species = {}
                    if 'species_selector' in st.session_state:
                        del st.session_state.species_selector
                    if 'analysis_data' in st.session_state:
                        del st.session_state.analysis_data
                    st.rerun()
                else:
                    st.error("Cannot proceed without e-Flora data. Please check your data files.")
        
        # Data diagnostics section
        with st.expander("🔧 Data Diagnostics"):
            if st.button("Test e-Flora Data", use_container_width=True):
                eflora_data = load_eflora_data()
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
                else:
                    st.error("Failed to load e-Flora data")
            
            st.markdown("**Required Files:**")
            st.markdown("""
            - `data/taxon.txt` - columns: 'id', 'scientificName'
            - `data/description.txt` - columns: 'id', 'description', 'type'
            - `data/vernacularname.txt` - columns: 'id', 'vernacularName'
            """)
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            for cache_dir in ['gbif_cache']:
                if os.path.exists(cache_dir):
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
            radius=radius_km * 1000,
            popup=f"Search radius: {radius_km} km",
            color="#2d5016",
            fill=True,
            fillColor="#a4b494",
            fillOpacity=0.2,
            weight=2
        ).add_to(preview_map)
        st_folium(preview_map, height=400, width=700)
        
    else:
        # Display search results
        st.success(f"Found {len(st.session_state.species_data)} species in the search area")

        # Apply rank filter if specified
        filtered_data = st.session_state.species_data
        if rank_name:  # From sidebar
            # Fetch hierarchies for filtering (always, as filter requires it)
            with st.spinner("Fetching taxonomic hierarchies for rank filtering..."):
                for sp in st.session_state.species_data:
                    _, _, ancestors = get_species_details(sp['name'], fetch_hierarchy=True)
                    sp['hierarchy'] = ancestors
            filtered_data = filter_species_by_rank(st.session_state.species_data, selected_rank, rank_name)
            st.info(f"Filtered to {len(filtered_data)} species in {selected_rank}: {rank_name}")
            if not include_hierarchy:
                st.warning("Hierarchy fetched temporarily for rank filtering. It will be included in analysis/export unless disabled.")
        
        # Prepare species options for multiselect based on filtered data
        species_list_limited = filtered_data[:50]
        species_options = [
            f"{sp['name']} - {sp['family']} ({sp['count']} records){sp.get('status_flag', '')}"
            for sp in species_list_limited
        ]
        st.session_state.species_options = species_options
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Species List", "🗺️ Map View", "📄 Export"])
        
        with tab1:
            st.subheader("Species Found in Search Area")
            
            # Create DataFrame for display
            df = pd.DataFrame(filtered_data)
            df = df[df['name'].str.strip() != '']
            df_display = df[['name', 'family', 'count']].copy()
            df_display.columns = ['Species', 'Family', 'Records']
            df_display = df_display.reset_index(drop=True)
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True, height=300)
            
            # Selection controls
            st.subheader("Select Species for Detailed Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Select All", use_container_width=True):
                    st.session_state.species_selector = st.session_state.species_options[:]
                    st.rerun()
            with col2:
                if st.button("Clear Selection", use_container_width=True):
                    st.session_state.species_selector = []
                    st.session_state.selected_species = {}
                    if 'analysis_data' in st.session_state:
                        del st.session_state.analysis_data
                    st.rerun()
            with col3:
                if st.button("Top 10", use_container_width=True):
                    st.session_state.species_selector = st.session_state.species_options[:10]
                    st.rerun()
            
            # Multiselect for species selection
            selected_labels = st.multiselect(
                "Select species:",
                st.session_state.species_options,
                key="species_selector",
                format_func=lambda x: x
            )
            
            # Map selected labels back to species names
            selected_names = []
            for label in selected_labels:
                name = label.split(' - ')[0]
                selected_names.append(name)
            
            st.session_state.selected_species = {name: True for name in selected_names}
            
            # Process selected species
            if len(selected_names) > 0:
                st.info(f"Selected {len(selected_names)} species for analysis")
                
                if st.button("📊 Generate Detailed Analysis", type="primary", use_container_width=True):
                    selected_species_data = [
                        sp for sp in filtered_data 
                        if sp['name'] in selected_names
                    ]
                    # Augment with details (including hierarchy only if option enabled)
                    augmented_data = []
                    with st.spinner(f"Fetching details for {len(selected_species_data)} species..."):
                        for sp in selected_species_data:
                            photos, taxon_id, ancestors = get_species_details(
                                sp['name'], 
                                fetch_hierarchy=include_hierarchy
                            )  # Respect option here
                            sp_copy = sp.copy()
                            sp_copy['hierarchy'] = ancestors if include_hierarchy else []
                            sp_copy['photos'] = photos  # Always store if images enabled, but fetch anyway
                            sp_copy['taxon_id'] = taxon_id
                            augmented_data.append(sp_copy)
                    st.session_state.analysis_data = augmented_data
                    st.rerun()
                
                # Display detailed analysis if generated
                if 'analysis_data' in st.session_state:
                    # Ensure analysis_data is a valid list
                    if isinstance(st.session_state.analysis_data, list) and len(st.session_state.analysis_data) > 0:
                        st.subheader("🔍 Detailed Species Information")
                        
                        # Pagination setup
                        page_size = 10
                        total_pages = max(1, math.ceil(len(st.session_state.analysis_data) / page_size))
                        
                        # Create slider only if we have multiple pages
                        if total_pages > 1:
                            page = st.slider("Page", 1, total_pages, 1, key="detail_page")
                            st.info(f"Showing page {page} of {total_pages}")
                        else:
                            page = 1
                            st.info(f"Showing all {len(st.session_state.analysis_data)} species")
                        
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(st.session_state.analysis_data))
                        paginated_species = st.session_state.analysis_data[start_idx:end_idx]
                        
                        for species in paginated_species:
                            with st.expander(f"📋 {species['name']} - {species['family']} ({species['count']} records)", expanded=True):
                                
                                # Use pre-fetched details if available
                                if 'hierarchy' in species and species['hierarchy']:
                                    ancestors = species['hierarchy']
                                    photos = species.get('photos', [])
                                    taxon_id = species.get('taxon_id')
                                else:
                                    # Fallback fetch if needed (e.g., hierarchy disabled but now enabled)
                                    photos, taxon_id, ancestors = get_species_details(
                                        species['name'], 
                                        fetch_hierarchy=include_hierarchy
                                    )

                                if include_images:
                                    col1, col2 = st.columns([3, 1])
                                else:
                                    col1 = st.columns([1])[0]
                                
                                with col1:
                                    # Get local e-Flora description
                                    success, description = get_local_eflora_description(
                                        species['name'], st.session_state.eflora_data
                                    )
                                    
                                    if success:
                                        st.markdown(description)
                                    else:
                                        st.warning(f"No local description available")
                                        st.markdown(f"**Scientific Name:** {species['name']}")
                                        st.markdown(f"**Family:** {species['family']}")
                                        st.markdown(f"**GBIF Records:** {species['count']}")

                                    # Display hierarchy only if option enabled and available
                                    if include_hierarchy and ancestors:
                                        hierarchy_links = []
                                        for anc in ancestors:
                                            inat_link = f"https://www.inaturalist.org/taxa/{anc.get('id', '')}" if anc.get('id') else "#"
                                            hierarchy_links.append(f"[{anc['name']} ({anc['rank']})]({inat_link})")
                                        hierarchy_str = " > ".join(hierarchy_links)
                                        st.markdown(f"**Taxonomic Hierarchy:** {hierarchy_str}")
                                    elif include_hierarchy:
                                        st.info("Taxonomic hierarchy could not be retrieved from iNaturalist.")

                                if include_images:
                                    with col2:
                                        # Display iNaturalist images (using the 'photos' variable)
                                        if photos:
                                            st.markdown("**Photos from iNaturalist:**")
                                            for img_data in photos[:3]:  # Limit to 3 images
                                                try:
                                                    response = requests.get(img_data['url'], headers=INAT_HEADERS, timeout=10)
                                                    if response.status_code != 200:
                                                        st.warning(f"Failed to load image (HTTP {response.status_code})")
                                                        continue
                                                    img = Image.open(io.BytesIO(response.content))
                                                    st.image(img, caption=img_data['caption'], use_container_width=True)
                                                    st.markdown(" ")
                                                except Exception as e:
                                                    st.warning(f"Failed to load image: {str(e)[:100]}")
                                            
                                            if taxon_id:
                                                inat_link = f"https://www.inaturalist.org/taxa/{taxon_id}"
                                                st.markdown(f"[View on iNaturalist ↗]({inat_link})")
                                        else:
                                            st.info("No photos available")
                    else:
                        # analysis_data exists but is empty or invalid
                        st.warning("No species data available for analysis. Please select species and click 'Generate Detailed Analysis'.")
        
        with tab2:
            st.subheader("🗺️ Species Distribution Map")
            if hasattr(st.session_state, 'all_records') and st.session_state.all_records:
                species_map = create_species_map(
                    st.session_state.all_records,
                    st.session_state.species_data,
                    latitude,
                    longitude
                )
                st_folium(species_map, height=600, width=700)
                st.info("🎯 Red marker = search center | Colored dots = species observations")
            else:
                st.warning("Map data not available")
        
        with tab3:
            st.subheader("📄 Export Data")
            
            # Use analysis_data if available, otherwise current selection
            if 'analysis_data' in st.session_state and st.session_state.analysis_data and len(st.session_state.analysis_data) > 0:
                selected_species_data = st.session_state.analysis_data
            else:
                selected_species_data = [
                    sp for sp in filtered_data 
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
                            "hierarchy": species.get('hierarchy', []) if include_hierarchy else [],
                            "description": description if success else "No description available"
                        })
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"botanical_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    with st.expander("Preview JSON", expanded=True):
                        create_copy_button(json_str, "JSON")
                        st.json(export_data)
                
                else:
                    # Markdown export
                    markdown_parts = [
                        f"# Botanical Identification Report",
                        f"",
                        f"**Location:** {latitude}, {longitude}",
                        f"**Search Radius:** {radius_km} km",
                        f"**Target Taxon:** {taxon_name}",
                        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"**Selected Species:** {len(selected_species_data)}",
                        f"",
                        f"## Species Details",
                        f""
                    ]
                    
                    for species in selected_species_data:
                        success, description = get_local_eflora_description(
                            species['name'], st.session_state.eflora_data
                        )
                        
                        # Add hierarchy to MD only if option enabled
                        hierarchy_md = ""
                        if include_hierarchy and species.get('hierarchy'):
                            hierarchy_list = "\n".join([f"- [{anc['name']} ({anc['rank']})](https://www.inaturalist.org/taxa/{anc.get('id', '')})" for anc in species['hierarchy']])
                            hierarchy_md = f"**Taxonomic Hierarchy:**\n{hierarchy_list}\n\n"

                        markdown_parts.extend([
                            f"### {species['name']}",
                            f"**Family:** {species['family']}",
                            f"**GBIF Records:** {species['count']}",
                            hierarchy_md,
                            description if success else "No description available.",
                            f"",
                            "---",
                            f""
                        ])
                    
                    markdown_str = "\n".join(markdown_parts)
                    
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=markdown_str,
                        file_name=f"botanical_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    with st.expander("Preview Markdown", expanded=True):
                        create_copy_button(markdown_str, "Markdown")
                        st.markdown(markdown_str)
            else:
                st.warning("Please select species in the Species List tab first")

    # Add footer
    add_footer()

if __name__ == "__main__":
    main()