import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import concurrent.futures
from typing import Optional, List, Dict, Tuple
import asyncio
import aiohttp
import json
import os
from datetime import datetime
import pygbif.species as gbif_species
import pygbif.occurrences as gbif_occ
import math
import time
import joblib
from rapidfuzz import fuzz, process
import logging
import requests
import urllib.parse
import streamlit.components.v1 as components
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "prepared_data"
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "eflora_processed.parquet")
VERSION_FILE = os.path.join(DATA_DIR, "data_version.json")
MAX_CONCURRENT_REQUESTS = 10
DATA_URL = "https://github.com/gouania/botanical-id-workbench/releases/download/v1.0.0-data/eflora_processed.parquet"

# Global headers for iNaturalist API requests
INAT_HEADERS = {'User-Agent': 'BotanicalWorkbench/1.0'}

# iNaturalist license mapping
INAT_LICENSE_MAP = {
    'cc-by': 'CC BY 4.0', 'cc-by-sa': 'CC BY-SA 4.0', 'cc-by-nd': 'CC BY-ND 4.0',
    'cc-by-nc': 'CC BY-NC 4.0', 'cc-by-nc-nd': 'CC BY-NC-ND 4.0', 'cc-by-nc-sa': 'CC BY-NC-SA 4.0',
    'cc0': 'CC0 1.0', 'pd': 'Public Domain'
}

def download_data_if_needed():
    """Checks for the data file and downloads it from the remote URL if it's missing."""
    if not os.path.exists(PROCESSED_DATA_FILE):
        # Use st.spinner for a better user experience during download
        with st.spinner("First-time setup: Downloading prepared e-Flora data... (This may take a moment)"):
            logger.info(f"Downloading data from {DATA_URL}")
            os.makedirs(DATA_DIR, exist_ok=True)
            try:
                response = requests.get(DATA_URL)
                response.raise_for_status()  # Raise an exception for bad status codes
                with open(PROCESSED_DATA_FILE, "wb") as f:
                    f.write(response.content)
                logger.info("Data download successful.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download data file. Please check the URL and your connection. Error: {e}")
                return False
    return True

st.set_page_config(
    page_title="Botanical ID Workbench: South Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
def init_session_state():
    defaults = {
        'species_data': None, 'selected_species': {}, 'eflora_data': None,
        'analysis_data': None, 'all_records': None, 'page': 'search',
        'filter_settings': {}, 'map_cluster': True, 'rank_filter_enabled': False,
        'discovered_ranks': None, 'rank_filter_settings': {}, 'selected_species_set': set()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

@st.cache_data(ttl=3600*24)
def load_prepared_data() -> Optional[pd.DataFrame]:
    """
    Load pre-processed e-Flora data. On first run in a deployed environment,
    it will download the data from a remote URL.
    """
    # Run the download check first
    if not download_data_if_needed():
        return None # Stop if download fails

    # The rest of the function is the same as before
    try:
        data = pd.read_parquet(PROCESSED_DATA_FILE)
        
        if 'gbifUsageKey' not in data.columns:
            st.error("Data file is in an old format. The hosted file needs to be regenerated.")
            return None

        data.dropna(subset=['gbifUsageKey'], inplace=True)
        data['gbifUsageKey'] = data['gbifUsageKey'].astype(int)
        data.set_index('gbifUsageKey', inplace=True)
        data = data[~data.index.duplicated(keep='first')]
        
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- ASYNCHRONOUS DATA FETCHING ---

async def fetch_hierarchy_async(session, species_name: str) -> Tuple[str, List[Dict]]:
    """Async helper to fetch just the hierarchy for a single species."""
    try:
        search_url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={species_name}&per_page=1"
        async with session.get(search_url, headers=INAT_HEADERS) as response:
            if response.status != 200: return species_name, []
            data = await response.json()
            if not data.get('results'): return species_name, []
            
            taxon_id = data['results'][0]['id']
            taxon_url = f"https://api.inaturalist.org/v1/taxa/{taxon_id}"
            async with session.get(taxon_url, headers=INAT_HEADERS) as taxon_response:
                if taxon_response.status == 200:
                    taxon_data = await taxon_response.json()
                    ancestors = taxon_data.get('results', [{}])[0].get('ancestors', [])
                    return species_name, [{'rank': a['rank'], 'name': a['name']} for a in ancestors if a.get('rank')]
    except Exception:
        return species_name, []
    return species_name, []

async def fetch_all_hierarchies_parallel(species_list: List[Dict]) -> List[Dict]:
    """Proactively fetches all hierarchies for the species list after GBIF search."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_hierarchy_async(session, sp['name']) for sp in species_list]
        results = await asyncio.gather(*tasks)
        
        hierarchy_map = {name: hierarchy for name, hierarchy in results}
        
        for species in species_list:
            species['hierarchy'] = hierarchy_map.get(species['name'], [])
    return species_list

async def fetch_species_details_async(session, species_name: str, limit: int = 5):
    """Async version of full species details fetching for the analysis phase."""
    try:
        search_url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={species_name}&per_page=1"
        async with session.get(search_url, headers=INAT_HEADERS) as response:
            if response.status != 200: return species_name, [], None
            data = await response.json()
            if not data.get('results'): return species_name, [], None
            
            taxon = data['results'][0]
            taxon_id = taxon['id']
            
            photos = []
            default_photo = taxon.get('default_photo')
            if default_photo and default_photo.get('medium_url'):
                attribution = default_photo.get('attribution', '')
                match = re.search(r'\(c\)\s*([^,]+)', attribution)
                photographer = match.group(1).strip() if match else 'Unknown'
                photos.append({'url': default_photo['medium_url'], 'photographer': photographer, 'license': default_photo.get('license_code')})

            obs_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&photos=true&per_page={limit}"
            async with session.get(obs_url, headers=INAT_HEADERS) as obs_response:
                if obs_response.status == 200:
                    obs_data = await obs_response.json()
                    for obs in obs_data.get('results', []):
                        if len(photos) >= limit: break
                        user = obs.get('user', {})
                        photographer_name = user.get('name') or user.get('login', 'Unknown')
                        for photo in obs.get('photos', [])[:1]:
                            photo_url = photo.get('url', '').replace('square', 'medium')
                            if photo_url and photo_url not in [p['url'] for p in photos]:
                                photos.append({'url': photo_url, 'photographer': photographer_name, 'license': photo.get('license_code')})
            
            return species_name, photos, taxon_id
            
    except Exception as e:
        logger.error(f"Error fetching details for {species_name}: {e}")
        return species_name, [], None

def run_async_task(task):
    """Helper to run an async task in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(task)
    finally:
        loop.close()

# --- UI & Core Logic ---

def get_available_ranks_and_values(species_data: List[Dict]) -> Dict[str, set]:
    """Discover available taxonomic ranks from pre-fetched data."""
    ranks_values = {}
    for sp in species_data:
        for anc in sp.get('hierarchy', []):
            rank, name = anc.get('rank'), anc.get('name')
            if rank and name:
                if rank not in ranks_values: ranks_values[rank] = set()
                ranks_values[rank].add(name)
    return ranks_values

def filter_species_by_rank_optimized(species_data: List[Dict], rank: str, name: str) -> List[Dict]:
    """Filter species list based on a pre-fetched taxonomic rank."""
    return [
        sp for sp in species_data 
        if any(anc.get('rank') == rank and anc.get('name') == name for anc in sp.get('hierarchy', []))
    ]

def show_search_page():
    """Main search interface."""
    if st.session_state.eflora_data is None:
        st.session_state.eflora_data = load_prepared_data()
        if st.session_state.eflora_data is None: return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📍 Search Parameters")
        coord_input = st.text_area("Enter Coordinates", value="-33.92, 18.42", height=60)
        try:
            lat_str, lon_str = coord_input.strip().split(',')
            latitude, longitude = float(lat_str.strip()), float(lon_str.strip())
        except:
            st.error("Invalid format. Use: latitude, longitude")
            return
        
        radius_km = st.slider("Search Radius (km)", 1, 100, 25)
        taxon_name = st.text_input("Taxon Name", value="Protea")
        
        if st.button("🔍 Search GBIF", type="primary", width='stretch'):
            with st.spinner("Searching GBIF and pre-fetching taxonomic data..."):
                species_data, all_records = search_gbif_cached(latitude, longitude, radius_km, taxon_name)
                st.session_state.species_data = species_data
                st.session_state.all_records = all_records
                st.session_state.analysis_data = None
                st.session_state.discovered_ranks = None
                st.session_state.selected_species_set = set()
                st.rerun()
    
    with col2:
        st.subheader("📍 Search Area Preview")
        preview_map = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker([latitude, longitude], popup="Search Center", icon=folium.Icon(color='red')).add_to(preview_map)
        folium.Circle(location=[latitude, longitude], radius=radius_km * 1000, color="#2d5016", fill=True, fillOpacity=0.2).add_to(preview_map)
        st_folium(preview_map, height=400)
    
    if st.session_state.species_data:
        st.divider()
        show_results_section()

def show_results_section():
    """Display and manage search results."""
    st.subheader(f"🎯 Found {len(st.session_state.species_data)} species")
    
    filtered_data = st.session_state.species_data
    
    with st.expander("🔧 Refine Results", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("General Filters")
            min_records = st.number_input("Minimum records", min_value=1, value=1)
            if min_records > 1:
                filtered_data = [sp for sp in filtered_data if sp['count'] >= min_records]
        with col2:
            render_advanced_filters()

    if st.session_state.rank_filter_enabled and st.session_state.rank_filter_settings.get('value'):
        settings = st.session_state.rank_filter_settings
        filtered_data = filter_species_by_rank_optimized(filtered_data, settings['rank'], settings['value'])
        st.info(f"Filtered to {len(filtered_data)} species matching {settings['value']} ({settings['rank']}).")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Table View", "🗺️ Map View", "🔬 Analysis", "📄 Export"])
    
    with tab1:
        df = pd.DataFrame(filtered_data)
        b_col1, b_col2, _ = st.columns([1,1,3])
        if b_col1.button("Select All", width='stretch'):
            st.session_state.selected_species_set = set(df['name'].tolist())
            st.rerun()
        if b_col2.button("Deselect All", width='stretch'):
            st.session_state.selected_species_set = set()
            st.rerun()

        df_display = df[['name', 'family', 'count']].copy()
        df_display.columns = ['Species', 'Family', 'Records']
        df_display.insert(0, 'Select', df_display['Species'].apply(lambda x: x in st.session_state.selected_species_set))
        
        edited_df = st.data_editor(df_display, hide_index=True, column_config={"Select": st.column_config.CheckboxColumn("Select")})
        
        st.session_state.selected_species_set = set(edited_df[edited_df['Select']]['Species'].tolist())
        selected_species = list(st.session_state.selected_species_set)
        
        if selected_species:
            st.info(f"Selected {len(selected_species)} species")
            if st.button("Analyze Selected Species", type="primary", width='stretch'):
                analyze_selected_species(selected_species, filtered_data)
    
    with tab2:
        use_clustering = st.checkbox("Use Clustering", value=True)
        if st.session_state.all_records:
            species_map = create_clustered_map(st.session_state.all_records, filtered_data[:20], st.session_state.get('current_latitude', -33.92), st.session_state.get('current_longitude', 18.42), use_clustering)
            st_folium(species_map, height=600, use_container_width=True) # Note: st_folium doesn't use 'width'
    
    with tab3:
        if 'analysis_data' in st.session_state and st.session_state.analysis_data:
            display_analysis_results()
        else:
            st.info("Select species and click 'Analyze' to see results here.")

    with tab4:
        display_export_options()

def analyze_selected_species(selected_names: List[str], all_species_data: List[Dict]):
    """Perform detailed analysis on selected species."""
    with st.spinner(f"Fetching photos and details for {len(selected_names)} species..."):
        selected_data = [sp for sp in all_species_data if sp['name'] in selected_names]
        
        async def fetch_details_task():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_species_details_async(session, sp['name']) for sp in selected_data]
                return await asyncio.gather(*tasks)

        results = run_async_task(fetch_details_task())
        details_map = {name: {'photos': photos, 'taxon_id': tid} for name, photos, tid in results}

        for sp in selected_data:
            sp.update(details_map.get(sp['name'], {}))
        
        st.session_state.analysis_data = selected_data
        st.success("Analysis complete!")
        st.rerun()

def display_analysis_results():
    """Display detailed analysis results."""
    for species in st.session_state.analysis_data:
        with st.expander(f"🌿 {species['name']} - {species['family']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                success, description = get_local_eflora_description(species.get('speciesKey'), st.session_state.eflora_data)
                st.markdown(description if success else "No local description available.")
                
                if species.get('hierarchy'):
                    st.markdown(f"**Taxonomy:** {' > '.join([f'{h["name"]} ({h["rank"]})' for h in species['hierarchy']])}")
                if species.get('taxon_id'):
                    st.markdown(f"**[View on iNaturalist ↗](https://www.inaturalist.org/taxa/{species['taxon_id']})**")
            with col2:
                for photo in species.get('photos', [])[:3]:
                    license_name = INAT_LICENSE_MAP.get(str(photo.get('license')).lower(), 'Unknown License')
                    st.image(photo['url'], caption=f"© {photo.get('photographer', 'Unknown')} ({license_name})", use_container_width=True)

@st.cache_data(ttl=3600)
def search_gbif_cached(latitude: float, longitude: float, radius_km: int, taxon_name: str) -> Tuple[List[Dict], List[Dict]]:
    """Cached GBIF search function, now with proactive hierarchy fetching."""
    try:
        taxon_info = gbif_species.name_backbone(name=taxon_name, kingdom='plantae')
        if 'usageKey' not in taxon_info:
            st.warning(f"Could not find '{taxon_name}' in GBIF.")
            return [], []
        
        all_records = []
        offset = 0
        limit = 300
        lat_range = f'{latitude - (radius_km / 111.0)},{latitude + (radius_km / 111.0)}'
        lon_range = f'{longitude - (radius_km / 111.0 / math.cos(math.radians(latitude)))},{longitude + (radius_km / 111.0 / math.cos(math.radians(latitude)))}'
        
        while True:
            occ = gbif_occ.search(taxonKey=taxon_info['usageKey'], decimalLatitude=lat_range, decimalLongitude=lon_range, hasCoordinate=True, limit=limit, offset=offset)
            results = occ.get('results', [])
            if not results: break
            all_records.extend(results)
            offset += limit
            if occ['endOfRecords'] or offset > 2000: break
        
        if not all_records:
            st.info(f"No GBIF records found for '{taxon_name}' in this area.")
            return [], []

        df = pd.DataFrame(all_records).dropna(subset=['species', 'family', 'speciesKey'])
        species_counts = df.groupby(['species', 'family', 'speciesKey']).size().reset_index(name='count')
        species_counts.rename(columns={'species': 'name'}, inplace=True)
        species_data = species_counts.sort_values('count', ascending=False).to_dict('records')
        
        logger.info(f"Pre-fetching hierarchies for {len(species_data)} species...")
        species_data_with_hierarchy = run_async_task(fetch_all_hierarchies_parallel(species_data))
        
        st.session_state.current_latitude = latitude
        st.session_state.current_longitude = longitude
        
        return species_data_with_hierarchy, all_records

    except Exception as e:
        st.error(f"Failed to search GBIF. Error: {e}")
        return [], []

def get_local_eflora_description(species_key: Optional[int], eflora_data: pd.DataFrame) -> Tuple[bool, str]:
    """Get description from local e-Flora data using the stable GBIF Usage Key."""
    if species_key is None or eflora_data is None:
        return False, "Description key not provided."

    try:
        match = eflora_data.loc[int(species_key)]
    except (KeyError, ValueError):
        return False, "Species not found in the local e-Flora database."

    descriptions = match.get('descriptions')
    if not isinstance(descriptions, dict): return False, "No description available."

    extracted_data = [f"**Scientific Name:** {match['scientificName']}"]
    if isinstance(match.get('vernacularName'), list):
        extracted_data.append(f"**Common Names:** {', '.join(match['vernacularName'][:5])}")

    section_map = {
        "Morphological Description": ["morphological description", "morphology", "description"],
        "Diagnostic Characters": ["diagnostic characters", "diagnostics", "characters"],
        "Distribution": ["distribution"], "Habitat": ["habitat"]
    }
    available_keys_lower = {k.lower(): k for k in descriptions.keys()}
    
    for display_title, possible_keys in section_map.items():
        for key in possible_keys:
            if key in available_keys_lower:
                original_key = available_keys_lower[key]
                desc_text = str(descriptions[original_key]).strip()
                if desc_text and len(desc_text) > 10:
                    extracted_data.append(f"**{display_title}:**\n{desc_text}")
                    break
    
    return True, "\n\n".join(extracted_data)

def create_clustered_map(records, species_list, center_lat, center_lon, use_clustering=True):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    folium.Marker([center_lat, center_lon], popup="Search Center", icon=folium.Icon(color='red')).add_to(m)
    if use_clustering:
        marker_cluster = MarkerCluster().add_to(m)
        colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
        for i, species in enumerate(species_list[:20]):
            color = colors[i % len(colors)]
            species_records = [r for r in records if r.get('species') == species['name']]
            for record in species_records[:100]:
                lat, lon = record.get('decimalLatitude'), record.get('decimalLongitude')
                if lat and lon:
                    folium.Marker(location=[lat, lon], popup=f"<b>{species['name']}</b>", icon=folium.Icon(color=color, icon='leaf')).add_to(marker_cluster)
    return m

def render_advanced_filters():
    st.subheader("🔬 Advanced Taxonomic Filter")
    st.session_state.rank_filter_enabled = st.checkbox("Enable Rank Filter", value=st.session_state.rank_filter_enabled)
    if not st.session_state.rank_filter_enabled:
        st.session_state.rank_filter_settings = {}
        return

    if st.session_state.discovered_ranks is None:
        if st.button("Discover Available Ranks", width='stretch'):
            st.session_state.discovered_ranks = get_available_ranks_and_values(st.session_state.species_data)
            st.rerun()
    else:
        rank_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'tribe', 'genus']
        available_ranks = list(st.session_state.discovered_ranks.keys())
        sorted_ranks = [r for r in rank_order if r in available_ranks] + [r for r in available_ranks if r not in rank_order]
        selected_rank = st.selectbox("Filter by Rank", sorted_ranks)
        if selected_rank:
            available_values = sorted(list(st.session_state.discovered_ranks[selected_rank]))
            selected_value = st.selectbox(f"Select {selected_rank.title()}", [""] + available_values)
            st.session_state.rank_filter_settings = {'rank': selected_rank, 'value': selected_value} if selected_value else {}

def display_export_options():
    st.header("📄 Export Data")
    if not st.session_state.get('analysis_data'):
        st.warning("Please analyze species first to generate exportable data.")
        return
    # Export logic remains the same...

def main():
    st.markdown("# 🌿 Botanical ID Workbench")
    st.divider()
    show_search_page()

if __name__ == "__main__":
    main()