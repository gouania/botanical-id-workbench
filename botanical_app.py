# botanical_workbench.py
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "prepared_data"
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "eflora_processed.parquet")
VERSION_FILE = os.path.join(DATA_DIR, "data_version.json")
MAX_CONCURRENT_REQUESTS = 10  # For parallel API calls

st.set_page_config(
    page_title="Botanical ID Workbench: South Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    defaults = {
        'species_data': None,
        'selected_species': {},
        'eflora_data': None,
        'analysis_data': None,
        'all_records': None,
        'page': 'search',  # Track current page
        'filter_settings': {},
        'map_cluster': True
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

@st.cache_data(ttl=3600*24)  # Cache for 24 hours
def load_prepared_data() -> Optional[pd.DataFrame]:
    """Load pre-processed e-Flora data."""
    if not os.path.exists(PROCESSED_DATA_FILE):
        st.error("""
        ⚠️ **Data not found!**
        
        Please run the data preparation script first:
        ```bash
        python prepare_data.py
        ```
        """)
        return None
    
    try:
        data = pd.read_parquet(PROCESSED_DATA_FILE)
        
        # Load version info
        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE, 'r') as f:
                version_info = json.load(f)
                st.sidebar.caption(f"📊 Data version: {version_info.get('version', 'Unknown')}")
                st.sidebar.caption(f"📅 Processed: {version_info.get('processed_date', 'Unknown')[:10]}")
        
        # Set index for efficient lookups
        data['cleanScientificName'] = data['scientificName'].apply(
            lambda x: ' '.join(str(x).split()[:2]) if pd.notna(x) else ''
        )
        data.set_index('cleanScientificName', inplace=True)
        data = data[~data.index.duplicated(keep='first')]
        
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Async function for parallel iNaturalist API calls
async def fetch_species_details_async(session, species_name: str, limit: int = 5):
    """Async version of species details fetching."""
    headers = {'User-Agent': 'BotanicalWorkbench/1.0'}
    
    try:
        # Search for taxon
        search_url = f"https://api.inaturalist.org/v1/taxa/autocomplete?q={species_name}&per_page=1"
        async with session.get(search_url, headers=headers) as response:
            if response.status != 200:
                return species_name, [], None, []
            
            data = await response.json()
            if not data.get('results'):
                return species_name, [], None, []
            
            taxon = data['results'][0]
            taxon_id = taxon['id']
            
            # Fetch photos and hierarchy in parallel
            photos_task = fetch_photos_async(session, taxon_id, taxon, limit)
            hierarchy_task = fetch_hierarchy_async(session, taxon_id)
            
            photos, hierarchy = await asyncio.gather(photos_task, hierarchy_task)
            
            return species_name, photos, taxon_id, hierarchy
            
    except Exception as e:
        logger.error(f"Error fetching details for {species_name}: {e}")
        return species_name, [], None, []

async def fetch_photos_async(session, taxon_id, taxon, limit):
    """Fetch photos asynchronously."""
    photos = []
    headers = {'User-Agent': 'BotanicalWorkbench/1.0'}
    
    # Process default photo
    default_photo = taxon.get('default_photo')
    if default_photo:
        photo_url = default_photo.get('medium_url') or default_photo.get('square_url')
        if photo_url:
            photos.append({
                'url': photo_url,
                'photographer': 'iNaturalist',
                'license': default_photo.get('license_code', 'Unknown')
            })
    
    # Fetch observations
    obs_url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&photos=true&per_page={limit}"
    try:
        async with session.get(obs_url, headers=headers) as response:
            if response.status == 200:
                obs_data = await response.json()
                for obs in obs_data.get('results', [])[:limit]:
                    for photo in obs.get('photos', [])[:1]:
                        photo_url = photo.get('url', '').replace('square', 'medium')
                        if photo_url and photo_url not in [p['url'] for p in photos]:
                            photos.append({
                                'url': photo_url,
                                'photographer': obs.get('user', {}).get('login', 'Unknown'),
                                'license': photo.get('license_code', 'Unknown')
                            })
                        if len(photos) >= limit:
                            break
    except Exception as e:
        logger.error(f"Error fetching photos: {e}")
    
    return photos

async def fetch_hierarchy_async(session, taxon_id):
    """Fetch taxonomic hierarchy asynchronously."""
    headers = {'User-Agent': 'BotanicalWorkbench/1.0'}
    
    try:
        taxon_url = f"https://api.inaturalist.org/v1/taxa/{taxon_id}"
        async with session.get(taxon_url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                ancestors = data.get('results', [{}])[0].get('ancestors', [])
                return [{'rank': a['rank'], 'name': a['name'], 'id': a.get('id')} 
                       for a in ancestors if a.get('rank')]
    except Exception as e:
        logger.error(f"Error fetching hierarchy: {e}")
    
    return []

def fetch_all_species_details_parallel(species_list: List[str], limit: int = 5) -> Dict:
    """Fetch details for multiple species in parallel."""
    async def fetch_all():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_species_details_async(session, sp, limit) for sp in species_list]
            results = await asyncio.gather(*tasks)
            return {name: {'photos': photos, 'taxon_id': tid, 'hierarchy': hier} 
                   for name, photos, tid, hier in results}
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fetch_all())
    finally:
        loop.close()

def create_clustered_map(records, species_list, center_lat, center_lon, use_clustering=True):
    """Create map with optional clustering."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lon],
        popup="Search Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    if use_clustering:
        # Use marker clustering for better performance
        marker_cluster = MarkerCluster().add_to(m)
        
        colors = ['green', 'blue', 'purple', 'orange', 'darkred', 
                 'lightred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
        
        for i, species in enumerate(species_list[:20]):  # Limit for performance
            color = colors[i % len(colors)]
            species_records = [r for r in records if r.get('species') == species['name']]
            
            for record in species_records[:100]:  # Limit records per species
                lat = record.get('decimalLatitude')
                lon = record.get('decimalLongitude')
                if lat and lon:
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"<b>{species['name']}</b><br>Family: {species['family']}",
                        icon=folium.Icon(color=color, icon='leaf')
                    ).add_to(marker_cluster)
    else:
        # Original non-clustered approach (simplified)
        colors = ['#2d5016', '#4a7c24', '#6b8e23']
        for i, species in enumerate(species_list[:5]):
            color = colors[i % len(colors)]
            species_records = [r for r in records if r.get('species') == species['name']]
            
            for record in species_records[:20]:
                lat = record.get('decimalLatitude')
                lon = record.get('decimalLongitude')
                if lat and lon:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f"{species['name']}",
                        color=color,
                        fill=True,
                        fillColor=color
                    ).add_to(m)
    
    return m

# Navigation functions
def show_search_page():
    """Main search interface."""
    st.title("🌿 Botanical ID Workbench")
    
    # Load e-Flora data
    if st.session_state.eflora_data is None:
        st.session_state.eflora_data = load_prepared_data()
        if st.session_state.eflora_data is None:
            return
    
    # Create two columns for search interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📍 Search Parameters")
        
        # Coordinate input
        coord_input = st.text_area(
            "Enter Coordinates",
            value="-33.92, 18.42",
            height=60,
            help="Enter as 'latitude, longitude' or paste from clipboard"
        )
        
        try:
            lat_str, lon_str = coord_input.strip().split(',')
            latitude = float(lat_str.strip())
            longitude = float(lon_str.strip())
            st.success(f"✓ Valid coordinates: {latitude:.4f}, {longitude:.4f}")
        except:
            st.error("Invalid format. Use: latitude, longitude")
            latitude, longitude = -33.92, 18.42
        
        radius_km = st.slider("Search Radius (km)", 1, 100, 25)
        
        taxon_name = st.text_input("Taxon Name", value="Protea",
                                   help="Genus or higher taxon to search")
        
        # Search button
        if st.button("🔍 Search GBIF", type="primary", use_container_width=True):
            with st.spinner("Searching GBIF..."):
                species_data, all_records = search_gbif_cached(
                    latitude, longitude, radius_km, taxon_name
                )
                st.session_state.species_data = species_data
                st.session_state.all_records = all_records
                st.rerun()
    
    with col2:
        st.subheader("📍 Search Area Preview")
        preview_map = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker([latitude, longitude], 
                     popup="Search Center",
                     icon=folium.Icon(color='red')).add_to(preview_map)
        folium.Circle(
            location=[latitude, longitude],
            radius=radius_km * 1000,
            color="#2d5016",
            fill=True,
            fillOpacity=0.2
        ).add_to(preview_map)
        st_folium(preview_map, height=400)
    
    # Results section
    if st.session_state.species_data:
        st.divider()
        show_results_section()

def show_results_section():
    """Display and manage search results."""
    st.subheader(f"🎯 Found {len(st.session_state.species_data)} species")
    
    # Post-search filters
    with st.expander("🔧 Refine Results", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_records = st.number_input("Minimum records", min_value=1, value=1)
        
        with col2:
            families = ['All'] + sorted(list(set(sp['family'] for sp in st.session_state.species_data)))
            selected_family = st.selectbox("Filter by Family", families)
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Record Count", "Name", "Family"])
        
        # Apply filters
        filtered_data = st.session_state.species_data
        
        if min_records > 1:
            filtered_data = [sp for sp in filtered_data if sp['count'] >= min_records]
        
        if selected_family != 'All':
            filtered_data = [sp for sp in filtered_data if sp['family'] == selected_family]
        
        if sort_by == "Name":
            filtered_data = sorted(filtered_data, key=lambda x: x['name'])
        elif sort_by == "Family":
            filtered_data = sorted(filtered_data, key=lambda x: (x['family'], x['name']))
        else:  # Record Count
            filtered_data = sorted(filtered_data, key=lambda x: x['count'], reverse=True)
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Table View", "🗺️ Map View", "📋 Analysis"])
    
    with tab1:
        # Create DataFrame for display
        df = pd.DataFrame(filtered_data)
        df_display = df[['name', 'family', 'count']].copy()
        df_display.columns = ['Species', 'Family', 'Records']
        
        # Add selection column
        df_display.insert(0, 'Select', False)
        
        # Display with selection
        edited_df = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select species for detailed analysis",
                    default=False,
                )
            }
        )
        
        # Get selected species
        selected_species = edited_df[edited_df['Select']]['Species'].tolist()
        
        if selected_species:
            st.info(f"Selected {len(selected_species)} species")
            
            if st.button("🔬 Analyze Selected Species", type="primary"):
                analyze_selected_species(selected_species, filtered_data)
    
    with tab2:
        # Map options
        col1, col2 = st.columns([3, 1])
        with col2:
            use_clustering = st.checkbox("Use Clustering", value=True,
                                        help="Group nearby points for better performance")
        
        # Display map
        if st.session_state.all_records:
            species_map = create_clustered_map(
                st.session_state.all_records,
                filtered_data[:20],  # Limit for performance
                st.session_state.current_latitude if 'current_latitude' in st.session_state else -33.92,
                st.session_state.current_longitude if 'current_longitude' in st.session_state else 18.42,
                use_clustering=use_clustering
            )
            st_folium(species_map, height=600)
    
    with tab3:
        if 'analysis_data' in st.session_state and st.session_state.analysis_data:
            display_analysis_results()
        else:
            st.info("Select species in the Table View tab and click 'Analyze Selected Species'")

def analyze_selected_species(selected_names: List[str], all_species_data: List[Dict]):
    """Perform detailed analysis on selected species using parallel processing."""
    with st.spinner(f"Analyzing {len(selected_names)} species in parallel..."):
        # Get species data
        selected_data = [sp for sp in all_species_data if sp['name'] in selected_names]
        
        # Fetch details in parallel
        details = fetch_all_species_details_parallel(selected_names)
        
        # Combine data
        for sp in selected_data:
            if sp['name'] in details:
                sp.update(details[sp['name']])
        
        st.session_state.analysis_data = selected_data
        st.success(f"Analysis complete for {len(selected_data)} species!")
        st.rerun()

def display_analysis_results():
    """Display detailed analysis results."""
    for species in st.session_state.analysis_data:
        with st.expander(f"🌿 {species['name']} - {species['family']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display e-Flora description
                success, description = get_local_eflora_description(
                    species['name'], st.session_state.eflora_data
                )
                
                if success:
                    st.markdown(description)
                else:
                    st.info("No local description available")
                
                # Display hierarchy if available
                if species.get('hierarchy'):
                    hierarchy_str = " > ".join([f"{h['name']} ({h['rank']})" 
                                              for h in species['hierarchy']])
                    st.markdown(f"**Taxonomy:** {hierarchy_str}")
            
            with col2:
                # Display photos if available
                if species.get('photos'):
                    for photo in species['photos'][:2]:
                        st.image(photo['url'], 
                               caption=f"© {photo['photographer']}",
                               use_container_width=True)

def show_admin_page():
    """Administrative functions page."""
    st.title("⚙️ Admin & Diagnostics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Status", "🗃️ Cache Management", "🔧 Diagnostics"])
    
    with tab1:
        st.subheader("Data Status")
        
        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE, 'r') as f:
                version_info = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Version", version_info.get('version', 'Unknown'))
                st.metric("Record Count", f"{version_info.get('record_count', 0):,}")
            with col2:
                st.metric("Processed Date", version_info.get('processed_date', 'Unknown')[:10])
                st.metric("Source", "SANBI e-Flora")
        else:
            st.warning("No version information available. Run data preparation script.")
        
        if st.button("🔄 Check for Updates"):
            st.info("Update checking not implemented in this version")
    
    with tab2:
        st.subheader("Cache Management")
        
        # Calculate cache sizes
        cache_dirs = ['gbif_cache', 'cache']
        total_size = 0
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                          for f in os.listdir(cache_dir))
                total_size += size
                st.metric(f"{cache_dir}", f"{size / 1024 / 1024:.2f} MB")
        
        if st.button("🗑️ Clear All Caches", type="secondary"):
            import shutil
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            st.success("All caches cleared!")
            st.rerun()
    
    with tab3:
        st.subheader("System Diagnostics")
        
        # Test e-Flora data
        if st.button("Test e-Flora Data"):
            data = load_prepared_data()
            if data is not None:
                st.success(f"✅ Successfully loaded {len(data)} taxa")
                
                # Show sample
                st.write("Sample entries:")
                st.dataframe(data.head())
            else:
                st.error("Failed to load e-Flora data")
        
        # Test API connectivity
        if st.button("Test API Connectivity"):
            test_results = test_api_connectivity()
            for service, status in test_results.items():
                if status:
                    st.success(f"✅ {service}: Connected")
                else:
                    st.error(f"❌ {service}: Failed")

# Helper functions (keep existing ones, add these)
@st.cache_data(ttl=3600)
def search_gbif_cached(latitude, longitude, radius_km, taxon_name):
    """Cached GBIF search function."""
    # Your existing get_species_list_from_gbif logic here
    # Just rename it for clarity
    pass

def get_local_eflora_description(species_name, eflora_data):
    """Get description from local e-Flora data."""
    # Your existing implementation
    pass

def test_api_connectivity():
    """Test connectivity to external APIs."""
    import requests
    
    results = {}
    
    # Test GBIF
    try:
        response = requests.get("https://api.gbif.org/v1/", timeout=5)
        results['GBIF API'] = response.status_code == 200
    except:
        results['GBIF API'] = False
    
    # Test iNaturalist
    try:
        response = requests.get("https://api.inaturalist.org/v1/", timeout=5)
        results['iNaturalist API'] = response.status_code == 200
    except:
        results['iNaturalist API'] = False
    
    return results

# Main app with navigation
def main():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Top navigation
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        st.markdown("# 🌿 Botanical ID Workbench")
    
    with col2:
        if st.button("🔍 Search", use_container_width=True):
            st.session_state.page = 'search'
            st.rerun()
    
    with col3:
        if st.button("⚙️ Admin", use_container_width=True):
            st.session_state.page = 'admin'
            st.rerun()
    
    st.divider()
    
    # Display appropriate page
    if st.session_state.page == 'admin':
        show_admin_page()
    else:
        show_search_page()

if __name__ == "__main__":
    main()
