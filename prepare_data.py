# prepare_data.py
"""
Standalone data preparation script for Botanical Workbench.
Run this separately to download and prepare e-Flora data.
"""

import os
import requests
import zipfile
import pandas as pd
import json
from datetime import datetime
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotanicalDataPreparer:
    def __init__(self, data_dir="prepared_data"):
        self.data_dir = data_dir
        self.eflora_url = "https://ipt.sanbi.org.za/archive.do?r=flora_descriptions&v=1.42"
        self.version_file = os.path.join(data_dir, "data_version.json")
        self.processed_file = os.path.join(data_dir, "eflora_processed.parquet")
        
    def ensure_directory(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_remote_checksum(self):
        """Get checksum of remote file (simplified - could use HEAD request)."""
        # In production, you'd want to check actual file metadata
        return hashlib.md5(self.eflora_url.encode()).hexdigest()
    
    def download_and_extract(self):
        """Download and extract e-Flora data."""
        logger.info("Downloading e-Flora data...")
        zip_path = os.path.join(self.data_dir, "temp_eflora.zip")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(self.eflora_url, stream=True, headers=headers)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            os.remove(zip_path)
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False
    
    def process_data(self):
        """Process the raw data files into a single optimized file."""
        logger.info("Processing e-Flora data...")
        
        try:
            # Load files
            taxa_df = pd.read_csv(
                os.path.join(self.data_dir, 'taxon.txt'), 
                sep='\t', header=0,
                usecols=['id', 'scientificName'], 
                dtype={'id': str}
            )
            
            desc_df = pd.read_csv(
                os.path.join(self.data_dir, 'description.txt'), 
                sep='\t', header=0,
                usecols=['id', 'description', 'type'], 
                dtype={'id': str}
            )
            
            vernacular_df = pd.read_csv(
                os.path.join(self.data_dir, 'vernacularname.txt'), 
                sep='\t', header=0,
                usecols=['id', 'vernacularName'], 
                dtype={'id': str}
            )
            
            # Process and merge
            for df in [taxa_df, desc_df, vernacular_df]:
                df.rename(columns={'id': 'taxonID'}, inplace=True)
            
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
            
            # Save as Parquet for efficient loading
            eflora_data.to_parquet(self.processed_file, index=False)
            
            # Save version information
            version_info = {
                "source_url": self.eflora_url,
                "processed_date": datetime.now().isoformat(),
                "checksum": self.get_remote_checksum(),
                "record_count": len(eflora_data),
                "version": "1.42"
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            logger.info(f"Successfully processed {len(eflora_data)} taxa")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
    
    def prepare(self, force_update=False):
        """Main preparation method."""
        self.ensure_directory()
        
        # Check if data exists and is current
        if not force_update and os.path.exists(self.processed_file):
            logger.info("Processed data already exists. Use --force to re-download.")
            return True
        
        # Download and process
        if self.download_and_extract():
            return self.process_data()
        
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare botanical data for the workbench")
    parser.add_argument("--force", action="store_true", help="Force re-download and processing")
    parser.add_argument("--data-dir", default="prepared_data", help="Data directory")
    args = parser.parse_args()
    
    preparer = BotanicalDataPreparer(data_dir=args.data_dir)
    
    if preparer.prepare(force_update=args.force):
        print("✅ Data preparation successful!")
        print(f"📁 Data saved to: {args.data_dir}")
    else:
        print("❌ Data preparation failed!")
        exit(1)
