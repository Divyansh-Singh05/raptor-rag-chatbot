#!/usr/bin/env python3
"""
Self-Contained Universal Launcher for TDS RAG Application
This script creates a completely self-contained executable that:
1. Extracts all embedded files to a temporary directory
2. Checks for existing embeddings and chunks
3. Runs unified_rag_pipeline.py only when needed
4. Launches the Streamlit app automatically
5. Cleans up temporary files on exit
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
import json
import tempfile
import shutil
import atexit
from pathlib import Path
import hashlib
from datetime import datetime

# Global variable to track temp directory
TEMP_DIR = None

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def extract_embedded_files():
    """Extract all embedded files to a temporary directory"""
    global TEMP_DIR
    
    # Create a temporary directory
    TEMP_DIR = tempfile.mkdtemp(prefix="tds_rag_app_")
    log_message(f"Extracting files to: {TEMP_DIR}")
    
    # Files and directories to extract
    items_to_extract = [
        'data',
        'embeddings', 
        'chunks.json',
        'requirements.txt',
        '.env',
        'tds_app6.py',
        'model.py',
        'invoice_processor.py',
        'ocr.py',
        'bfs_backtrack.py',
        'unified_rag_pipeline.py',
        'advanced_parsing.py',
        'pdf_objects.py',
        'raptor_pipeline.py',
        'raptor_setup.py',
        'vectorstore_ingest.py'
    ]
    
    # Extract each item
    for item in items_to_extract:
        src_path = get_resource_path(item)
        dst_path = os.path.join(TEMP_DIR, item)
        
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                log_message(f"Extracted directory: {item}")
            elif os.path.isfile(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                log_message(f"Extracted file: {item}")
        except Exception as e:
            # Some files might not exist, that's okay
            if item in ['chunks.json', 'embeddings']:
                log_message(f"Optional file not found: {item} (will be created)")
            else:
                log_message(f"Warning: Could not extract {item}: {str(e)}")
    
    return TEMP_DIR

def cleanup_temp_files():
    """Clean up temporary files"""
    global TEMP_DIR
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            log_message("Cleaned up temporary files")
        except Exception as e:
            log_message(f"Warning: Could not clean up temp files: {str(e)}")

def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def calculate_data_hash(work_dir):
    """Calculate hash of all files in data directory to detect changes"""
    data_dir = Path(work_dir) / "data"
    if not data_dir.exists():
        return None
    
    hash_md5 = hashlib.md5()
    for file_path in sorted(data_dir.glob("*")):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_embeddings_exist(work_dir):
    """Check if embeddings and chunks exist"""
    embeddings_dir = Path(work_dir) / "embeddings"
    chunks_file = Path(work_dir) / "chunks.json"
    
    faiss_file = embeddings_dir / "index.faiss"
    pkl_file = embeddings_dir / "index.pkl"
    
    return (
        faiss_file.exists() and 
        pkl_file.exists() and 
        chunks_file.exists() and 
        chunks_file.stat().st_size > 0
    )

def load_hash_record(work_dir):
    """Load the previously stored data hash"""
    hash_file = Path(work_dir) / ".data_hash"
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                return f.read().strip()
        except:
            return None
    return None

def save_hash_record(work_dir, data_hash):
    """Save the current data hash"""
    hash_file = Path(work_dir) / ".data_hash"
    with open(hash_file, 'w') as f:
        f.write(data_hash)

def needs_rag_processing(work_dir):
    """Determine if RAG processing is needed"""
    # Check if embeddings exist
    if not check_embeddings_exist(work_dir):
        log_message("Embeddings not found. RAG processing required.")
        return True
    
    # Check if data has changed
    current_hash = calculate_data_hash(work_dir)
    if current_hash is None:
        log_message("No data directory found. Skipping RAG processing.")
        return False
    
    stored_hash = load_hash_record(work_dir)
    if current_hash != stored_hash:
        log_message("Data changes detected. RAG processing required.")
        return True
    
    log_message("Embeddings exist and data unchanged. Skipping RAG processing.")
    return False

def run_rag_pipeline(work_dir):
    """Run the unified RAG pipeline"""
    log_message("Starting RAG pipeline processing...")
    
    try:
        # Change to work directory
        original_cwd = os.getcwd()
        os.chdir(work_dir)
        
        # Add work directory to Python path
        sys.path.insert(0, work_dir)
        
        # Import and run the unified RAG pipeline
        import unified_rag_pipeline
        log_message("RAG pipeline completed successfully.")
        
        # Save the current data hash
        current_hash = calculate_data_hash(work_dir)
        if current_hash:
            save_hash_record(work_dir, current_hash)
        
        # Restore original directory
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        log_message(f"Error running RAG pipeline: {str(e)}")
        # Restore original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
        return False

def wait_for_streamlit(port=8501, max_attempts=30):
    """Wait for Streamlit to start"""
    import socket
    
    for attempt in range(max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                log_message(f"Streamlit is running on port {port}")
                return True
        except:
            pass
        
        time.sleep(1)
    
    return False

def launch_streamlit(work_dir):
    """Launch the Streamlit application"""
    log_message("Starting Streamlit application...")
    
    # Set environment variables
    env = os.environ.copy()
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    env['STREAMLIT_SERVER_PORT'] = '8501'
    env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    try:
        # Launch Streamlit in a subprocess
        cmd = [sys.executable, "-m", "streamlit", "run", "tds_app6.py", "--server.headless", "true"]
        
        process = subprocess.Popen(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Wait for Streamlit to start
        if wait_for_streamlit():
            log_message("Opening application in browser...")
            time.sleep(2)  # Give it a moment to fully initialize
            webbrowser.open('http://localhost:8501')
            
            log_message("TDS RAG Application is now running!")
            log_message("You can access it at: http://localhost:8501")
            log_message("Close this window to stop the application.")
            
            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                log_message("Shutting down application...")
                process.terminate()
                process.wait()
        else:
            log_message("Failed to start Streamlit application")
            process.terminate()
            return False
            
    except Exception as e:
        log_message(f"Error launching Streamlit: {str(e)}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("=" * 60)
    log_message("TDS RAG Application Launcher Starting...")
    print("=" * 60)
    
    # Register cleanup function
    atexit.register(cleanup_temp_files)
    
    try:
        # Extract embedded files to temporary directory
        work_dir = extract_embedded_files()
        
        # Change to work directory
        os.chdir(work_dir)
        
        # Add work directory to Python path
        sys.path.insert(0, work_dir)
        
        # Check if RAG processing is needed
        if needs_rag_processing(work_dir):
            log_message("Initializing knowledge base...")
            if not run_rag_pipeline(work_dir):
                log_message("Failed to initialize knowledge base. Application may not work correctly.")
                response = input("Continue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    return 1
        
        # Launch Streamlit application
        success = launch_streamlit(work_dir)
        
        if not success:
            log_message("Failed to launch application.")
            input("Press Enter to exit...")
            return 1
        
        return 0
        
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}")
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("Application interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)