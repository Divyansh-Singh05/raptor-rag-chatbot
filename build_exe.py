#!/usr/bin/env python3
"""
Build script to create a completely self-contained .exe file
This creates a single executable that contains all files and dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install PyInstaller")
            return False

def create_spec_file():
    """Create a spec file for a completely self-contained executable"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Get all data files that need to be embedded
def get_data_files():
    data_files = []
    
    # Add the entire data directory
    if os.path.exists('data'):
        for root, dirs, files in os.walk('data'):
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.relpath(root, '.')
                data_files.append((src, dst))
    
    # Add embeddings directory if it exists
    if os.path.exists('embeddings'):
        for root, dirs, files in os.walk('embeddings'):
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.relpath(root, '.')
                data_files.append((src, dst))
    
    # Add individual files
    individual_files = [
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
    
    for file in individual_files:
        if os.path.exists(file):
            data_files.append((file, '.'))
    
    return data_files

# Hidden imports for packages that might not be detected automatically
hidden_imports = [
    'streamlit',
    'streamlit.cli',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.runtime.state',
    'streamlit.runtime.caching',
    'streamlit.web.cli',
    'streamlit.web.bootstrap',
    'altair',
    'pandas',
    'numpy',
    'faiss',
    'sentence_transformers',
    'transformers',
    'torch',
    'PIL',
    'pytesseract',
    'pdf2image',
    'langchain',
    'langchain_community',
    'langchain_openai',
    'openai',
    'anthropic',
    'google.generativeai',
    'sklearn',
    'matplotlib',
    'plotly',
    'requests',
    'boto3',
    'python-dotenv',
    'fitz',  # PyMuPDF
    'docx2txt',
    'openpyxl',
    'xlrd',
    'pdfplumber',
    'streamlit_option_menu',
    'streamlit_chat',
    'streamlit_extras',
    'click',
    'tornado',
    'watchdog',
    'gitpython',
    'pympler',
    'validators',
    'packaging',
    'toml',
    'tzlocal',
    'pytz',
    'dateutil',
    'cachetools',
    'tenacity',
    'jsonschema',
    'streamlit.components.v1.components',
    'streamlit.delta_generator',
    'streamlit.elements',
    'streamlit.runtime',
    'streamlit.runtime.uploaded_file_manager'
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=get_data_files(),
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'tkinter.ttk',
        'matplotlib.backends._backend_tk',
        'PIL.ImageTk',
        'PIL._tkinter_finder'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TDS_RAG_Application',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('TDS_RAG_Application.spec', 'w') as f:
        f.write(spec_content)
    
    print("Created TDS_RAG_Application.spec file")

def prepare_for_build():
    """Prepare the environment for building"""
    print("Preparing build environment...")
    
    # Check required files
    required_files = ['launcher.py', 'tds_app6.py', 'model.py']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        return False
    
    # Create empty directories if they don't exist
    Path('embeddings').mkdir(exist_ok=True)
    
    # Create empty chunks.json if it doesn't exist
    chunks_file = Path('chunks.json')
    if not chunks_file.exists():
        with open(chunks_file, 'w') as f:
            f.write('[]')
        print("Created empty chunks.json file")
    
    return True

def build_executable():
    """Build the self-contained executable using PyInstaller"""
    print("Building self-contained executable...")
    
    try:
        # Prepare for build
        if not prepare_for_build():
            return False
        
        # Create the spec file
        create_spec_file()
        
        # Run PyInstaller with the spec file
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "--onefile",  # This is crucial for single file
            "TDS_RAG_Application.spec"
        ]
        
        print("Starting PyInstaller build process...")
        print("This may take 10-20 minutes depending on your system...")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("Build completed successfully!")
            
            # Check if the exe was created
            exe_path = Path("dist/TDS_RAG_Application.exe")
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"Executable created: {exe_path}")
                print(f"File size: {size_mb:.1f} MB")
                return True
            else:
                print("Error: Executable was not created")
                return False
        else:
            print("Build failed!")
            print(f"Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error during build: {str(e)}")
        return False

def test_executable():
    """Test the created executable"""
    exe_path = Path("dist/TDS_RAG_Application.exe")
    if exe_path.exists():
        print("\\nTesting executable...")
        print("The executable should:")
        print("1. Extract files to a temporary directory")
        print("2. Process documents (if needed)")
        print("3. Start Streamlit and open browser")
        print("4. Clean up when closed")
        
        response = input("\\nWould you like to test the executable now? (y/N): ").lower().strip()
        if response == 'y':
            try:
                subprocess.run([str(exe_path)], check=False)
            except KeyboardInterrupt:
                print("Test interrupted")

def cleanup_build_files():
    """Clean up temporary build files"""
    build_dir = Path("build")
    spec_file = Path("TDS_RAG_Application.spec")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Cleaned up build directory")
    
    if spec_file.exists():
        spec_file.unlink()
        print("Cleaned up spec file")

def main():
    """Main build function"""
    print("=" * 60)
    print("TDS RAG Application - Self-Contained EXE Builder")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("launcher.py").exists():
        print("Error: launcher.py not found. Please run this script from the project root directory.")
        return 1
    
    # Install PyInstaller if needed
    if not install_pyinstaller():
        return 1
    
    # Build the executable
    if build_executable():
        print("\\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        
        exe_path = Path("dist/TDS_RAG_Application.exe")
        print(f"Your self-contained executable: {exe_path.absolute()}")
        print("\\nDistribution Instructions:")
        print("1. Copy ONLY the .exe file to any Windows machine")
        print("2. Double-click to run (no installation needed)")
        print("3. First run extracts files and may take a few minutes")
        print("4. App will open automatically in browser")
        print("\\nIMPORTANT:")
        print("- The .exe contains ALL your data and code")
        print("- No additional files needed")
        print("- Works on any Windows machine without Python")
        print("- API keys are embedded (keep the .exe secure)")
        
        # Test the executable
        test_executable()
        
        # Ask if user wants to clean up
        response = input("\\nClean up build files? (Y/n): ").lower().strip()
        if response != 'n':
            cleanup_build_files()
        
        return 0
    else:
        print("\\nBuild failed. Please check the error messages above.")
        print("\\nCommon solutions:")
        print("1. Ensure all Python files are present")
        print("2. Check that virtual environment has all dependencies")
        print("3. Try: pip install --upgrade pyinstaller")
        print("4. Make sure .env file exists with API keys")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\\nPress Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\nBuild interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\\nUnexpected error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)