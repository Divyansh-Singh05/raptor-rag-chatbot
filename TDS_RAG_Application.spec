
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
