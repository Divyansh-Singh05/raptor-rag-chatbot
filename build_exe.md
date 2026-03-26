# TDS RAG Application - Universal EXE Builder

This guide will help you create a universal `.exe` file for your TDS RAG Application that can run on any Windows machine without requiring Python installation.

## Files Overview

1. **`launcher.py`** - The main launcher script that becomes your .exe
2. **`build_exe.py`** - Python script to build the executable
3. **`build_exe.bat`** - Simple batch file to run the build process

## Quick Start

### Method 1: Using the Batch File (Easiest)
1. Double-click `build_exe.bat`
2. Wait for the build to complete
3. Find your `.exe` in the `dist/` folder

### Method 2: Using Python Directly
1. Open Command Prompt/Terminal in your project directory
2. Activate your virtual environment (if using one):
   ```bash
   venv\Scripts\activate  # Windows
   ```
3. Run the build script:
   ```bash
   python build_exe.py
   ```

## What the Launcher Does

The created `.exe` file will:

1. **Smart RAG Processing**: Only runs `unified_rag_pipeline.py` when:
   - First time running (no embeddings exist)
   - Data files in `/data/` folder have changed
   - Embeddings are corrupted or missing

2. **Automatic Browser Launch**: Opens the Streamlit app in your default browser

3. **Environment Management**: Handles all Python dependencies and environment setup

4. **Progress Logging**: Shows clear status messages during startup

## How to Use the Generated EXE

1. **Copy the entire project folder** to any Windows machine
2. **Double-click `TDS_RAG_Application.exe`**
3. **Wait for initialization** (first run may take a few minutes)
4. **Application opens automatically** in your browser at `http://localhost:8501`

## Important Notes

### File Structure Requirements
The `.exe` must be in the same directory as:
```
rag-ey-final/
├── data/                           # Your source documents
├── embeddings/                     # Vector database (auto-created)
├── chunks.json                     # Document chunks (auto-created)
├── .env                           # Environment variables
├── TDS_RAG_Application.exe        # Your built executable
└── [all other Python files]      # Required Python modules
```

### Distribution
- **Copy the entire folder**, not just the `.exe` file
- The `.exe` is portable but needs its supporting files
- Target machines don't need Python installed

### Performance Notes
- **First Run**: May take 2-5 minutes to process documents and create embeddings
- **Subsequent Runs**: Starts in 10-30 seconds
- **After Data Changes**: Automatically reprocesses only what's needed

## Troubleshooting

### Build Fails
- Ensure all files from your project structure are present
- Check that your virtual environment has all required packages
- Try running `pip install pyinstaller` manually

### EXE Won't Start
- Check that all project files are in the same directory as the `.exe`
- Ensure `.env` file contains your API keys
- Run from Command Prompt to see error messages

### Browser Doesn't Open
- The app is still running at `http://localhost:8501`
- Manually open this URL in your browser
- Check Windows Firewall isn't blocking the application

### Slow First Startup
- This is normal - the app is processing your documents
- Subsequent starts will be much faster
- Progress is shown in the console window

## Customization

### Changing the Port
Edit `launcher.py` line with `STREAMLIT_SERVER_PORT` to use a different port.

### Adding More Files
Edit `build_exe.py` in the `data_files` list to include additional files in the executable.

### Icon
Add `icon='path/to/icon.ico'` in the `build_exe.py` EXE section to add a custom icon.

## Technical Details

- **PyInstaller**: Used to create the standalone executable
- **Embedded Python**: Includes Python runtime and all dependencies
- **File Integrity**: Uses MD5 hashing to detect data changes
- **Process Management**: Handles Streamlit subprocess lifecycle
- **Resource Management**: Automatically cleans up on exit

## File Sizes
- Typical `.exe` size: 200-500 MB (includes Python + AI libraries)
- Full distribution folder: 300-800 MB
- This is normal for AI applications with large dependencies

---

## Support

If you encounter issues:
1. Check that your original Python application works correctly
2. Ensure all dependencies are installed in your virtual environment
3. Verify all files from the original structure are present
4. Check Windows Event Viewer for detailed error messages