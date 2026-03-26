import os
from pathlib import Path
from ocr import EnhancedInvoiceProcessor, extract_invoice_details_with_llm

INVOICE_DIR = "Invoice_samples"
processor = EnhancedInvoiceProcessor()

def process_invoice_file(file_path):
    # Open file in binary mode for processor
    with open(file_path, "rb") as f:
        # Mimic Streamlit's UploadedFile object
        class UploadedFile:
            def __init__(self, file, name):
                self.file = file
                self.name = name
                self.type = self._detect_type()
            def read(self):
                self.file.seek(0)
                return self.file.read()
            def _detect_type(self):
                ext = Path(self.name).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg"]:
                    return f"image/{ext[1:]}"
                elif ext == ".pdf":
                    return "application/pdf"
                elif ext == ".docx":
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif ext == ".txt":
                    return "text/plain"
                else:
                    return "application/octet-stream"
        uploaded = UploadedFile(f, file_path)
        text, file_type, meta = processor.extract_invoice_text(uploaded)
        invoice_data = extract_invoice_details_with_llm(text)
        print(f"\n--- {os.path.basename(file_path)} ---")
        print(invoice_data)

def main():
    for fname in os.listdir(INVOICE_DIR):
        fpath = os.path.join(INVOICE_DIR, fname)
        if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".pdf", ".docx", ".txt"]):
            process_invoice_file(fpath)

if __name__ == "__main__":
    main()