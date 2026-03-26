import os
import json
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model import llm


INVOICE_EXTRACTION_PROMPT = """
You are an expert in analyzing invoices and business documents. Extract the following information from the provided text:

Invoice/Document Text:
{invoice_text}

Please extract and respond in EXACTLY this JSON format:
{{
    "state": "State where transaction occurred",
    "city": "City where transaction occurred", 
    "amount": "Transaction amount (numerical value with currency)",
    "payment_type": "Clear, detailed description of the goods/services being paid for",
    "vendor_name": "Name of the vendor/service provider",
    "invoice_number": "Invoice or bill number",
    "date": "Invoice or transaction date",
    "confidence": "Confidence percentage (0-100)"
}}

Focus on:
- Look for location details (billing address, service location, etc.)
- Identify the exact nature of goods/services (be specific - e.g., "Software development services", "Equipment rental", "Professional consultation")
- Extract transaction amount with currency
- Find vendor/company details
- Extract invoice number and date if available

If information is not clearly available, use "Not found" for that field.
Do NOT attempt to classify TDS sections - only extract factual information from the document.
"""

SERVICE_CLASSIFICATION_PROMPT = """
You are a financial document intelligence expert.

Your task is to classify the payment description into one of the following categories:

1. Goods Sale
2. Passenger Transport
3. Goods Transport / Logistics
4. Technical Service / SaaS / Software
5. Maintenance / AMC
6. Revamping / Civil Work
7. Glazing / Fabrication
8. Rent / Lease
9. Accommodation / Hospitality
10. Supervision / Site Charges
11. Consulting Services
12. Audit / CA Fees
13. Marketing Services
14. Advertising / Promotion
15. Insurance
16. Commission / Brokerage
17. Legal / Compliance
18. Training / Education
19. Security / Manpower Supply
20. Telecasting / Broadcasting
21. Utility / Subscriptions
22. Professional Fees (Other)
23. Event Management
24. Medical / Healthcare Services
25. Others / Miscellaneous

Payment Description: {payment_description}

Analyze the payment description and respond in EXACTLY this JSON format:
{{
    "primary_service": "Category name from the list above",
    "all_detected_services": ["Category 1", "Category 2"]
}}

Examples:
- For "Sewing machine and Aluminum foils" → Primary: "Goods Sale", All: ["Goods Sale"]
- For "Azure usage charges" → Primary: "Technical Service / SaaS / Software", All: ["Technical Service / SaaS / Software"]
- For "Legal services" → Primary: "Legal / Compliance", All: ["Legal / Compliance"]
"""


def extract_text_from_image(image: Image.Image) -> str:
    """Runs OCR on a PIL image"""
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Converts PDF to images and applies OCR"""
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = "\n\n".join(extract_text_from_image(page) for page in pages)
    return full_text


def extract_invoice_details_with_llm(invoice_text: str) -> dict:
    """LLM-based invoice field extraction"""
    try:
        prompt = INVOICE_EXTRACTION_PROMPT.format(invoice_text=invoice_text[:4000])
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt})

        try:
            # More robust JSON cleaning
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Find JSON object if it's embedded in other text
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start != -1 and end != 0:
                cleaned = cleaned[start:end]
            
            invoice_data = json.loads(cleaned)
            
            # Ensure all required fields exist
            required_fields = ["state", "city", "amount", "payment_type", "vendor_name", "invoice_number", "date", "confidence"]
            for field in required_fields:
                if field not in invoice_data:
                    invoice_data[field] = "Not found"
            
        except Exception as e:
            print(f"JSON parsing error for invoice extraction: {e}")
            print(f"Raw response: {response}")
            invoice_data = {
                "state": "Not found",
                "city": "Not found",
                "amount": "Not found",
                "payment_type": "General payment",
                "vendor_name": "Not found",
                "invoice_number": "Not found",
                "date": "Not found",
                "confidence": 50,
            }

        return invoice_data

    except Exception as e:
        print(f"Invoice extraction error: {e}")
        return {
            "state": "Error in extraction",
            "city": "Error in extraction",
            "amount": "Error in extraction",
            "payment_type": "Error in extraction",
            "vendor_name": "Error in extraction",
            "invoice_number": "Error in extraction",
            "date": "Error in extraction",
            "confidence": 0,
        }


def classify_service_from_payment_type(payment_description: str) -> dict:
    """LLM-based service category classification"""
    try:
        # Skip if payment description is empty or error
        if not payment_description or payment_description in ["Not found", "Error in extraction", "General payment"]:
            return {
                "primary_service": "Others / Miscellaneous",
                "all_detected_services": ["Others / Miscellaneous"]
            }
        
        prompt = SERVICE_CLASSIFICATION_PROMPT.format(payment_description=payment_description)
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt})

        try:
            # More robust JSON cleaning
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Find JSON object if it's embedded in other text
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start != -1 and end != 0:
                cleaned = cleaned[start:end]
            
            service_data = json.loads(cleaned)
            
            # Ensure required fields exist
            if "primary_service" not in service_data:
                service_data["primary_service"] = "Others / Miscellaneous"
            if "all_detected_services" not in service_data:
                service_data["all_detected_services"] = [service_data["primary_service"]]
                
        except Exception as e:
            print(f"JSON parsing error for service classification: {e}")
            print(f"Raw response: {response}")
            print(f"Payment description: {payment_description}")
            
            # Fallback classification based on keywords
            service_data = classify_by_keywords(payment_description)

        return service_data

    except Exception as e:
        print(f"Service classification error: {e}")
        return {
            "primary_service": "Others / Miscellaneous",
            "all_detected_services": ["Others / Miscellaneous"]
        }


def classify_by_keywords(payment_description: str) -> dict:
    """Fallback keyword-based classification"""
    description_lower = payment_description.lower()
    
    # Define keyword mappings
    keyword_mappings = {
        "Goods Sale": ["machine", "product", "item", "goods", "sale", "purchase", "bag", "phone", "computer", "headphone", "furniture", "cream", "pipe", "rubber", "shirt", "foil"],
        "Technical Service / SaaS / Software": ["azure", "software", "cloud", "subscription", "app", "web development", "design", "creative cloud", "freshservice"],
        "Passenger Transport": ["transport", "taxi", "ride", "bus", "flight", "passenger"],
        "Goods Transport / Logistics": ["shipping", "freight", "courier", "delivery", "logistics"],
        "Legal / Compliance": ["legal", "counsel", "litigation", "trademark", "hearing", "drafting", "gst registration"],
        "Maintenance / AMC": ["repair", "maintenance", "amc", "service"],
        "Professional Fees (Other)": ["professional services", "consultation", "fees"]
    }
    
    for category, keywords in keyword_mappings.items():
        if any(keyword in description_lower for keyword in keywords):
            return {
                "primary_service": category,
                "all_detected_services": [category]
            }
    
    return {
        "primary_service": "Others / Miscellaneous",
        "all_detected_services": ["Others / Miscellaneous"]
    }


def process_invoice_file(file_path: str) -> dict:
    """Main function to handle PDF or image invoices"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file_path)
        text = extract_text_from_image(image)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")

    invoice_data = extract_invoice_details_with_llm(text)
    service_info = classify_service_from_payment_type(invoice_data.get("payment_type", ""))
    return {**invoice_data, **service_info}


# Main processing loop
folder_path = "Invoice_samples"  # Your folder path

# Supported file types
valid_extensions = [".pdf", ".jpg", ".jpeg", ".png"]

print("Processing invoices...")
print("-" * 100)

for filename in os.listdir(folder_path):
    # Check file extension
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue  # Skip unsupported files

    full_path = os.path.join(folder_path, filename)
    try:
        print(f"Processing {filename}...")
        result = process_invoice_file(full_path)
        print(f"✓ {filename}: Primary Service = {result.get('primary_service', 'Unknown')}")
        print(f"  Payment Type: {result.get('payment_type', 'Unknown')}")
        print("-" * 50)
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        print("-" * 50)