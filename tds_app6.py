import streamlit as st
from bfs_backtrack import bfs_search
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import docx
import tempfile
import json
import re
from datetime import datetime
import numpy as np
import cv2

# Load and verify environment variables
load_dotenv()
required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Simplified OCR Extraction Prompt - NO TDS Classification
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


# Enhanced TDS Analysis Prompt - Using the accurate persona from tds_app.py
TDS_ANALYSIS_PROMPT = """
Role: You are a tax professional who is an expert in Tax Deducted at Source i.e. TDS provisions under Indian Income Tax Act.
Your capabilities include explaining TDS provisions, analyzing the transactions and suggesting the most appropriate TDS Section applicable to that transaction.
You serve individual taxpayers, small business owners, and tax professionals, offering a quick reference to TDS provisions and will solve all queries pertaining to TDS provisions
You address misunderstandings by referencing the updated TDS provisions, offering examples for clarity, and directing to professional advice for intricate matters.
If any question falls outside your domain, you will respond with "I am sorry. This question is not my expertise. Do you have any other queries pertaining to TDS provisions?"
If the query of the user is confusing or not clear, ask for more clarity by saying "Could you please rephrase that or provide more details? I'm not sure I understand what you're asking."
Your knowledge is based on the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment.pdf" file uploaded in the 'Knowledge Centre'
While giving answers, please provide Confidence Score in the end.
Also provide the details about the source basis which the answer is derived from. If there are multiple sources, show all sources as a list. Also, if the user has uploaded any document or image, then please refer that document/image as one of the sources.
At the end, ask the user "Do you need any further information on the above response?"

The detailed tasks required to be done while answering any query and question asked by the user are as under:
Task 1 - If the User asks any question pertaining to applicability or non-applicability of TDS, you will use the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment.pdf" file available in the Knowledge Centre to analyze and identify the initial TDS Section.
Task 2 - If the user has uploaded any document (such as Invoice, Purchase Order or Contract/Agreement), you will perform the following tasks:
​- Go through the entire document and identify "Description of Goods / Services" i.e. the goods or services for which the invoice is created
​- Analyze the "Description of Good/Services" and use the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment" file available in the Knowledge Centre to analyze and identify the initial TDS Section.
Task 3 - If the Initial TDS Section is either Section 194J or Section 194C or Section 194H or Section 194I or No TDS, then please refer to these data sources for additional information about taxability:
​- For Section 194J, refer file 'TDS Paper - Section 194J - v1 - 27.03.2025.docx' uploaded in the 'Knowledge Centre'
​- For Section 194C, refer file 'TDS Paper - Section 194C- v1 - 0402.2025.docx' uploaded in the 'Knowledge Centre'
​- For Section 194H, refer file 'TDS Paper - Section 194H - v1 - 0603.2025.docx' uploaded in the 'Knowledge Centre'
​- For Section 194I, refer file 'TDS Paper - Section 194I - v1 - 27.03.2025.docx' uploaded in the 'Knowledge Centre'
​- For No TDS, refer file 'TDS Paper - NO TDS - v1 - 28.03.2025.docx' uploaded in the 'Knowledge Centre'

Some KEY CONSIDERATIONS while answering the query:
1. CONTEXT-DRIVEN RESPONSES: You must base your responses strictly on the provided context and the documents without relying on pre-existing knowledge or external open-source/online information.
2. THRESHOLD ASSUMPTION: Assume that all applicable threshold limits for deductors and deductees are met, making TDS provisions applicable.
3. SUB-SECTION BIFURCATION:
- If Section 194J is applicable, specify the correct sub-section based on the nature of the service:
- 194J(1)(a): Fees for professional services
- 194J(1)(b): Fees for technical services
- 194J(1)(ba): Any remuneration or fees or commission by whatever name called, other than those on which tax is deductible under section 192 (TDS on Salary), to a director of a company
- 194J(1)(c): Payment of royalty for sale, distribution or exhibition of cinematographic films.
- 194J(1)(d): Non-compete fees and other compensation in connection with business or profession.
- If Section 194I is applicable, specify the correct sub-section based on the type of asset rented:
- 194I(1)(a): Rent on plant, machinery, or equipment.
- 194I(1)(b): Rent on land, building, furniture.
4. GOODS-BASED IDENTIFICATION (SECTION 194Q):
- If the description explicitly contains any goods name, the term "goods," or "purchase of goods" or “Supply of goods” (without any services included), Section 194Q should be applied directly without evaluating other TDS sections.
5. ALTERNATIVE TDS SECTIONS: If another TDS section is potentially applicable, suggest it along with justification, enabling users to make an informed decision.
6. ACCURACY & VERIFICATION: Cross-verify all responses against the given context to ensure correctness, completeness, and freedom from hallucinations, errors, or omissions.

DESIRED OUTPUT FORMAT
1.​STRICTLY SHOW THE OUTPUT IN THE FOLLOWING FORMAT:
First visualize the answer in the following JSON Format and convert it into a table markdown format for the user and give definite answers only
[
{{
"Sr. No.": "1",
"Goods or Services": "Salary Income",
"Brief Explanation of Goods or services": "It refers to Salary income received by an employee during the course of employment.",
"TDS Section": "Applicable TDS section (specify sub-section wherever applicable. i.e., instead of 194J, specify 194J(1)(a), 194J(1)(b), 194J(1)(ba), 194J(1)(c), or 194J(1)(d) and instead of 194I, specify 194I(1)(a) or 194I(1)(b))",
"TDS Rate": "Applicable TDS rate (%)",
"Probability TDS Section": "Probability percentage of this TDS section being applicable",
"Reason TDS Section": "Justification/conditions for the TDS section's applicability. The reason for selecting a specific TDS Section should be sensible and justifiable, ensuring logical consistency with the given context and relevant tax provisions.",
"Alt TDS Section": "Alternate applicable TDS section (if any)",
"Alt TDS Rate": "TDS rate under the alternate section",
"Prob Alt TDS Section": "Probability percentage of the alternate TDS section being applicable",
"Reason Alt TDS Section": "Justification/conditions for the alternate TDS section's applicability. The reason for selecting a specific alternate TDS Section should be sensible and justifiable, ensuring logical consistency with the given context and relevant tax provisions."
}},
{{
"Sr. No.": "2",
"Goods or Services": "Reimbursement of Expenses",
"Brief Explanation of Goods or services": "It refers to reimbursement of expenses incurred by employee",
"TDS Section": "Applicable TDS section (specify sub-section wherever applicable. i.e., instead of 194J, specify 194J(1)(a), 194J(1)(b), 194J(1)(ba), 194J(1)(c), or 194J(1)(d) and instead of 194I, specify 194I(1)(a) or 194I(1)(b))",
"TDS Rate": "Applicable TDS rate (%)",
"Probability TDS Section": "Probability percentage of this TDS section being applicable",
"Reason TDS Section": "Justification/conditions for the TDS section's applicability. The reason for selecting a specific TDS Section should be sensible and justifiable, ensuring logical consistency with the given context and relevant tax provisions.",
"Alt TDS Section": "Alternate applicable TDS section (if any)",
"Alt TDS Rate": "TDS rate under the alternate section",
"Prob Alt TDS Section": "Probability percentage of the alternate TDS section being applicable",
"Reason Alt TDS Section": "Justification/conditions for the alternate TDS section's applicability. The reason for selecting a specific alternate TDS Section should be sensible and justifiable, ensuring logical consistency with the given context and relevant tax provisions."
}}
]
2. Do not show the JSON output to the user. Only populate the Final Table format
3. ONLY IN AN EXTREME SITUATION WHERE THE QUERY OR QUESTION ASKED BY THE USER IS SUCH THAT IT CANNOT BE SHOWN IN THE ABOVE FORMAT, YOU CAN ANSWER IN ANY FORMAT

User Query:
{query}

Context Chunks:
{context}

{invoice_info}
"""

GENERAL_TDS_PROMPT = """
Role: You are a knowledgeable tax professional expert in TDS provisions under the Indian Income Tax Act.

IMPORTANT RESTRICTIONS:
1. If the query is unrelated to TDS/taxes, respond: "I don't have enough knowledge to answer that question. However, I'm here to help with any TDS-related queries you might have!"
2. For simple conversational queries, respond naturally and offer help with TDS.
3. Only answer TDS-related questions using provided context.
4. Provide responses in clear paragraph format, NOT in JSON or table format.

Context from TDS Knowledge Base:
{context}

Invoice/Document Information (if available):
{invoice_info}

Conversation History:
{conversation_history}

User Query: {query}

Please provide a helpful and accurate response in paragraph format based on the context and your expertise.
If invoice information is available, use it to provide more specific guidance.
"""

NON_TDS_DETECTION_PROMPT = """
Analyze this query and determine if it's related to TDS, taxation, compliance, or accounting.

Query: {query}

Respond with one word:
- "TDS" if related to TDS/taxation
- "CONVERSATIONAL" if a simple greeting
- "UNRELATED" if completely unrelated
Response:
"""

# Azure OpenAI config - Updated to match invoice_processor.py
try:
    from model import llm
except ImportError:
    # Fallback if model.py is not available
    llm = AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-10-21"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
    )

def advanced_preprocess_for_ocr(pil_image):
    """Advanced preprocessing for OCR: deskew, denoise, threshold, morph open."""
    img = np.array(pil_image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Deskew
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Denoise and threshold
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    # Morphological opening to remove small noise
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(img)

def enhance_image_for_ocr(image):
    try:
        if image.mode!='L':
            image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5)).filter(ImageFilter.SHARPEN)
        return image
    except:
        return image

def full_ocr_pipeline(pil_image):
    enhanced = enhance_image_for_ocr(pil_image)
    final = advanced_preprocess_for_ocr(enhanced)
    return final

def extract_text_from_pdf(pdf_path: str) -> str:
    """Converts PDF to images and applies OCR - matches invoice_processor.py"""
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = "\n\n".join(pytesseract.image_to_string(page) for page in pages)
    return full_text

def extract_text_from_image(image: Image.Image) -> str:
    """Runs OCR on a PIL image - matches invoice_processor.py"""
    return pytesseract.image_to_string(image)


def get_context_for_query(query, max_docs=20, threshold=0.5):
    docs = bfs_search(query, threshold=threshold, max_docs=max_docs)
    context = []
    for doc in docs:
        meta = doc.metadata
        ref = f"{meta.get('source_file','unknown')} (chunk {meta.get('chunk_index',meta.get('cluster','-'))})"
        context.append(f"[{ref}]: {doc.page_content[:500]}")
    return "\n\n".join(context)

def is_explanation_query(query):
    """Check if user is asking for explanations/reasoning - these should be in paragraph format"""
    explanation_keywords = ['explain','why','how','reason','reasoning','justification','because','what is the logic','tell me about','describe','elaborate']
    q = query.lower()
    return any(k in q for k in explanation_keywords)

def is_tds_query(query):
    """Check if query is TDS-related - most TDS queries should default to table format"""
    tds_keywords = ['tds','tax','deduction','section','rate','provision','compliance','audit','fees','professional','payment','invoice','bill']
    q = query.lower()
    return any(k in q for k in tds_keywords)

def detect_query_type(query):
    try:
        prompt = NON_TDS_DETECTION_PROMPT.format(query=query)
        chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
        resp = chain.invoke({"prompt": prompt}).strip().upper()
        return resp if resp in ["TDS","CONVERSATIONAL","UNRELATED"] else "TDS"
    except:
        return "TDS"

def format_conversation_history(messages):
    history = []
    for msg in messages[-6:]:
        role = "User" if msg["role"]=="user" else "Assistant"
        history.append(f"{role}: {msg['content'][:300]}...")
    return "\n".join(history)

def extract_invoice_text(uploaded_file):
    try:
        ftype = uploaded_file.type
        if ftype.startswith('image'):
            image = Image.open(uploaded_file)
            # Use the same logic as invoice_processor.py
            text = extract_text_from_image(image)
            return text, 'image'
        elif ftype=='application/pdf':
            with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp:
                tmp.write(uploaded_file.read()); tmp.flush()
                # Use the same PDF processing as invoice_processor.py
                text = extract_text_from_pdf(tmp.name)
                os.unlink(tmp.name)  # Clean up temp file
            return text, 'pdf'
        elif ftype in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document','application/msword']:
            with tempfile.NamedTemporaryFile(delete=False,suffix='.docx') as tmp:
                tmp.write(uploaded_file.read()); tmp.flush()
                doc = docx.Document(tmp.name)
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                os.unlink(tmp.name)  # Clean up temp file
            return text, 'docx'
        elif ftype=='text/plain':
            return uploaded_file.read().decode(errors="ignore"), 'txt'
        else:
            return "[Unsupported file type]", 'unknown'
    except Exception as e:
        return f"[Extraction Error: {e}]", 'unknown'

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
        "Legal / Compliance": ["legal", "counsel", "litigation", "trademark", "hearing", "drafting", "gst registration", "examination report", "legal services", "attorney", "lawyer", "compliance", "regulatory", "patent", "copyright", "order copy"],
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


def process_invoice_file(uploaded_file) -> dict:
    """Main function to handle uploaded files - matches invoice_processor.py logic"""
    try:
        # Extract text from uploaded file
        extracted_text, file_type = extract_invoice_text(uploaded_file)

        # Get invoice details using LLM (same as invoice_processor.py)
        invoice_data = extract_invoice_details_with_llm(extracted_text)

        # Classify service from payment type (same as invoice_processor.py)
        service_info = classify_service_from_payment_type(invoice_data.get("payment_type", ""))

        # Combine all information (same as invoice_processor.py)
        result = {**invoice_data, **service_info}
        return result

    except Exception as e:
        print(f"Error processing file: {str(e)}")  # Use print instead of st.error for consistency
        return {
            "state": "Error in processing",
            "city": "Error in processing", 
            "amount": "Error in processing",
            "payment_type": "Error in processing",
            "vendor_name": "Error in processing",
            "invoice_number": "Error in processing",
            "date": "Error in processing",
            "confidence": 0,
            "primary_service": "Others / Miscellaneous",
            "all_detected_services": ["Others / Miscellaneous"]
        }

def format_invoice_info(invoice_data: dict) -> str:
    """Format invoice and classification information for LLM or user-friendly display"""
    if not invoice_data:
        return "No invoice information available."

    # Extract service classification
    primary_service = invoice_data.get('primary_service', 'Unknown')
    all_services = invoice_data.get('all_detected_services', [])

    # Format list into comma-separated string
    all_services_str = ", ".join(all_services) if all_services else "None detected"

    info = f"""
**Invoice / Document Summary:**

- Vendor: **{invoice_data.get('vendor_name', 'Not available')}**
- Invoice Number: {invoice_data.get('invoice_number', 'Not available')}
- Date: {invoice_data.get('date', 'Not available')}
- Location: {invoice_data.get('city', 'Not available')}, {invoice_data.get('state', 'Not available')}
- Amount: {invoice_data.get('amount', 'Not available')}
- Payment Description: {invoice_data.get('payment_type', 'Not available')}
- Primary Service Category: **{primary_service}**
- All Detected Services: {all_services_str}
- Extraction Confidence: {invoice_data.get('confidence', 'Not available')}%
"""
    return info.strip()


def format_tds_table(response):
    """Enhanced table formatting with better styling and detailed explanations"""
    # Clean up the response
    cleaned_response = response.strip()

    # Parse the table to extract data for enhanced formatting
    lines = cleaned_response.split('\n')
    table_started = False
    table_rows = []
    other_content = []

    for line in lines:
        if '|' in line and not line.strip().startswith('|---'):
            table_started = True
            table_rows.append(line.strip())
        elif not table_started:
            other_content.append(line)
        else:
            if line.strip():  # Non-empty line after table
                other_content.append(line)

    # Build enhanced response
    enhanced_response = "## 📊 **TDS Analysis Results**\n\n"

    if table_rows:
        # Enhanced table with better styling (pure markdown)

        # Process table rows for better formatting
        for i, row in enumerate(table_rows):
            if i == 0:  # Header row
                # Make header row more prominent
                cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Remove empty first/last
                enhanced_response += "| " + " | ".join([f"**{cell}**" for cell in cells]) + " |\n"
                # Enhanced separator with better alignment
                enhanced_response += "|" + "|".join([":---:" for _ in cells]) + "|\n"
            else:
                # Data rows with improved cell formatting
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                formatted_cells = []

                for j, cell in enumerate(cells):
                    if j == 0:  # Sr. No. column
                        formatted_cells.append(f"**{cell}**")
                    elif j == 1:  # Goods/Services column
                        formatted_cells.append(f"🔹 {cell}")
                    elif "Section" in table_rows[0].split('|')[j+1]:  # TDS Section columns
                        if cell and cell != "N/A" and cell != "-":
                            formatted_cells.append(f"**`{cell}`**")
                        else:
                            formatted_cells.append(cell)
                    elif "%" in cell or "rate" in table_rows[0].split('|')[j+1].lower():
                        if cell and cell != "N/A" and cell != "-":
                            formatted_cells.append(f"**{cell}**")
                        else:
                            formatted_cells.append(cell)
                    else:
                        formatted_cells.append(cell)

                enhanced_response += "| " + " | ".join(formatted_cells) + " |\n"

        enhanced_response += "\n"

        # Add detailed explanations section
        enhanced_response += "---\n\n"
        enhanced_response += "## 📖 **Detailed Analysis & Legal Interpretation**\n\n"

        # Extract key information from table for detailed explanation
        if len(table_rows) > 1:
            for i in range(1, len(table_rows)):
                cells = [cell.strip() for cell in table_rows[i].split('|')[1:-1]]
                if len(cells) >= 4:
                    sr_no = cells[0] if len(cells) > 0 else "N/A"
                    goods_services = cells[1] if len(cells) > 1 else "N/A"
                    explanation = cells[2] if len(cells) > 2 else "N/A"
                    tds_section = cells[3] if len(cells) > 3 else "N/A"
                    tds_rate = cells[4] if len(cells) > 4 else "N/A"
                    probability = cells[5] if len(cells) > 5 else "N/A"
                    reason = cells[6] if len(cells) > 6 else "N/A"
                    alt_section = cells[7] if len(cells) > 7 else "N/A"
                    alt_rate = cells[8] if len(cells) > 8 else "N/A"
                    alt_probability = cells[9] if len(cells) > 9 else "N/A"
                    alt_reason = cells[10] if len(cells) > 10 else "N/A"

                    enhanced_response += f"### 🔍 **Entry {sr_no}: {goods_services}**\n\n"

                    # Transaction Overview
                    enhanced_response += f"**📋 Transaction Overview:**\n"
                    enhanced_response += f"- **Type:** {goods_services}\n"
                    enhanced_response += f"- **Description:** {explanation}\n\n"

                    # Primary TDS Analysis
                    enhanced_response += f"**⚖️ Primary TDS Recommendation:**\n"
                    enhanced_response += f"- **Section:** {tds_section}\n"
                    enhanced_response += f"- **Rate:** {tds_rate}\n"
                    enhanced_response += f"- **Confidence:** {probability}\n"
                    enhanced_response += f"- **Legal Justification:** {reason}\n\n"

                    # Alternative Analysis (if available)
                    if alt_section and alt_section != "N/A" and alt_section != "-" and alt_section.lower() != "none":
                        enhanced_response += f"**🔄 Alternative TDS Option:**\n"
                        enhanced_response += f"- **Section:** {alt_section}\n"
                        enhanced_response += f"- **Rate:** {alt_rate}\n"
                        enhanced_response += f"- **Confidence:** {alt_probability}\n"
                        enhanced_response += f"- **Legal Justification:** {alt_reason}\n\n"
                    else:
                        enhanced_response += f"**🔄 Alternative Option:** No alternate section applicable for this transaction type.\n\n"

                    if i < len(table_rows) - 1:
                        enhanced_response += "---\n\n"



    else:
        # If no table found, return original response with enhancements
        enhanced_response += cleaned_response + "\n\n"

    # Add footer
    if "Do you need any further information" not in enhanced_response:
        enhanced_response += "---\n\n"
        enhanced_response += "💡 **Need more clarification?** Feel free to ask follow-up questions about:\n"
        enhanced_response += "- Specific TDS sections or rates\n"
        enhanced_response += "- Compliance requirements\n"
        enhanced_response += "- Document requirements\n"
        enhanced_response += "- Special cases or exemptions\n"

    return enhanced_response

def run_tds_analysis(query, conversation_history, invoice_data=None):
    # 1️⃣ Detect query type
    qtype = detect_query_type(query)
    if qtype=="UNRELATED":
        return "I don't have enough knowledge to answer that question. However, I'm here to help with any TDS-related queries you might have!"
    if qtype=="CONVERSATIONAL":
        greetings = [
            "Hello! I'm your TDS expert assistant.",
            "Hi there! How can I help you with TDS matters today?",
            "Great! I'm here to help you with any TDS-related questions.",
            "Good to see you! What TDS query can I assist with?"
        ]
        return f"{greetings[len(conversation_history)%len(greetings)]} Feel free to ask me about TDS rates, sections, compliance, or upload documents for analysis."

    # 2️⃣ RAG context from vector database
    context = get_context_for_query(query)

    # 3️⃣ Format invoice information
    invoice_info = format_invoice_info(invoice_data)

    # 4️⃣ Enhance query with invoice payment type if available
    enhanced_query = query
    if invoice_data and invoice_data.get('payment_type') and invoice_data['payment_type'] != "Not found":
        enhanced_query = f"[Payment Type from Invoice: {invoice_data['payment_type']}] {query}"

    # 5️⃣ Choose prompt and ensure proper formatting
    # Most TDS queries should be in table format, only explanations in paragraph format
    if is_explanation_query(enhanced_query):
        # Explanation queries - Use paragraph format
        prompt = GENERAL_TDS_PROMPT.format(
            context=context,
            invoice_info=invoice_info,
            conversation_history=format_conversation_history(conversation_history),
            query=enhanced_query
        )
        response = (ChatPromptTemplate.from_template("{prompt}")
                | llm | StrOutputParser()).invoke({"prompt": prompt})
        # Ensure it's in paragraph format (not JSON or table)
        return response
    elif is_tds_query(enhanced_query):
        # Most TDS queries - Use table format by default
        prompt = TDS_ANALYSIS_PROMPT.format(
            query=enhanced_query, context=context, invoice_info=invoice_info
        )
        response = (ChatPromptTemplate.from_template("{prompt}") 
                    | llm | StrOutputParser()).invoke({"prompt": prompt})
        # ALWAYS format as table, never show raw JSON
        return format_tds_table(response)
    else:
        # General queries - paragraph format
        prompt = GENERAL_TDS_PROMPT.format(
            context=context,
            invoice_info=invoice_info,
            conversation_history=format_conversation_history(conversation_history),
            query=enhanced_query
        )
        response = (ChatPromptTemplate.from_template("{prompt}")
                | llm | StrOutputParser()).invoke({"prompt": prompt})
        return response

# Streamlit UI
st.set_page_config(page_title="TDS Chatbot - Expert Assistant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "invoice_data" not in st.session_state:
    st.session_state.invoice_data = None
if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False

st.title("🤖 TDS Expert Chatbot")
st.markdown("Ask me anything about TDS provisions, rates, sections, or compliance!")

# File upload section integrated into the main chat interface
if not st.session_state.invoice_data and not st.session_state.processing_upload:
    with st.container():
        st.markdown("### 📄 Upload Invoice/Document (Optional)")
        uploaded_file = st.file_uploader(
            "Upload a document to extract payment details for TDS analysis",
            type=["pdf","txt","docx","png","jpg","jpeg"],
            key="main_file_uploader"
        )

        if uploaded_file is not None:
            st.session_state.processing_upload = True
            with st.spinner("🔍 Extracting payment and service details from document..."):
                # Process the uploaded file directly
                result = process_invoice_file(uploaded_file)

                # Store in session state
                st.session_state.invoice_data = result

            st.success("✅ Document processed successfully!")

            # Display extracted information
            with st.expander("📋 Extracted Information", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**📍 Basic Details:**")
                    st.write(f"Vendor: {result.get('vendor_name', 'Not found')}")
                    st.write(f"Invoice Number: {result.get('invoice_number', 'Not found')}")
                    st.write(f"Date: {result.get('date', 'Not found')}")
                    st.write(f"Amount: {result.get('amount', 'Not found')}")
                    st.write(f"Primary Service: {result.get('primary_service', 'Unknown')}")

                with col2:
                    st.write("**📍 Location & Services:**")
                    st.write(f"State: {result.get('state', 'Not found')}")
                    st.write(f"City: {result.get('city', 'Not found')}")
                    st.write(f"Services/Goods: {result.get('payment_type', 'Not found')}")
                    st.write(f"Detected Services: {', '.join(result.get('all_detected_services', [])) or 'None'}")
                    st.write(f"Confidence: {result.get('confidence', 'Not found')}%")

            st.info("💡 **Next Step:** Ask me about TDS rates or sections for these services, and I'll analyze using the legal knowledge base!")

            # Show preview of uploaded file
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Uploaded Document", width=400)
            else:
                with st.expander("📄 View Extracted Text"):
                    extracted_text, _ = extract_invoice_text(uploaded_file)
                    st.text_area("Extracted Content", extracted_text[:1500], height=200, disabled=True)

            st.session_state.processing_upload = False
            st.rerun()

if st.session_state.get("invoice_data"):
        invoice = st.session_state.invoice_data

        with st.container():
            st.markdown("### 📄 Current Document Context")
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.write(f"**Services/Goods:** {invoice.get('payment_type', 'Not available')}")
                st.write(f"**Primary Service:** {invoice.get('primary_service', 'Not available')}")
                st.write(f"**Vendor:** {invoice.get('vendor_name', 'Not available')}")

            with col2:
                st.write(f"**Amount:** {invoice.get('amount', 'Not available')}")
                st.write(f"**Location:** {invoice.get('city', 'Not available')}, {invoice.get('state', 'Not available')}")
                detected = ", ".join(invoice.get("all_detected_services", [])) or "None"
                st.write(f"**Detected Services:** {detected}")

            with col3:
                if st.button("🗑️ Clear Document"):
                    st.session_state.invoice_data = None
                    st.rerun()

            st.markdown("---")


# Chat interface with scrollable container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your TDS question..."):
        # Add user message
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query using TDS knowledge base..."):
                response = run_tds_analysis(
                    prompt,
                    st.session_state.messages,
                    st.session_state.invoice_data
                )
            st.markdown(response)

        # Add assistant response
        st.session_state.messages.append({"role":"assistant","content":response})

# Quick start section (only show if no messages)
if not st.session_state.messages:
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 TDS Rates Overview"):
            st.session_state.messages.append({"role":"user","content":"What are the different TDS rates for various types of payments?"})
            st.rerun()

    with col2:
        if st.button("📋 TDS Compliance"):
            st.session_state.messages.append({"role":"user","content":"What are the key TDS compliance requirements I should know?"})
            st.rerun()

    with col3:
        if st.button("🔍 Section 194C Analysis"):
            st.session_state.messages.append({"role":"user","content":"Analyze TDS section and rate for contractor payments"})
            st.rerun()

# Sidebar with additional controls and information
with st.sidebar:
    st.header("🛠️ Chat Controls")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("📋 Current Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Document Uploaded:** {'✅ Yes' if st.session_state.invoice_data else '❌ No'}")

    if st.session_state.invoice_data:
        payment_type = st.session_state.invoice_data.get('payment_type', 'N/A')
        if payment_type and len(str(payment_type)) > 30:
            payment_type = str(payment_type)[:30] + "..."
        st.write(f"**Services/Goods:** {payment_type}")
        st.write(f"**Confidence:** {st.session_state.invoice_data.get('confidence', 0)}%")

    st.markdown("---")

    st.subheader("🔄 Workflow")
    st.markdown("""
    **Enhanced Two-Step Process:**

    1. **📄 Document Upload & Extraction**
       - OCR extracts factual data only
       - No TDS classification at this stage
       - Focus on payment type, amount, location

    2. **🎯 TDS Analysis via Query**
       - Vector database provides legal context
       - LLM analyzes payment type against TDS rules
       - Accurate section & rate determination
    """)

    st.markdown("---")

    st.subheader("💡 Key Features")
    st.markdown("""
    - **🔍 Smart OCR**: Extracts payment details without premature TDS classification
    - **📊 Vector-based Analysis**: Uses legal knowledge base for accurate TDS classification
    - **🧠 Context Retention**: Remembers invoice details throughout conversation
    - **💬 Natural Chat**: Conversational interface with structured analysis
    - **📄 Multi-format Support**: PDF, DOCX, images, and text files
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "💼 Simplified TDS Expert Assistant - OCR + Vector DB Analysis | "
    "📚 Based on Income Tax Act 1961 - TDS Provisions 2024"
    "</div>",
    unsafe_allow_html=True
)