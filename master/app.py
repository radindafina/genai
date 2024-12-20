output_folder = "output_images"

import streamlit as st
from PIL import Image
import os
from openai import AzureOpenAI
import fitz  # PyMuPDF
import shutil
import base64
import re

# Replace with your Azure OpenAI details
AZURE_API_KEY = "6a13a95bf6774542a24b438b9d98dd42"
AZURE_ENDPOINT = "https://my-dna-openai-2.openai.azure.com/"
AZURE_DEPLOYMENT_ID = "2024-08-01-preview"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  
def pdf_to_images(pdf_file_bytes, output_folder, dpi=300):
    # Open PDF directly using bytes
    pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)

        # Save the image to the output folder
        image_filename = f"{output_folder}/page_{page_number + 1}.png"
        pix.save(image_filename)

    pdf_document.close()


# Function to convert images in a folder to Base64
def get_base64_images(directory):
    base64_images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            with open(os.path.join(directory, filename), 'rb') as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                base64_images.append(base64_image)
    return base64_images

def extract_text_from_pdf(uploaded_file):
    text = ""
    if uploaded_file is not None:
        # Read the uploaded file only once and store the bytes
        uploaded_file_bytes = uploaded_file.getvalue()

        if not uploaded_file_bytes:
            raise ValueError("The uploaded file is empty.")

        # Open the PDF using PyMuPDF from the byte stream
        pdf_document = fitz.open(stream=uploaded_file_bytes, filetype="pdf")

        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")

    return text


# Function to create the message structure for Azure OpenAI
def create_message(system_prompt, user_prompt, base64_images):
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    
    for base64_image in base64_images:
        message[1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}
        })
    return message

# Function to analyze the document with Azure OpenAI
def analyzing_document_with_prompt(system_prompt, user_prompt, message_structure):
    llm = AzureOpenAI(
        api_version=AZURE_DEPLOYMENT_ID,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )
    response = llm.chat.completions.create(
        messages=message_structure,
        model="gpt-4o",
        temperature=0,
        max_tokens=4000
    )
    extracted_content = response.choices[0].message.content

    # Split content into sections using regex
    sections = re.split(r"(Section [A-Z]+:)", extracted_content)

    # Display results section by section
    for i in range(1, len(sections), 2):
        section_header = sections[i].strip()
        section_content = sections[i + 1].strip()

        st.subheader(section_header)
        if "|" in section_content:
            st.markdown(section_content)
        else:
            st.write(section_content)  

# Streamlit App
st.title("Extraction of MPC file(s)")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF...", type=["pdf"])

if uploaded_file is not None:
    uploaded_file_bytes = uploaded_file.getvalue()  # Read file bytes once

    # Pass the file content to PDF processing
    pdf_to_images(uploaded_file_bytes, "output_images", dpi=300)
    pdf_text = extract_text_from_pdf(uploaded_file)  # Process the bytes

    st.write("Extracted Text from PDF:")
    st.write(pdf_text)
    
    # User prompt input
    prompt = st.text_input("Enter prompt for analysis")
    
    if st.button("Analyze"):
        st.info("Analyzing document...")
        
        # Base64-encode the extracted images
        base64_images = get_base64_images(output_folder)
        
        system_prompt = (
            "You are a data extraction specialist skilled at extracting and organizing information "
            "into structured tables. Your task is to carefully analyze and extract all available "
            "information from the provided images, no matter the format or structure."
        )
        
        user_prompt = """
        Please analyze the provided images and extract all available data.

        1. Group the data into sections (e.g., Section A, Section B, Section C) based on content and context.
        2. Each section must have a clear and precise header indicating its content.
        3. Under each section, provide a table with rows and columns for the extracted data. If data cannot be read or extracted due to quality issues, label it as 'Unreadable'.
        4. If the data has varying formats, adapt the extraction approach accordingly. For example:
            - If the data is presented as a list, structure it in a table with appropriate column labels.
            - If the data is fragmented or unclear, still attempt to extract it to the best of your ability, indicating any missing or unclear parts as 'Unreadable'.
        5. For any data that is not recognized or legible, write 'Unreadable' and explain in detail why the data could not be extracted.


        Format example for sections:

        **Section A:**
        | Column 1  | Column 2  | Column 3  |
        |-----------|-----------|-----------|
        | Value 1   | Value 2   | Value 3   |
        | Value 4   | Value 5   | Value 6   |

        **Section B:**
        | Column 1  | Column 2  | Column 3  |
        |-----------|-----------|-----------|
        | Value 7   | Value 8   | Unreadable|
        | Value 9   | Unreadable| Value 10  |
        """

        # Create the message structure for Azure OpenAI
        message_structure = create_message(system_prompt, user_prompt, base64_images)
        
        # Analyze the document
        analyzing_document_with_prompt(system_prompt, user_prompt, message_structure)
        

