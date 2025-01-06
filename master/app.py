output_folder = "output_images"

import streamlit as st
from PIL import Image
import os
import pandas as pd
import fitz
import shutil
import base64
import re
from openai import AzureOpenAI

AZURE_API_KEY = st.secrets["azure"]["api_key"]
AZURE_ENDPOINT = st.secrets["azure"]["endpoint"]
AZURE_DEPLOYMENT_ID = st.secrets["azure"]["deployment_id"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pdf_to_images(pdf_path, output_folder, dpi=300):
    try:
        shutil.rmtree(output_folder)
    except:
        pass  

    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))  
        image_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
        pix.save(image_path)

def get_base64_images(directory):
    base64_images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            with open(os.path.join(directory, filename), 'rb') as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                base64_images.append(base64_image)
    return base64_images

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")  
    return text

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


def analyzing_document_with_prompt(system_prompt, user_prompt, message):
    llm = AzureOpenAI(
        api_version=AZURE_DEPLOYMENT_ID,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY
    )

    response = ""
    thread_history = []
    while True:
        current_response = llm.chat.completions.create(
            messages=thread_history + message,
            model="gpt-4o",
            temperature=0,
            # max_tokens=4096  
        )
        thread_history.append({"role": "assistant", "content": current_response.choices[0].message.content})
        response += current_response.choices[0].message.content
        
        if current_response.choices[0].finish_reason == "stop":
            break
        elif current_response.choices[0].finish_reason == "length":
            message = [{"role": "user", "content": "Continue from where you left off."}]
    
    return response

# Streamlit UI
st.title("Extraction of MPC file(s)")

uploaded_file = st.file_uploader("Choose a PDF...", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Processing file: {uploaded_file.name}")
    with open(f"temp_{uploaded_file.name}", "wb") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
         
    pdf_to_images(f"temp_{uploaded_file.name}", "output_images", dpi=300)
    
    pdf_text = extract_text_from_pdf(f"temp_{uploaded_file.name}")
    st.write("Extracted Text from PDF:")
    st.write(pdf_text)
    
    if st.button("Analyze"):

        # Extract base64 images
        directory_path = './output_images/'
        base64_images = get_base64_images(directory_path)

        system_prompt = (
        "You are a data extraction specialist skilled at extracting and organizing information into structured tables. "
        "Your task is to carefully analyze and extract specified available fields from the provided images, no matter the format or structure."
        )

        user_prompt = """
        Please analyze the provided images and extract these fields: 
            
        1. Section A: Medical Bill/Receipt/Invoice
            Extract all data, sections and pages that contains charges, billing, receipts, or invoices, and present the output in a detailed table format and strictly do not truncate the output.
            - Use contextual understanding to infer which 'Description' in the extracted medical bill data most closely matches the 'Receipt Item Code' (RIC) from the provided JSON.
            - The model should consider semantic similarity, synonyms, or relevant keywords between the 'Description' and the 'Receipt Item Code.'
            For each match:
            - Retrieve the corresponding 'Grouping' and 'Covered By' values from the JSON and include them in the output table.
            - If no reasonable match can be inferred, set the 'Receipt Item Code,' 'Grouping,' and 'Covered By' columns to 'N/A.'            
            - Ensure the output table is complete, accurate, and contains no omissions or truncations.
            - Create another table (Overview Bill) to group by the "Covered By" from Detailed Bill table and sum all categories that falls under their respective "Covered By". 
            - Column for Output tables:
            Table 1: Overview Bill columns: Covered By, Total Amount
            Table 2: Detailed Bill columns: Item, Receipt Item Code, Amount, Grouping, Covered By
            
            The reference data for "Grouping" and "Covered By" values is provided in the JSON format below:
                
                {"Grouping":"Room and Board","Receipt Item Code":"Room and Board","Covered By":"Room and Board Benefit"}
                {"Grouping":"HDU","Receipt Item Code":"HDU","Covered By":"ICU Benefit"}
                {"Grouping":"ICU","Receipt Item Code":"ICU","Covered By":"ICU Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Ambulance","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Dedicated Medical Attendant","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Equipment Chargers","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Food & Beverages","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Lab / Pathology Services","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Lodger","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Medical Report Fee","Covered By":"N/A"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Medication / Pharmacy","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Take Home Drug","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Miscellaneous charges","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Nursing Fee / Procedure","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Physiotherapy / Rehabilitation","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Radiology / Diagnostic Imaging","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Ward Medical Supplies","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Maternity Complication","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Organ Transplant","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Hospital Charges","Receipt Item Code":"Intraocular lens","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Operation Theatre","Receipt Item Code":"OT Fee","Covered By":"Surgical Benefit"}
                {"Grouping":"Operation Theatre","Receipt Item Code":"Surgical Medical Supplies / consumable","Covered By":"Surgical Benefit"}
                {"Grouping":"Operation Theatre","Receipt Item Code":"OT Equipments","Covered By":"Surgical Benefit"}
                {"Grouping":"Operation Theatre","Receipt Item Code":"Medical implant / consignment item","Covered By":"Surgical Benefit"}
                {"Grouping":"Severity Level","Receipt Item Code":"Surgery Severity Level","Covered By":"Surgical Procedure Benefit"}
                {"Grouping":"Non payable items","Receipt Item Code":"Administrative / Registration / Admission / Service / Hospital Ancillary Fees","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"ID band / bracelet","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Medical report","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Take home external appliances","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Supplement / Vitamin","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Miscellaneous charges","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Health Screening / Medical Routine Check Up charges","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Photocopy / Certified True Copy / Duplicate Official Receipt","Covered By":"N/A"}
                {"Grouping":"Doctor Charges","Receipt Item Code":"Doctor Consultation","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Doctor Charges","Receipt Item Code":"Ward Review","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Doctor Charges","Receipt Item Code":"Surgical fee","Covered By":"Surgical Benefit"}
                {"Grouping":"Doctor Charges","Receipt Item Code":"Anaesthetist","Covered By":"Surgical Benefit"}
                {"Grouping":"Incubation","Receipt Item Code":"Incubation","Covered By":"Hospital Charges Benefit"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Consultation Fee","Covered By":"Pre-hosp and Post-hosp"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Second medical opinion (Malaysia)","Covered By":"Pre-hosp and Post-hosp"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Diagnostic Test","Covered By":"Pre-hosp and Post-hosp"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Medication","Covered By":"Pre-hosp and Post-hosp"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Physiotherapy / Rehabilitation","Covered By":"Pre-hosp"}
                {"Grouping":"Pre-Post Hosp","Receipt Item Code":"Other Charges","Covered By":"Pre-hosp and Post-hosp"}
                {"Grouping":"Cancer Treatment","Receipt Item Code":"Cancer Treatment_Medication","Covered By":"Cancer Treatment"}
                {"Grouping":"Cancer Treatment","Receipt Item Code":"Cancer Treatment_Diagnostic Test","Covered By":"Cancer Treatment"}
                {"Grouping":"Cancer Treatment","Receipt Item Code":"Cancer Treatment_Consultation","Covered By":"Cancer Treatment"}
                {"Grouping":"Cancer Treatment","Receipt Item Code":"Cancer Treatment_Treatment / Procedure","Covered By":"Cancer Treatment"}
                {"Grouping":"Kidney Dialysis","Receipt Item Code":"Kidney Dialysis_Medication","Covered By":"Kidney Dialysis Benefit"}
                {"Grouping":"Kidney Dialysis","Receipt Item Code":"Kidney Dialysis_Diagnostic Test","Covered By":"Kidney Dialysis Benefit"}
                {"Grouping":"Kidney Dialysis","Receipt Item Code":"Kidney Dialysis_Consultation","Covered By":"Kidney Dialysis Benefit"}
                {"Grouping":"Kidney Dialysis","Receipt Item Code":"Kidney Dialysis_Treatment / Procedure","Covered By":"Kidney Dialysis Benefit"}
                {"Grouping":"EAT","Receipt Item Code":"EAT_Medication","Covered By":"Emergency Treatment for Accidental Injury Benefit"}
                {"Grouping":"EAT","Receipt Item Code":"EAT_Diagnostic Test","Covered By":"Emergency Treatment for Accidental Injury Benefit"}
                {"Grouping":"EAT","Receipt Item Code":"EAT_Consultation","Covered By":"Emergency Treatment for Accidental Injury Benefit"}
                {"Grouping":"EAT","Receipt Item Code":"EAT_Treatment / Procedure","Covered By":"Emergency Treatment for Accidental Injury Benefit"}
                {"Grouping":"EAT","Receipt Item Code":"EAT_Dental","Covered By":"Emergency Treatment for Accidental Injury Benefit"}
                {"Grouping":"AMR","Receipt Item Code":"Traditional Treatment","Covered By":"Traditional Treatment"}
                {"Grouping":"AMR","Receipt Item Code":"Overseas Companion Allowance","Covered By":"Overseas Companion Allowance"}
                {"Grouping":"AMR","Receipt Item Code":"Prosthesis Allowance (per limb)","Covered By":"Prosthesis Allowance (per limb)"}
                {"Grouping":"AMR","Receipt Item Code":"Wheelchair Allowance","Covered By":"Wheelchair Allowance"}
                {"Grouping":"AMR","Receipt Item Code":"Medical Expenses","Covered By":"AMR"}
                {"Grouping":"Nursing Care","Receipt Item Code":"Home Nursing Care","Covered By":"Home Nursing Benefit"}
                {"Grouping":"Non payable items","Receipt Item Code":"Medical report","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Administrative / Registration / Admission / Service / Hospital Ancillary Fees","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Take home external appliances","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Supplement / Vitamin","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Miscellaneous charges","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Health Screening / Medical Routine Check Up charges","Covered By":"N/A"}
                {"Grouping":"Non payable items","Receipt Item Code":"Equipment Chargers","Covered By":"N/A"}
            """


        message = create_message(system_prompt, user_prompt, base64_images)
        analysis = analyzing_document_with_prompt(system_prompt, user_prompt, message)
        st.write("Analysis Result:")
        st.write(analysis)
        

