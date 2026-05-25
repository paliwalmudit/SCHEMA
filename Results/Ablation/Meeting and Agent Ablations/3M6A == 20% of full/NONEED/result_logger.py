import os
from docx import Document
import json
from docx2pdf import convert

def save_project_files(html_content, pipeline_data, name):
    """
    Creates a new folder and generates two files:
    1. An HTML file with visualization content
    2. A Word file with research pipeline content
    """
    folder_name = os.path.join("results", "Research_Meeting_Output")

    #loading the pipeline data in json format
    pipeline_data=json.loads(pipeline_data)

    # Create folder (if not exists)
    os.makedirs(folder_name, exist_ok=True)

    # --- 1. Save HTML File ---
    html_path = os.path.join(folder_name, name +"_graph_visualization.html")
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    # --- 2. Save Word File ---
    doc_path = os.path.join(folder_name, name +"_research_pipeline.docx")
    doc = Document()

    # --- 3. Save JSON File ---
    json_path = os.path.join(folder_name, name + "_research_pipeline.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(pipeline_data, json_file, indent=4, ensure_ascii=False)


    # Title
    doc.add_heading("Research Pipeline", level=1)
    doc.add_paragraph(" ")

    # Iterate and write pipeline data
    for section, text in pipeline_data.items():
        doc.add_heading(section.replace("_", " ").title(), level=2)
        doc.add_paragraph(str(text))
        doc.add_paragraph(" ")

    try:
         doc.save(doc_path)
    except Exception as e:
        print(f"Error saving Word file: {e}")
        print("saving in current directory with '-new' suffix in the name.")
        doc_path = os.path.join(folder_name, name +"_research_pipeline-new.docx")
        doc.save(doc_path)
    # --- 3. Convert DOCX → PDF (Windows) ---
    pdf_path = doc_path.replace(".docx", ".pdf")

    try:
        convert(doc_path, pdf_path)
    except Exception as e:
        print(f"PDF conversion failed: {e}")
        print("Ensure Microsoft Word is installed and not running in a blocked state.")

    print("\n \n --- Files Created Successfully --- \n")
    print(f"Files created successfully in folder: {folder_name}")
    print(f"HTML file: {html_path}")
    print(f"Word file: {doc_path}")
    print(f"PDF file: {pdf_path}")
    print(f"JSON file: {json_path}")
