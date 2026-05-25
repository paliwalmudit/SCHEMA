import os
from docx import Document
import json
from docx2pdf import convert

def save_project_files(pipeline_data, name):
    """
    Creates a new folder and generates two files:
    1. An HTML file with visualization content
    2. A Word file with research pipeline content
    """
    folder_name = os.path.join("results", "Research_Meeting_Output")
    json_folder = "Json_Results"

    #loading the pipeline data in json format
    pipeline_data=json.loads(pipeline_data)

    # Create folder (if not exists)
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # ---  Save JSON File ---
    name = name.replace("_Principal_Investigator_Agent","")
    json_path = os.path.join(json_folder, name + ".json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(pipeline_data, json_file, indent=4, ensure_ascii=False)

    # ---  Save Word File ---
    doc_path = os.path.join(folder_name, name +"_research_pipeline.docx")
    doc = Document()


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

    print("\n \n --- Files Created Successfully --- \n")
    print(f"Files created successfully in folders: {folder_name} and {json_folder}")
    print(f"Word file: {doc_path}")
    print(f"JSON file: {json_path}")
