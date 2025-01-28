import os
import re
import json
from transformers import pipeline

def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

    # Split document into paragraphs
    paragraphs = document.split('\n\n')

    # Removing empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip() != '']

    return paragraphs

def risk_detection(chunks):
    model_name = "google/flan-t5-base"
    nlp = pipeline("text2text-generation", model=model_name)

    results = []
    for chunk in chunks:
        # Generate analysis for risks
        prompt_risks = (
            "Review the following legal document text for potential risks, such as compliance issues, liabilities, "
            "or ambiguities. Provide a detailed explanation of the identified risks:\n\n"
            + chunk
        )
        risks_result = nlp(prompt_risks, max_length=512, do_sample=False)

        # Generate analysis for hidden obligations
        prompt_obligations = (
            "Examine the following legal document text for hidden obligations or dependencies, "
            "such as unmentioned responsibilities, implied terms, or contingent liabilities. "
            "List and explain these obligations or dependencies in detail:\n\n"
            + chunk
        )
        obligations_result = nlp(prompt_obligations, max_length=512, do_sample=False)

        # Generate recommendations
        prompt_recommendations = (
            "Based on the potential risks and hidden obligations identified in the legal document text, "
            "provide specific and actionable recommendations to address or mitigate them. Include legal "
            "strategies or best practices where applicable:\n\n"
            + chunk
        )
        recommendations_result = nlp(prompt_recommendations, max_length=512, do_sample=False)

        results.append({
            "context": chunk,
            "risks_analysis": risks_result[0]['generated_text'],
            "obligations_analysis": obligations_result[0]['generated_text'],
            "recommendations": recommendations_result[0]['generated_text']
        })

    return results

def main(file_path):
  print("Loading and preprocessing document...")
  chunks = load_and_preprocess(file_path)

  print("Detecting risks and generating recommendations...")

  analysis = risk_detection(chunks)

#Save results to a JSON file

  output_path = "risk_analysis.json"

  with open(output_path, "w", encoding="utf-8") as f:

    json.dump(analysis, f, ensure_ascii=False, indent=4)

  print(f"Analysis complete. Results saved to {output_path}")

# Execute the pipeline

if __name__ == "__main__":
  file_path = "law.txt"# Replace with your file path
  main(file_path)

   # Load the analysis results and print them
with open('risk_analysis.json', 'r') as f:
    data = json.load(f)
for entry in data:
    print("Context:\n", entry["context"])
    print("Risks Analysis:\n", entry["risks_analysis"])
    print("Obligations Analysis:\n", entry["obligations_analysis"])
    print("Recommendations:\n", entry["recommendations"])
    print("-" * 80)


import pandas as pd
import json
import openpyxl
from openpyxl.styles import Alignment

# Load data from JSON file
with open("risk_analysis.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Print the analysis details with each column below one another
for entry in data:
    print("Context:\n", entry["context"])
    print("Risks Analysis:\n", entry["risks_analysis"])
    print("Obligations Analysis:\n", entry["obligations_analysis"])
    print("Recommendations:\n", entry["recommendations"])
    print("-" * 80)

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel('risk_analysis_formatted.xlsx', index=False)

# Open the Excel file and adjust cell formatting for text wrapping and auto column width
wb = openpyxl.load_workbook('risk_analysis_formatted.xlsx')
ws = wb.active

# Adjust column width and enable text wrapping
for column in ws.columns:
    max_length = 0
    column_letter = column[0].column_letter  # Get the column name
    for cell in column:
        if cell.value is not None:
            cell.alignment = Alignment(wrap_text=True)  # Enable text wrapping
            max_length = max(max_length, len(str(cell.value)))
    adjusted_width = (max_length + 2)
    ws.column_dimensions[column_letter].width = adjusted_width

wb.save('risk_analysis_formatted.xlsx')

# Display the formatted DataFrame
print(df)
