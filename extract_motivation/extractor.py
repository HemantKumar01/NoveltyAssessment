import fitz  
import json
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# Initialize the Gemini client (it automatically looks for GEMINI_API_KEY in your .env file)
client = genai.Client()

def extract_text_from_pdf(pdf_path):
    """Reads the PDF file and extracts raw text."""
    print(f"Reading PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_arise_features(paper_text):
    """Sends the text to Gemini using the ARISE Appendix H prompt."""
    print("Sending text to Gemini for extraction. This may take a minute...")
    
    # The exact prompt from the ARISE paper Appendix H
    system_prompt = """
    You are an AI assistant tasked with summarizing research papers in a structured and clear manner.
    User Instructions: Please summarize the following research paper by providing the following details in JSON format:

    • Topic: Identify the key concepts, research questions, or objectives discussed in the paper. Summarize the main topic in one or two sentences, ensuring it captures the essence of the paper. Avoid including unnecessary details or examples.
    • Motivation: Provide an explanation of the current state of the field. What are the key achievements in this area of research, and what are the limitations or open challenges that this paper addresses? Describe why this research is important and how it aims to contribute to advancing the field.
    • Method: Explain the methodology used in the paper. In particular, describe the specific designs or approaches the authors implemented to address the limitations or gaps identified in the motivation. Provide a summary of the targeted designs, followed by a detailed explanation of each design individually.
        - Summary: Give an overview of the key innovations or design choices made to overcome existing limitations in the field.
        - Detailed designs: For each targeted design, provide a thorough explanation. Discuss innovations in model architecture, algorithms, data processing techniques, or training strategies. Please anonymize the names of the methods in the descriptions.
        - Problems solved: List the specific problems that each design addresses, separately.
        - Datasets: Mention the datasets used for training and testing the model, including any unique characteristics or challenges they present.
        - Metrics: Specify the evaluation metrics used to assess the model's performance, such as accuracy, precision, recall, F1 score, etc.

    Ensure that the output adheres to the following requirements:
    • Provide clear and concise explanations.
    • Summarize the content in a structured, easy-to-read format.
    • For the "method" section, ensure that:
        - Targeted designs: The summary should provide an overview of the key innovations or strategies.
        - Individual designs: Determine what small designs make up the overall framework, break down the whole framework into individual small design and use the orignal sentece from the paper to explain them in detail separately, such as the detail architectural changes, novel algorithms, or new techniques. Anonymize the names of methods and techniques by describing them in a general sense, avoiding any specific names.
        - Problems solved: List the specific problems each design addresses separately
        - Datasets: Mention the datasets used for training and testing the model, including any unique characteristics or challenges they present.
        - Metrics: Specify the evaluation metrics used to assess the model's performance, such as accuracy, precision, recall, F1 score, etc.

    Use the following JSON structure for the output:
    {
      "topic": "The main research object and scope of the study",
      "motivation": "Current state of the field, achievements, and limitations addressed by this study",
      "method": {
        "targeted_designs_summary": "A high-level summary of the designs or innovations made to address limitations",
        "targeted_designs_details": [
          {
            "design_name": "Name of the design",
            "description": "Detailed explanation of this design, including its purpose, how it addresses limitations, and any novel aspects (anonymized)",
            "problems_solved": "Problem that this design solve"
          }
        ],
        "datasets": "Datasets used in the experiments",
        "metrics": "Evaluation metrics used to assess the effectiveness of the approach"
      }
    }
    """

    user_prompt = f"Here is the provided paper:\n\n{paper_text}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json", 
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        return None

if __name__ == "__main__":
    target_pdf_filename = "target_paper.pdf" 
    output_json_filename = "extracted_data.json"

    if not os.path.exists(target_pdf_filename):
        print(f"Error: Could not find a file named '{target_pdf_filename}' in this folder.")
    else:
        paper_content = extract_text_from_pdf(target_pdf_filename)
        
        if paper_content:
            extracted_data = extract_arise_features(paper_content)
            
            if extracted_data:
                with open(output_json_filename, "w", encoding="utf-8") as f:
                    json.dump(extracted_data, f, indent=4)
                print(f"Success! Open '{output_json_filename}' in your VS Code sidebar to see the results.")