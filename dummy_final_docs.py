import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
import os
import csv
from logging_setup import logger

final_documents = None
def final_docs(input_path):
    global final_documents
    final_documents = []

    def load_files(input_path_or_list):
        all_documents = []
        file_count = 0  # Initialize the file counter

        def process_pdf(file_path):
            nonlocal file_count
            try:
                with fitz.open(file_path) as pdf:
                    logger.info(f"Processing PDF file: {file_path}")  # Use logger instead of print
                    successful_pages = 0
                    for page_number in range(len(pdf)):
                        page = pdf.load_page(page_number)
                        text = page.get_text()
                        if text.strip():
                            all_documents.append(Document(page_content=text, metadata={"source": file_path, "page_number": page_number}))
                            successful_pages += 1
                    if successful_pages > 0:
                        file_count += 1  # Increment if any pages had content
            except Exception as e:
                logger.error(f"An error occurred while processing {file_path}: {e}")

        def process_csv(file_path):
            nonlocal file_count
            try:
                with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    content = ""
                    for row in reader:
                        content += " ".join(row) + "\n"
                    if content.strip():
                        all_documents.append(Document(page_content=content, metadata={"source": file_path}))
                        file_count += 1  # Increment for each successful CSV file
                    logger.info(f"Processing CSV file: {file_path}")  # Use logger instead of print
            except Exception as e:
                logger.error(f"An error occurred while processing {file_path}: {e}")

        # Determine if the input is a string (path) or list of paths
        if isinstance(input_path_or_list, str):
            # Check if the string is a file path or folder path
            if os.path.isfile(input_path_or_list):
                # Single file path
                if input_path_or_list.endswith(".pdf"):
                    process_pdf(input_path_or_list)
                elif input_path_or_list.endswith(".csv"):
                    process_csv(input_path_or_list)
                else:
                    logger.warning(f"The provided file is neither a PDF nor a CSV: {input_path_or_list}")
        
            elif os.path.isdir(input_path_or_list):
                # Folder path
                try:
                    files = [os.path.join(input_path_or_list, f) for f in os.listdir(input_path_or_list) if f.endswith((".pdf", ".csv"))]
                    for file_path in files:
                        if file_path.endswith(".pdf"):
                            process_pdf(file_path)
                        elif file_path.endswith(".csv"):
                            process_csv(file_path)
                except Exception as e:
                    logger.error(f"An error occurred while processing files in {input_path_or_list}: {e}")

            else:
                logger.warning(f"The provided path is neither a valid file nor a folder: {input_path_or_list}")

        elif isinstance(input_path_or_list, list):
            # List of file paths
            for file_path in input_path_or_list:
                if os.path.isfile(file_path):
                    if file_path.endswith(".pdf"):
                        process_pdf(file_path)
                    elif file_path.endswith(".csv"):
                        process_csv(file_path)
                    else:
                        logger.warning(f"Invalid file type in list: {file_path}")
                else:
                    logger.warning(f"Invalid file path in list: {file_path}")

        else:
            logger.warning("Invalid input: Provide either a file path, folder path, or a list of PDF/CSV file paths.")

        logger.info(f"{file_count} file(s) under process.")
        return all_documents

    documents = load_files(input_path)
    if documents  != None:
        logger.info("File uploaded successfully!")
    else:
        logger.error("Error processing file. Please check input path and try again. Thank you!")

    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    final_documents = text_splitter.split_documents(documents)

    return final_documents
