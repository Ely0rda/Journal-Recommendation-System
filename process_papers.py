from google.colab import drive
import json
import pandas as pd
import os

def debug_data_structure(data, max_entries=2):
    """Debug helper to print data structure"""
    print("\nData Structure Analysis:")
    print(f"Type: {type(data)}")
    print(f"Length: {len(data)}")
    print("\nSample Entry:")
    for i, entry in enumerate(data[:max_entries]):
        print(f"\nEntry {i+1}:")
        for key in entry:
            print(f"- {key}: {type(entry[key])}")
            if key == "Publications":
                print(f"  Number of publications: {len(entry[key])}")
                if entry[key]:
                    print("  First publication keys:", list(entry[key][0].keys()))

def safe_float_convert(value, default=0.0):
    """Safely convert value to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return default
    return default

def calculate_journal_score(journal_info):
    """Calculate weighted journal score with validation"""
    if not journal_info or not isinstance(journal_info, dict):
        return 0.0
        
    # Get h-index
    h_index = safe_float_convert(journal_info.get('H-index'))
    
    # Get SJR
    sjr = journal_info.get('SJR', {})
    sjr_value = 0.0
    if isinstance(sjr, dict):
        sjr_value = safe_float_convert(sjr.get('sjr'))
    elif isinstance(sjr, (int, float, str)):
        sjr_value = safe_float_convert(sjr)
    
    # Get impact factor
    impact_factor = safe_float_convert(journal_info.get('Impact factor'))
    
    return 0.33 * h_index + 0.33 * sjr_value + 0.33 * impact_factor

def process_papers(input_file, output_file):
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    error_count = {
        'missing_fields': 0,
        'invalid_journal': 0,
        'conversion_error': 0
    }
    
    for author in data:
        documents = author.get('documents', [])
        if not documents:
            error_count['missing_fields'] += 1
            continue
        
        for doc in documents:
            try:
                if not isinstance(doc, dict):
                    error_count['invalid_journal'] += 1
                    continue
                    
                title = doc.get('title', '').strip()
                abstract = doc.get('abstract', '').strip()
                journal = doc.get('journal', {})
                
                # Update journal name extraction
                journal_name = journal.get('Name', '')  # Changed from 'name' to 'Name'
                if not journal_name:
                    error_count['missing_fields'] += 1
                    continue
                
                if not all([title, abstract, journal]):
                    error_count['missing_fields'] += 1
                    continue
                
                journal_score = calculate_journal_score(journal)
                
                processed_data.append({
                    'Title': title,
                    'Abstract': abstract,
                    'Keywords': doc.get('keywords', []),
                    'Journal': {
                        'journal_name': journal_name.strip(),  # Use extracted journal name
                        'Journal_Score': journal_score
                    }
                })
                
                if len(processed_data) % 100 == 0:
                    print(f"Processed {len(processed_data)} papers...")
                    
            except Exception as e:
                error_count['conversion_error'] += 1
                print(f"Error processing document: {str(e)}")
    
    print("\nError Summary:")
    for error_type, count in error_count.items():
        print(f"{error_type}: {count}")
    
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {len(processed_data)}")
    print(f"Total skipped: {sum(error_count.values())}")
    
    return processed_data

if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)
    
    input_file = '/content/drive/MyDrive/integrated_authors.json'
    output_file = '/content/drive/MyDrive/processed_data.json'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
    else:
        processed_data = process_papers(input_file, output_file)