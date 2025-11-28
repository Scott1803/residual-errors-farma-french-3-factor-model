import csv
import os
from datetime import datetime


def clean_eu_market_returns():
    """
    Clean EU market returns data from input/eu-market-returns.tsv
    
    - Parses dates from "DD-Mon-YYYY" format to "DD.MM.YY" format
    - Converts percentage strings (e.g., "+0,37%", "-1,70%") to float values
    - Saves cleaned data to output/eu-market-data-cleaned.tsv
    """
    # Define input and output paths
    input_path = os.path.join('input', 'eu-market-returns.tsv')
    output_dir = 'output'
    output_path = os.path.join(output_dir, 'eu-market-data-cleaned.tsv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the EU market returns data
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        cleaned_rows = []
        
        for row in reader:
            # Parse the date from "21-Nov-2025" to "21.11.25"
            date_str = row['exchange-date']
            date_obj = datetime.strptime(date_str, '%d-%b-%Y')
            formatted_date = date_obj.strftime('%d.%m.%y')
            
            # Parse the change value from "+0,37%" or "-1,70%" to float
            change_str = row['change']
            # Remove "+" and "%" symbols
            change_str = change_str.replace('+', '').replace('%', '').strip()
            # Replace comma with dot for decimal separator
            change_str = change_str.replace(',', '.')
            # Convert to float and divide by 100 to get decimal form
            change_value = float(change_str) / 100
            # Format back with comma as decimal separator
            change_formatted = str(change_value).replace('.', ',')
            
            cleaned_rows.append({
                'exchange-date': formatted_date,
                'change': change_formatted
            })
    
    # Write cleaned data to output file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['exchange-date', 'change'], delimiter='\t')
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"EU market returns cleaned and saved to {output_path}")
    print(f"Processed {len(cleaned_rows)} rows")


if __name__ == "__main__":
    clean_eu_market_returns()
