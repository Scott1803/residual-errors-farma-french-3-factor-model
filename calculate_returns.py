import csv
import os


def calculate_returns():
    """
    Calculate daily returns from share price data.
    
    Reads share prices from input/share-prices.tsv, calculates daily returns
    using the formula: (price_t1 - price_t0) / price_t1
    and writes the results to output/daily-returns.tsv
    """
    # Define input and output paths
    input_path = os.path.join('input', 'share-prices.tsv')
    output_dir = 'output'
    output_path = os.path.join(output_dir, 'daily-returns.tsv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the share price data
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        # Read header (Date + company ISINs)
        header = next(reader)
        
        # Read all data rows
        data_rows = list(reader)
    
    # Prepare output data
    output_rows = []
    
    # Process each row starting from the second one (index 1)
    # First row has no previous price, so we can't calculate returns
    for i in range(len(data_rows)):
        if i == 0:
            # For the first data row, we cannot calculate returns (no previous day)
            # We'll skip it or write empty values
            continue
        
        current_row = data_rows[i]
        previous_row = data_rows[i - 1]
        
        # Start with the date
        returns_row = [current_row[0]]
        
        # Calculate returns for each company (columns 1 onwards)
        for col_idx in range(1, len(header)):
            price_t0_str = previous_row[col_idx] if col_idx < len(previous_row) else ''
            price_t1_str = current_row[col_idx] if col_idx < len(current_row) else ''
            
            # Check if both prices are available (not empty)
            if price_t0_str.strip() and price_t1_str.strip():
                try:
                    # Replace comma with dot for German number format
                    price_t0 = float(price_t0_str.replace(',', '.'))
                    price_t1 = float(price_t1_str.replace(',', '.'))
                    
                    # Calculate return: (price_t1 - price_t0) / price_t0
                    if price_t1 != 0:
                        daily_return = (price_t1 - price_t0) / price_t0
                        # Format with comma as decimal separator to match input format
                        returns_row.append(str(daily_return).replace('.', ','))
                    else:
                        returns_row.append('')
                except ValueError:
                    # If conversion fails, write empty value
                    returns_row.append('')
            else:
                # If either price is missing, write empty value
                returns_row.append('')
        
        output_rows.append(returns_row)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(header)
        
        # Write all return rows
        writer.writerows(output_rows)
    
    print(f"Daily returns calculated and saved to {output_path}")
    print(f"Processed {len(output_rows)} trading days")


if __name__ == "__main__":
    calculate_returns()
