import pandas as pd
import os


def load_data(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'raw', filename)

    data=pd.read_csv(file_path)

    print("Shape:",data.shape)
    print("Columns:",data.columns.tolist())
    return data

##Function to output a string type description of each row
def row_to_text(row):
    # Using an f-string to build a natural-sounding sentence. 
    sentence = (
        f"Patient {row['Name']} is a {row['Age']}-year-old {row['Gender']} "
        f"(who was admitted to {row['Hospital']} "
        f"on {row['Date of Admission']} as an '{row['Admission Type']}' case for "
        f"{row['Medical Condition']}. Under the care of Dr. {row['Doctor']}, "
        f"their test results were marked as '{row['Test Results']}', and they "
        f"were prescribed {row['Medication']}. The total billing amount for "
        f"this visit was ${row['Billing Amount']}."
    )
    
    return sentence

##Function built to apply the function above to whole dataset
def build_snippets(df):
    df['Patient_ID'] = df.index
    
    # Apply the row_to_text function across the rows (axis=1)
    df['Snippet'] = df.apply(row_to_text, axis=1)

    ##Will return id with description column only
    return df[['Patient_ID', 'Snippet']]

def save_snippets(df,output_path):
    df.to_csv(output_path, index=False)
    print(f"\nSuccess: Processed snippets saved to {output_path}")






if __name__ == "__main__":
    df = load_data('healthcare_dataset.csv')
    first_row_text = row_to_text(df.iloc[0])

    ##Generate description of each column
    descriptions=build_snippets(df)
    ##Printing the first 5 rows to test
    print(descriptions.head(5))

    ##Printing to check that whther any row doesnt have NaN value left
    nan_count = descriptions['Snippet'].isna().sum()
    print(f"Total NaN snippets found: {nan_count}")


    ##Safe the snippet to the a separate file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, '..', 'data', 'processed', 'snippets.csv')
    
    # Execute the save function
    save_snippets(descriptions, output_file)