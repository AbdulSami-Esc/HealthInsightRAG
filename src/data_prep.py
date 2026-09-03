import os
from pathlib import Path

import pandas as pd
##The puirpose of creating this whole proghram is to ensure structural transformation of tabular patient records
##into semantic natural language to make it easy for model to understand

def load_data(filename:str)->pd.DataFrame:
    file_path = Path(__file__).parent.parent / 'data' / 'raw' / filename
    data=pd.read_csv(file_path)

    print("Shape:",data.shape)
    print("Columns:",data.columns.tolist())
    return data

##Ddi this so that all NaN values in any column of the patienmr rows can be handl;ed properly
def _safe_value(value):
    if pd.isna(value):
        return "|Nan"
    return str(value)


##Function to output a string type description of each row
def row_to_text(row:pd.Series)->str:
    name = _safe_value(row.get('Name', '|Nan'))
    age = _safe_value(row.get('Age', '|Nan'))
    gender = _safe_value(row.get('Gender', '|Nan'))
    hospital = _safe_value(row.get('Hospital', '|Nan'))
    admission_date = _safe_value(row.get('Date of Admission', '|Nan'))
    admission_type = _safe_value(row.get('Admission Type', '|Nan'))
    medical_condition = _safe_value(row.get('Medical Condition', '|Nan'))
    doctor = _safe_value(row.get('Doctor', '|Nan'))
    test_results = _safe_value(row.get('Test Results', '|Nan'))
    medication = _safe_value(row.get('Medication', '|Nan'))
    billing_amount = _safe_value(row.get('Billing Amount', '|Nan'))

    sentence = (
        f"Patient {name} is a {age}-year-old {gender} "
        f"(who was admitted to {hospital} on {admission_date} as an '{admission_type}' case for "
        f"{medical_condition}. Under the care of Dr. {doctor}, their test results were marked as "
        f"'{test_results}', and they were prescribed {medication}. The total billing amount for "
        f"this visit was ${billing_amount}."
    )

    return sentence

def build_snippets(df):
    ##Function built to apply the function above to whole dataset

    snippets_df = pd.DataFrame()
    snippets_df['Patient_ID'] = df.index
    snippets_df['Snippet'] = df.apply(row_to_text, axis=1)
    return snippets_df


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