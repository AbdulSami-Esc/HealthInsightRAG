import sys
from pathlib import Path
import time
import pandas as pd

# Setup paths of main.py
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from main import route_query as router_func

# 40 Test Queries: 10 per Lane
TEST_CASES = [
    # Lane 1: GENERAL (Medical definitions, pathology, standard pharmacology)
    ("What are the early warning signs of acute appendicitis?", "GENERAL"),
    ("Can you define atrial fibrillation and its common risk factors?", "GENERAL"),
    ("What is the mechanism of action of metformin in type 2 diabetes?", "GENERAL"),
    ("What lifestyle changes are recommended for chronic kidney disease stage 2?", "GENERAL"),
    ("Differentiate between viral and bacterial pneumonia symptoms.", "GENERAL"),
    ("What are the common side effects of ACE inhibitors?", "GENERAL"),
    ("How does atherosclerosis lead to myocardial infarction?", "GENERAL"),
    ("What is the typical incubation period for influenza A?", "GENERAL"),
    ("Explain the difference between osteoarthritis and rheumatoid arthritis.", "GENERAL"),
    ("What are the clinical diagnostic criteria for hypertension?", "GENERAL"),

    # Lane 2: SINGLE_PATIENT (Contains specific record ID)
    ("What medications was patient P0001 prescribed upon discharge?", "SINGLE_PATIENT"),
    ("Summarize the operative report for record P0015.", "SINGLE_PATIENT"),
    ("What were the findings of the chest CT scan for P0042?", "SINGLE_PATIENT"),
    ("Can you review the physical examination notes for patient P0104?", "SINGLE_PATIENT"),
    ("Did patient P0023 have any recorded drug allergies?", "SINGLE_PATIENT"),
    ("What was the initial diagnosis recorded for patient P0088 in the emergency note?", "SINGLE_PATIENT"),
    ("Retrieve the post-operative recovery plan for patient P0201.", "SINGLE_PATIENT"),
    ("What was the blood pressure reading recorded during admission for P0005?", "SINGLE_PATIENT"),
    ("Show me the family medical history mentioned in P0310.", "SINGLE_PATIENT"),
    ("What dosage of lisinopril was given to patient P0012?", "SINGLE_PATIENT"),

    # Lane 3: COHORT (Thematic search across multiple records without specific ID)
    ("Find all patients who presented with severe shortness of breath or dyspnea.", "COHORT"),
    ("Identify patients who underwent a laparoscopic cholecystectomy.", "COHORT"),
    ("Which records show a diagnosis of congestive heart failure with reduced ejection fraction?", "COHORT"),
    ("List all patients with documented history of chronic obstructive pulmonary disease (COPD).", "COHORT"),
    ("Find records of patients admitted with diabetic ketoacidosis.", "COHORT"),
    ("Search for any patients that developed post-operative wound infections.", "COHORT"),
    ("Which patients had abnormal electrocardiogram (EKG) findings indicating ischemia?", "COHORT"),
    ("Find transcription records discussing patients treated for deep vein thrombosis.", "COHORT"),
    ("List cases where patients were treated for acute ischemic stroke with tPA.", "COHORT"),
    ("Identify patients with stage 3 or 4 chronic kidney disease mentioned in consultations.", "COHORT"),

    # Lane 4: ANALYTICAL (Aggregations, counts, percentages, metadata queries)
    ("How many total clinical chunks exist in the dataset, and what is the breakdown by section?", "ANALYTICAL"),
    ("What is the total count of distinct patients in the metadata?", "ANALYTICAL"),
    ("Which medical specialty has the highest number of transcription records?", "ANALYTICAL"),
    ("What is the average number of chunks per patient record?", "ANALYTICAL"),
    ("Count how many records belong to the Cardiology specialty versus Neurology.", "ANALYTICAL"),
    ("What are the top 5 most frequent medical specialties in the chunk metadata?", "ANALYTICAL"),
    ("How many chunks are categorized under the description bucket?", "ANALYTICAL"),
    ("What percentage of the dataset corresponds to surgical operation notes?", "ANALYTICAL"),
    ("Give me the total row count of the chunk_metadata CSV file.", "ANALYTICAL"),
    ("How many records have missing or null values in the sample_name column?", "ANALYTICAL"),
]

##Only isolting and running the route query function so that router is isolated and less token cost
def run_router_eval():
    results = []
    correct_count = 0
    total = len(TEST_CASES)

    print("\n" + "=" * 95)
    print(f"{'#':<3} | {'Query (Truncated)':<42} | {'Expected':<15} | {'Outcome':<15} | {'Status'}")
    print("=" * 95)

    for idx, (query, expected) in enumerate(TEST_CASES, start=1):
        try:
            # Call your router function
            raw_res = router_func(query)

            # Extract the 'intent' key returned by your JSON router
            if isinstance(raw_res, dict):
                outcome = raw_res.get("intent", "MISSING_INTENT_KEY")
            else:
                outcome = str(raw_res).strip()

            outcome = str(outcome).upper()

        except Exception as e:
            outcome = f"ERROR: {str(e)[:15]}"


        except Exception as e:
            outcome = f"ERROR: {str(e)[:15]}"

        is_match = outcome == expected
        if is_match:
            correct_count += 1
            status = "PASS [✓]"
        else:
            status = "FAIL [✗]"

        short_query = query[:40] + "..." if len(query) > 40 else query
        print(f"{idx:<3} | {short_query:<42} | {expected:<15} | {outcome:<15} | {status}")

        results.append({
            "query": query,
            "expected_outcome": expected,
            "outcome": outcome,
            "match": "PASS" if is_match else "FAIL",
        })

        # Brief sleep to avoid hitting Groq RPM tier limits
        time.sleep(0.3)

    print("=" * 95)
    accuracy = (correct_count / total) * 100
    print(f"\nFinal Evaluation Score: {correct_count}/{total} correct ({accuracy:.1f}% accuracy)")

    # Save to CSV
    df = pd.DataFrame(results)
    csv_filename = ROOT_DIR / "tests" / "query category evaluation.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results successfully saved to: {csv_filename}\n")


if __name__ == "__main__":
    run_router_eval()