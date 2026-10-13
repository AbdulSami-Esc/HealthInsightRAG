import os
import sys
from unittest.mock import MagicMock
import pytest

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.prompt_manager as pm
from src.prompt_manager import (
    TOKEN_THRESHOLD_SMALL,
    build_managed_context,
    count_context_tokens,
    stratified_temporal_window,
)

pm.combined_text = ""


P09001_RAW_CHUNKS = [
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 0,
        "text": (
            "CHIEF COMPLAINT AND DETAILED HISTORY OF PRESENT ILLNESS: A 54-year-old male presents to the acute clinical center "
            "with severe, progressive left lower quadrant abdominal pain radiating across the pelvic brim into the suprapubic area, "
            "accompanied by continuous nausea, anorexia, low-grade fevers, and chills over the preceding 72 hours. The discomfort began "
            "insidiously as dull cramping before escalating significantly to an 8 out of 10 on the visual analog pain scale, exacerbated by "
            "ambulation, coughing, and positional transitions. The patient notes obstipation without bowel movements or passage of flatus "
            "for 48 hours prior to evaluation. He explicitly denies overt gastrointestinal bleeding, hematochezia, melena, hematemesis, "
            "or previous history of colon polyps. Past medical history is notable for essential hypertension maintained on lisinopril "
            "10 mg daily and hyperlipidemia managed with atorvastatin 20 mg nightly, with no previous history of abdominal operations, "
            "diverticular flare-ups, or inflammatory bowel disease. Social history is negative for tobacco abuse, illicit substance use, "
            "or chronic alcohol intake. Family history is negative for early-onset colorectal carcinoma or inflammatory bowel disorders. "
            "Continuous cardiac monitoring is maintained alongside pulse oximetry. Peripheral IV access is verified and flushed with normal saline without infiltration."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 1,
        "text": (
            "PHYSICAL EXAMINATION AND COMPREHENSIVE VITALS: Initial clinical vitals demonstrate a blood pressure of 138/84 mmHg, "
            "heart rate 98 beats per minute regular rhythm, respiratory rate 18 breaths per minute, oxygen saturation 98% on ambient air, "
            "and tympanic temperature 100.8 degrees Fahrenheit. General appearance: The patient is alert, oriented to person, place, and time, "
            "lying in moderate acute distress with hips flexed in the lateral decubitus position to minimize peritoneal irritation. "
            "HEENT exam shows normocephalic features, moist mucous membranes without pharyngeal exudates. Cardiovascular exam reveals regular "
            "rate and rhythm, normal S1 and S2 without audible gallops, rubs, or pathological murmurs. Respiratory exam shows lungs clear to "
            "auscultation bilaterally throughout all lung fields. Abdominal examination demonstrates mild-to-moderate abdominal distension, "
            "severe direct tenderness localized strictly over the sigmoid colon in the left lower quadrant, accompanied by focal involuntary "
            "muscular guarding. There are no signs of generalized peritoneal irritation, absence of board-like abdominal rigidity, and no "
            "diffuse rebound tenderness. Auscultation reveals bowel sounds that are markedly hypoactive in all quadrants. "
            "The patient remains stable under the direct supervision of the clinical team with vital signs and pain scores recorded on a regular schedule."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 2,
        "text": (
            "LABORATORY WORKUP AND SERUM BIOCHEMISTRY: Initial diagnostic workup demonstrates a pronounced peripheral leukocytosis "
            "with a total white blood cell count of 15.4 K/uL, showing 82% segmented neutrophils and 6% band forms. Hemoglobin is stable at "
            "14.1 g/dL, hematocrit 42.4%, and platelet count 260 K/uL. Quantitative serum inflammatory markers reveal marked systemic elevation "
            "of C-reactive protein at 48 mg/L and erythrocyte sedimentation rate at 38 mm/hr. Arterial and venous serum lactate levels are reassuringly "
            "within reference limits at 1.2 mmol/L. Comprehensive metabolic profile reveals serum sodium 139 mEq/L, potassium 4.1 mEq/L, "
            "chloride 102 mEq/L, bicarbonate 24 mEq/L, blood urea nitrogen 19 mg/dL, serum creatinine 1.0 mg/dL, and fasting serum glucose "
            "112 mg/dL. Hepatic enzymes including alanine aminotransferase, aspartate aminotransferase, total bilirubin, and alkaline phosphatase "
            "are completely normal. Clean-catch urinalysis demonstrates absence of pyuria, leukocyte esterase, nitrites, and red blood cells. "
            "Nursing staff confirms the patient is comfortable and following all clinical directives as documented in the protocol."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 3,
        "text": (
            "DIAGNOSTIC CROSS-SECTIONAL IMAGING: Intravenous and oral contrast-enhanced computed tomography of the abdomen and pelvis "
            "reveals marked segmental circumferential wall thickening of the mid-sigmoid colon measuring up to 7 mm in cross-sectional wall width. "
            "Pronounced hyperemic enhancement, mesenteric vessel engorgement, and extensive surrounding pericolic mesenteric fat stranding are "
            "prominently demonstrated. A 1.2 cm peridiverticular micro-abscess and localized phlegmonous reaction is visualized abutting the "
            "antimesenteric serosal border, without evidence of macroscopic extraluminal perforation or pelvic fluid collection. There is no "
            "pneumoperitoneum, retroperitoneal gas, or free subdiaphragmatic air visualized. The appendix, liver, gallbladder, pancreas, "
            "spleen, adrenal glands, and bilateral kidneys are radiologically unremarkable with no evidence of obstruction or calculi. "
            "No distant metastatic deposits or pathological retroperitoneal lymphadenopathy identified."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 4,
        "text": (
            "CLINICAL ASSESSMENT AND STAGING: The clinical findings and radiological presentation are consistent with acute sigmoid diverticulitis, "
            "classified under the Modified Hinchey Classification as Stage Ia, characterized by localized pericolic inflammation, micro-perforation, "
            "and phlegmonous infiltration confined to the mesenteric border. Hemodynamic parameters remain entirely compensated without signs of "
            "systemic inflammatory response syndrome deterioration, metabolic acidosis, or septic shock. After multidisciplinary discussion, emergent "
            "operative intervention or Hartmann procedure is deferred in favor of admission for a non-operative medical strategy with broad-spectrum "
            "intravenous antibiotic therapy and rigorous serial clinical monitoring. Baseline electrocardiogram shows sinus rhythm without acute ischemic changes."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 5,
        "text": (
            "INPATIENT ORDERS AND INTRAVENOUS RESUSCITATION: The patient is admitted to the inpatient gastroenterology service under continuous "
            "nursing care. The patient is placed on strict nil per os (NPO) status to facilitate gastrointestinal decompression and bowel rest, "
            "alleviating luminal pressures across the inflamed sigmoid segments. Resuscitative intravenous fluid hydration is initiated using Plasmalyte "
            "solution at a continuous infusion rate of 125 mL/hr via peripheral venous catheter. Daily complete metabolic panels and morning complete "
            "blood counts are scheduled to track inflammatory downtrends. Mandated serial clinical abdominal examinations are scheduled every "
            "6 hours to assess for signs of surgical abdomen or peritonitis."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 6,
        "text": (
            "ANTIMICROBIAL REGIMEN AND PHARMACOLOGY: Empirical broad-spectrum intravenous antimicrobial coverage is initiated immediately to provide "
            "coverage against enteric gram-negative bacilli and anaerobic organisms, consisting of Ceftriaxone 1g IV daily combined with Metronidazole "
            "500mg IV every 8 hours. Multimodal non-narcotic analgesia is established with intravenous acetaminophen 1000mg administered every "
            "8 hours as needed for discomfort. Opioid analgesics including morphine and oxycodone are specifically withheld to prevent secondary "
            "hypomotility, medication-induced paralytic ileus, and diagnostic masking of acute peritoneal signs. Intravenous ondansetron 4mg is "
            "made available PRN for nausea."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 7,
        "text": (
            "SURGICAL CONSULTATION AND SAFETY CRITERIA: The acute care general surgical team evaluated the patient at bedside. The surgical "
            "consultants concur with conservative medical management given the localized nature of the Modified Hinchey Stage Ia phlegmon. "
            "Precise operative indications and escalation triggers are explicitly documented in the medical record: persistent spikes in fever "
            "beyond 72 hours of antimicrobial therapy, rapid leukocytosis elevation, worsening diffuse abdominal tenderness, signs of systemic sepsis, "
            "or evidence of macro-perforation on repeat imaging. In the absence of these triggers, non-operative inpatient management remains indicated."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 8,
        "text": (
            "HOSPITAL COURSE AND PROGRESSION DAYS 2-3: Over hospital days 2 and 3, the patient exhibits significant clinical improvement. "
            "Left lower quadrant tenderness subsides substantially, and voluntary guarding resolves completely. Body temperature remains afebrile "
            "at 98.4 F throughout. Repeat complete blood count shows marked leukocytosis resolution with white blood cell count downtrending "
            "to 9.8 K/uL. The patient successfully passes flatus and reports return of appetite. The diet is transitioned from clear liquids to full "
            "liquids, followed by a low-residue oral diet which the patient tolerates without nausea, discomfort, or distension."
        ),
    },
    {
        "encounter_id": "E09901",
        "patient_id": "P09001",
        "visit_date": "2026-01-10",
        "visit_number": 1,
        "bucket_name": "transcription",
        "chunk_position": 9,
        "text": (
            "DISCHARGE STATUS AND HOME INSTRUCTIONS: The patient is discharged in stable condition on hospital day 4. He is prescribed a 10-day "
            "outpatient course of oral Ciprofloxacin 500mg twice daily and oral Metronidazole 500mg three times daily to complete antibiotic "
            "eradication. Comprehensive instructions are given regarding activity modification, avoidance of heavy lifting, and strict red-flag "
            "precautions: high fevers, intractable vomiting, inability to maintain hydration, or sharp recurrences of abdominal pain. "
            "The patient is scheduled for routine clinic follow-up in two weeks."
        ),
    },
    {
        "encounter_id": "E09902",
        "patient_id": "P09001",
        "visit_date": "2026-01-24",
        "visit_number": 2,
        "bucket_name": "transcription",
        "chunk_position": 0,
        "text": (
            "OUTPATIENT FOLLOW-UP EVALUATION AND CLINICAL REVIEW: The patient presents for a scheduled 2-week clinical evaluation following "
            "hospital discharge for acute sigmoid diverticulitis. He confirms completing the entire 10-day outpatient antimicrobial course "
            "of oral Ciprofloxacin and Metronidazole with high compliance and zero adverse reactions. The patient endorses complete resolution "
            "of left lower quadrant abdominal discomfort, no fevers, and no chills. Bowel movements have returned to his normal daily baseline "
            "pattern, producing soft formed stools without tenesmus, hematochezia, or dark stools. Appetite and functional capacity are fully restored. "
            "Review of systems is negative for nausea, weight loss, or constitutional complaints. Medication adherence is confirmed. "
            "The patient expresses satisfaction with clinical recovery and understands the necessity of ongoing surveillance."
        ),
    },
    {
        "encounter_id": "E09902",
        "patient_id": "P09001",
        "visit_date": "2026-01-24",
        "visit_number": 2,
        "bucket_name": "transcription",
        "chunk_position": 1,
        "text": (
            "LONG-TERM GASTROENTEROLOGY PLAN AND PHYSICAL EXAM: Physical examination reveals normal resting vitals: blood pressure 124/78 mmHg, "
            "heart rate 72 bpm regular, and afebrile. Abdominal examination reveals a soft, non-distended abdomen with zero tenderness across "
            "all four quadrants on deep palpation, without organomegaly or masses. Normoactive bowel sounds are auscultated. Laboratory reassessment "
            "confirms complete biochemical normalization with C-reactive protein at 3.1 mg/L and white blood cell count at 6.8 K/uL. "
            "The patient is advised to transition gradually to a high-fiber maintenance dietary regimen and maintain generous hydration. "
            "A screening colonoscopy is scheduled in 8 weeks to evaluate the colonic mucosa, rule out underlying polyps, and verify structural resolution."
        ),
    },
    {
        "encounter_id": "E09903",
        "patient_id": "P09001",
        "visit_date": "2026-03-15",
        "visit_number": 3,
        "bucket_name": "transcription",
        "chunk_position": 0,
        "text": (
            "PRIMARY CARE PRESENTATION FOR ACUTE RHINOPHARYNGITIS: The patient presents two months after his gastroenterology follow-up "
            "for an unrelated acute primary care consultation complaining of 48 hours of scratchy sore throat, continuous watery rhinorrhea, "
            "nasal congestion, sneezing paroxysms, and mild subjective chills with generalized fatigue. The patient expressly denies abdominal pain, "
            "nausea, vomiting, diarrhea, constipation, or changes in stool frequency or character. Gastrointestinal review of systems is completely "
            "negative and non-contributory. The patient reports stable baseline health up until the onset of upper respiratory symptoms two days ago. "
            "No sick contacts at home, but several coworkers recently reported similar viral upper respiratory symptoms."
        ),
    },
    {
        "encounter_id": "E09903",
        "patient_id": "P09001",
        "visit_date": "2026-03-15",
        "visit_number": 3,
        "bucket_name": "transcription",
        "chunk_position": 1,
        "text": (
            "EXAMINATION FINDINGS AND SUPPORTIVE TREATMENT PLAN: Vitals show oral temperature 100.2 F, blood pressure 120/76 mmHg, pulse 78 bpm, "
            "and oxygen saturation 99% on ambient air. Physical exam shows mild erythematous posterior pharyngeal injection without tonsillar exudate, "
            "ulcers, or cervical adenopathy. Bilateral lung fields are clear to auscultation without wheezing, rales, or rhonchi. Abdomen is flat, "
            "soft, entirely non-tender, and benign. Point-of-care rapid antigen testing is negative for Group A Streptococcus, Influenza A/B, "
            "and COVID-19. Diagnosis: Acute viral upper respiratory tract infection. Plan: Supportive management including hydration, saline nasal rinses, "
            "warm throat lozenges, and OTC acetaminophen 650mg PRN for low-grade fever. Return precautions outlined if symptoms persist beyond 10 days."
        ),
    },
]


@pytest.fixture
def mock_groq_client():
    client = MagicMock()

    def fake_chat_completion(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = messages[-1]["content"] if messages else ""

        if "Visit 1" in prompt_text:
            digest = "1. Acute sigmoid diverticulitis (Modified Hinchey Stage Ia).\n2. LLQ pain, leukocytosis (WBC 15.4), and micro-abscess on CT.\n3. IV Ceftriaxone and IV Metronidazole therapy."
        elif "Visit 2" in prompt_text:
            digest = "1. Follow-up for resolved acute sigmoid diverticulitis.\n2. Asymptomatic, normalized WBC (6.8) and CRP (3.1).\n3. Colonoscopy scheduled in 8 weeks."
        elif "Visit 3" in prompt_text:
            digest = "1. Acute viral upper respiratory tract infection (URI).\n2. Benign abdominal exam, negative Strep/Flu.\n3. Supportive care with hydration and rest."
        else:
            digest = "Clinical digest summary."

        choice = MagicMock()
        choice.message.content = digest
        response = MagicMock()
        response.choices = [choice]
        return response

    client.chat.completions.create.side_effect = fake_chat_completion
    return client


def print_scorecard(metrics: dict) -> None:
    width = 60
    raw = metrics["raw"]
    compressed = metrics["compressed"]
    ratio = (compressed / raw) * 100 if raw else 0.0
    
    print("\n" + "=" * width)
    print(f"{'MAP-REDUCE STRESS TEST SCORECARD':^{width}}")
    print("=" * width)
    print(f"  - Raw Input Tokens:    {raw:>7}")
    print(f"  - Compressed Tokens:   {compressed:>7}")
    print(f"  - Compression Ratio:   {ratio:>6.2f}% ({100 - ratio:.2f}% saved)")
    print(f"  - Map LLM Calls:       {metrics['llm_calls']:>7} call(s)")
    print("-" * width)
    for milestone, passed in metrics["milestones"].items():
        status = "[ PASS ]" if passed else "[ FAIL ]"
        print(f"  {status} {milestone}")
    print("=" * width + "\n")


def test_map_reduce_stress_pipeline(mock_groq_client):
    raw_tokens = count_context_tokens(P09001_RAW_CHUNKS)
    windowed_chunks = stratified_temporal_window(P09001_RAW_CHUNKS)
    windowed_tokens = count_context_tokens(windowed_chunks)

    # Assert token volume successfully breaches small threshold
    assert raw_tokens > 2000
    assert windowed_tokens > TOKEN_THRESHOLD_SMALL

    # Execute managed context generation (triggers Map-Reduce branch)
    managed_context = build_managed_context(P09001_RAW_CHUNKS, mock_groq_client)
    compressed_tokens = count_context_tokens(managed_context)
    normalized = managed_context.lower()

    # Milestone verification
    milestones = {
        "Diverticulitis Diagnosis": "diverticulitis" in normalized,
        "Hinchey Staging Reference": "hinchey" in normalized,
        "Ceftriaxone Treatment": "ceftriaxone" in normalized,
        "Metronidazole Treatment": "metronidazole" in normalized,
        "Colonoscopy Follow-up": "colonoscopy" in normalized,
        "Viral URI (Visit 3 Isolated)": "respiratory" in normalized,
    }

    metrics_payload = {
        "raw": raw_tokens,
        "compressed": compressed_tokens,
        "llm_calls": mock_groq_client.chat.completions.create.call_count,
        "milestones": milestones,
    }

    print_scorecard(metrics_payload)

    # Core assertions
    assert mock_groq_client.chat.completions.create.call_count == 3
    assert compressed_tokens < raw_tokens
    assert (compressed_tokens / raw_tokens) < 0.60
    assert all(milestones.values())

    # Temporal isolation check (Visit 3 should not contain Visit 1 terms)
    if "--- VISIT 3 CLINICAL DIGEST ---" in managed_context:
        visit_3_part = managed_context.split("--- VISIT 3 CLINICAL DIGEST ---")[1].lower()
        assert "diverticulitis" not in visit_3_part