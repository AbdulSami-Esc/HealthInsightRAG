import pandas as pd
##this is the file for tool function definition

##This function can help to perform operations like count & average
def get_condition_stats(df: pd.DataFrame, condition: str, operation: str = "count", column: str = "Billing Amount"):
    #3this is doen to ensure that real condition is types in by the user
    if not isinstance(condition, str) or not condition.strip():
        return {"status": "error", "message": "condition must be a non-empty string"}

    operation = operation.strip().lower()
    if operation not in ("count", "average"):
        return {"status": "error", "message": f"unsupported operation '{operation}', expected 'count' or 'average'"}

    mask = df["Medical Condition"].fillna("").str.lower() == condition.strip().lower()
    matches = df[mask]

    if matches.empty:
        return {
            "status": "not_found",
            "message": f"No patients found with condition '{condition}'."
        }

    if operation == "count":
        return {
            "status": "ok",
            "condition": condition,
            "operation": "count",
            "value": int(matches.shape[0])
        }
    ##MKoree safety features to ensure that user doent make any generic case insensitivity mistakes
    if column not in df.columns:
        return {"status": "error", "message": f"column '{column}' not found in dataset"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"column '{column}' is not numeric, cannot average"}

    avg_value = matches[column].mean()
    return {
        "status": "ok",
        "condition": condition,
        "operation": "average",
        "column": column,
        "value": round(float(avg_value), 2)
    }

##Defines high risk patient
def find_high_risk_patients(df: pd.DataFrame, limit: int = 20):
    required_cols = {"Test Results", "Admission Type", "Name", "Medical Condition", "Hospital"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"status": "error", "message": f"missing expected columns: {missing}"}

    mask = (df["Test Results"] == "Abnormal") & (df["Admission Type"] == "Emergency")
    matches = df[mask]

    if matches.empty:
        return {"status": "not_found", "message": "No high-risk patients found."}

    total_matches = int(matches.shape[0])
    selected = matches[["Name", "Medical Condition", "Admission Type", "Hospital"]].head(limit)

    return {
        "status": "ok",
        "total_matches": total_matches,
        "returned": int(selected.shape[0]),
        "truncated": total_matches > limit,
        "patients": selected.to_dict(orient="records")
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/healthcare_dataset.csv")

    print(get_condition_stats(df, "Diabetes", "count"))
    print(get_condition_stats(df, "Diabetes", "average"))
    print(get_condition_stats(df, "NotARealCondition", "count"))
    print(get_condition_stats(df, "Diabetes", "median"))  # should error gracefully

    result = find_high_risk_patients(df, limit=5)
    print(result["status"], result.get("total_matches"), result.get("returned"))