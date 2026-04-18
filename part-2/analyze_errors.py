import os
import sqlite3
from collections import Counter

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def analyze_errors():
    nl_file = "data/dev.nl"
    gt_file = "data/dev.sql"
    pred_file = "results/t5_ft_experiment_job2_dev.sql"
    db_file = "data/flight_database.db"
    
    if not os.path.exists(pred_file):
        print(f"Error: {pred_file} not found. Please ensure you run this on the HPC where your results folder exists!")
        return
        
    nls = load_lines(nl_file)
    gts = load_lines(gt_file)
    preds = load_lines(pred_file)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    errors = []
    
    for i in range(len(nls)):
        nl = nls[i]
        gt = gts[i]
        pred = preds[i]
        
        if pred == gt:
            continue
            
        # Error Class 1: Execution Syntax Error
        exec_error = False
        try:
            cursor.execute(pred)
        except Exception as e:
            exec_error = True
            errors.append(('Syntax/Execution Error', nl, pred, gt, f"SQLite crashed with error: {str(e)}"))
            continue
            
        # Fetching records to compare outputs
        try:
            cursor.execute(gt)
            gt_records = cursor.fetchall()
            cursor.execute(pred)
            pred_records = cursor.fetchall()
        except Exception:
            continue
            
        if set(gt_records) == set(pred_records):
            continue # Correct by execution metric
            
        # Error Class 2: Missing Constraints (Under-constrained)
        # E.g., The model missed a WHERE condition causing it to return way too many generic rows
        if len(pred_records) > len(gt_records):
            errors.append(('Under-Constrained (Missing WHERE)', nl, pred, gt, "The model failed to translate all linguistic constraints into SQL, returning too many rows."))
            continue
            
        # Error Class 3: Hallucinated/Over-constrained (Empty Result)
        # E.g., The model hallucinated a wrong date, wrong airline code, or conflicting WHERE clause
        if len(pred_records) == 0 and len(gt_records) > 0:
            errors.append(('Over-Constrained (Empty Result)', nl, pred, gt, "The model hallucinated strict/conflicting WHERE constraints, causing the query to return zero records."))
            continue
            
        # Error Class 4: Incorrect Data Retrieval
        errors.append(('Data Retrieval Mismatch', nl, pred, gt, "The query executed successfully but retrieved overlapping/wrong columns or distinct values compared to ground truth."))
        
    conn.close()
    
    counts = Counter([e[0] for e in errors])
    
    print("\n" + "="*50)
    print("TABLE 5: QUALITATIVE ERROR ANALYSIS OUTPUT")
    print("="*50)
    
    for err_type, count in counts.items():
        print(f"\n--- {err_type} ---")
        print(f"Statistics: {count}/{len(nls)}")
        
        # Grab first example of this error
        example = next(e for e in errors if e[0] == err_type)
        print(f"1. Natural Language : {example[1]}")
        print(f"2. Model Prediction : {example[2]}")
        print(f"3. Expected SQL (GT): {example[3]}")
        print(f"4. Error Description: {example[4]}")

if __name__ == "__main__":
    analyze_errors()
