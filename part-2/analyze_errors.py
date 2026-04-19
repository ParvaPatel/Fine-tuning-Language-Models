import os
import sqlite3
import threading
from collections import Counter

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def try_execute_sql(query, db_file, timeout_secs=1.0):
    """Executes a SQL query with a hard timeout to prevent Hanging on Cartesian Products."""
    result = {'ok': False, 'records': [], 'err_msg': ''}

    def _run():
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute(query)
            result['records'] = cursor.fetchall()
            result['ok'] = True
            conn.close()
        except Exception as e:
            result['err_msg'] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)
    
    if t.is_alive():
        return False, [], "TIMEOUT: Query hung on a massive Cartesian Product"
    return result['ok'], result['records'], result['err_msg']


def analyze_errors():
    nl_file = "data/dev.nl"
    gt_file = "data/dev.sql"
    # Fallback paths incase they renamed it or ran different job
    pred_file = "results/t5_ft_experiment_job2_dev.sql"
    db_file = "data/flight_database.db"
    
    if not os.path.exists(pred_file):
        print(f"Error: {pred_file} not found.")
        return
        
    nls = load_lines(nl_file)
    gts = load_lines(gt_file)
    preds = load_lines(pred_file)
    
    print(f"Successfully loaded {len(nls)} examples from DEV set.")
    print("Connecting to database and evaluating queries with Timeouts...")
    
    errors = []
    
    for i in range(len(nls)):
        if i % 50 == 0 and i > 0:
            print(f"  -> Processed {i}/{len(nls)} SQL predictions...")
            
        nl = nls[i]
        gt = gts[i]
        pred = preds[i]
        
        if pred == gt:
            continue
            
        # Error Class 1: Execution Syntax Error / Timeout
        pred_ok, pred_records, pred_err = try_execute_sql(pred, db_file)
        if not pred_ok:
            # Categorize the exact failure
            errors.append(('Syntax/Execution Error', nl, pred, gt, f"Failure: {pred_err}"))
            continue
            
        gt_ok, gt_records, _ = try_execute_sql(gt, db_file)
        if not gt_ok:
            continue
            
        if set(gt_records) == set(pred_records):
            continue # Correct by execution
            
        # Error Class 2: Under-constrained
        if len(pred_records) > len(gt_records):
            errors.append(('Under-Constrained (Missing WHERE)', nl, pred, gt, "The model failed to translate all linguistic constraints into SQL, returning too many general rows."))
            continue
            
        # Error Class 3: Over-constrained (Empty Result)
        if len(pred_records) == 0 and len(gt_records) > 0:
            errors.append(('Over-Constrained (Empty Result)', nl, pred, gt, "The model hallucinated strict/conflicting WHERE constraints, causing the query to return zero records."))
            continue
            
        errors.append(('Data Retrieval Mismatch', nl, pred, gt, "The query executed effectively but retrieved different/wrong columns compared to ground truth."))
        
    counts = Counter([e[0] for e in errors])
    
    print("\n" + "="*50)
    print("TABLE 5: QUALITATIVE ERROR ANALYSIS OUTPUT")
    print("="*50)
    
    for err_type, count in counts.items():
        print(f"\n--- {err_type} ---")
        print(f"Statistics: {count}/{len(nls)}")
        
        example = next(e for e in errors if e[0] == err_type)
        print(f"1. Natural Language : {example[1]}")
        print(f"2. Model Prediction : {example[2]}")
        print(f"3. Expected SQL (GT): {example[3]}")
        print(f"4. Error Description: {example[4]}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_errors()
