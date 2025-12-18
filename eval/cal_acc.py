import json
import os
import argparse
from tqdm import tqdm
import pandas as pd

EVAL_RESULT_DIR = "../eval_results"
SAVE_DIR = "../eval_results_summary"


score_map = {
    "Good": 2,
    "Medium": 1,
    "Poor": 0
}

CATEGORY = {
    "science": ["chemistry", "physics", "biology", "earth", "math", "medicine", "life"],
    "game": ["chess", "puzzle", "gomoku", "sudoku", "maze", "minesweeper", "number_sliding_puzzle", "sticks", "xiangqi", "Go"],
    "semantic": ["idioms", "metaphors", "definition"],
    "hypothetical": ["subjective_change", "objective_change"],
    "humanity": ["transportation", "sport", "social", "safety", "festival", "dress", "food", "emotion"],
    "vision rule": ["anomaly", "color", "count", "direction", "position", "shape", "size", "style", "view", "motion"]
}



def cal_score_per_instance(
    model_res_dir,
    model_save_dir
):
    """
    calculate score per instance and save to json file
    Arguments:
        model_res_dir: str, model result directory
        model_save_dir: str, model save directory
    """
    for task_name in tqdm(os.listdir(model_res_dir)):
        task_save_folder = os.path.join(model_res_dir, task_name)
        for filename in os.listdir(task_save_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(task_save_folder, filename)
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                tn = data["task_name"]
                index = data["index"]
                

                eval_result_score = {
                    "Instruction Following": [],
                    "Visual Consistency": [],
                    "Visual Fidelity": [],
                    "Rule Coherence": []
                }

                eval_results = data.get("eval_results", [])
                checklist = data.get("checklist", [])

                assert len(eval_results) == len(checklist), "Eval results and checklist length mismatch"

                for idx in range(len(eval_results)):
                    key = list(checklist[idx].keys())[0]
                    ans = eval_results[idx]
                    score = score_map[ans]

                    eval_result_score[key].append(score)

                avg_scores = {
                    "Instruction Following": sum(eval_result_score["Instruction Following"]) / len(eval_result_score["Instruction Following"]) if eval_result_score["Instruction Following"] else 0.0,
                    "Visual Consistency": sum(eval_result_score["Visual Consistency"]) / len(eval_result_score["Visual Consistency"]) if eval_result_score["Visual Consistency"] else 0.0,
                    "Visual Fidelity": sum(eval_result_score["Visual Fidelity"]) / len(eval_result_score["Visual Fidelity"]) if eval_result_score["Visual Fidelity"] else 0.0,
                    "Rule Coherence": sum(eval_result_score["Rule Coherence"]) / len(eval_result_score["Rule Coherence"]) if eval_result_score["Rule Coherence"] else 0.0
                }


                save_data = {
                    "category": data["category"],
                    "task_name": tn,
                    "index": index,
                    "avg_scores": avg_scores
                }

                save_path = os.path.join(model_save_dir, task_name, f"{index}.json")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(save_data, f, indent=4)



def report_acc(
    model_save_dir
):
    """
    report average score per task
    
    """
    dimension_answers = {
        "Instruction Following": [],
        "Visual Consistency": [],
        "Visual Fidelity": [],
        "Rule Coherence": []
    }

    category_answers = {cat: [] for cat in CATEGORY.keys()}
    category_dimension = {cat: {dim: [] for dim in dimension_answers.keys()} for cat in CATEGORY.keys()}
    all_scores = []

    for task_name in os.listdir(model_save_dir):
        task_save_folder = os.path.join(model_save_dir, task_name)
        for filename in os.listdir(task_save_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(task_save_folder, filename)
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                category = data["category"]
                avg_scores = data["avg_scores"]

                for dim in dimension_answers.keys():
                    if not (dim == "Instruction Following" and category == "vision rule"):
                        # vision rule has no Instruction Following dimension
                        dimension_answers[dim].append(avg_scores[dim])
                        category_dimension[category][dim].append(avg_scores[dim])
                        # all_scores.append(avg_scores[dim])

                # calculate average score per category
                if category != "vision rule":
                    category_answers[category].append(
                        (avg_scores["Instruction Following"] +
                         avg_scores["Visual Consistency"] +
                         avg_scores["Visual Fidelity"] +
                         avg_scores["Rule Coherence"]) / 4.0
                    )

                    all_scores.append((avg_scores["Instruction Following"] +
                            avg_scores["Visual Consistency"] +
                            avg_scores["Visual Fidelity"] +
                            avg_scores["Rule Coherence"]) / 4.0)
                else:
                    category_answers[category].append(
                        (avg_scores["Visual Consistency"] +
                         avg_scores["Visual Fidelity"] +
                         avg_scores["Rule Coherence"]) / 3.0
                    )
                    all_scores.append((
                            avg_scores["Visual Consistency"] +
                            avg_scores["Visual Fidelity"] +
                            avg_scores["Rule Coherence"]) / 3.0)


    # total average score
    total_avg = sum(all_scores) / len(all_scores) / 2.0 * 100.0

    # average per dimension
    cat_results = {cat: (sum(scores) / len(scores) / 2.0 * 100.0)  
                   for cat, scores in category_answers.items()}

    # per dimension per category
    cat_dim_results = {
        cat: {dim: (sum(scores) / len(scores) / 2.0 * 100.0 if scores else 0.0)
              for dim, scores in dim_scores.items()}
        for cat, dim_scores in category_dimension.items()
    }

    # output to excel file
    excel_rows = []

    avg = {
        "Instruction Following": [],
        "Visual Consistency": [],
        "Visual Fidelity": [],
        "Rule Coherence": []
    }
    # write category results
    for cat, dims in cat_dim_results.items():
        avg_val = cat_results[cat]
        cat_display = cat.title()
        if cat != "vision rule":
            excel_rows.extend([
                {"Category": cat_display, "Dimension": "IF", "Score": round(dims["Instruction Following"], 2)},
                {"Category": "", "Dimension": "VC", "Score": round(dims["Visual Consistency"], 2)},
                {"Category": "", "Dimension": "VF", "Score": round(dims["Visual Fidelity"], 2)},
                {"Category": "", "Dimension": "RC", "Score": round(dims["Rule Coherence"], 2)},
                {"Category": "", "Dimension": "Avg", "Score": round(avg_val, 2)},
            ])
        else:
            excel_rows.extend([
                {"Category": cat_display, "Dimension": "VC", "Score": round(dims["Visual Consistency"], 2)},
                {"Category": "", "Dimension": "VF", "Score": round(dims["Visual Fidelity"], 2)},
                {"Category": "", "Dimension": "RC", "Score": round(dims["Rule Coherence"], 2)},
                {"Category": "", "Dimension": "Avg", "Score": round(avg_val, 2)},
            ])
        if cat != "vision rule":
            avg["Instruction Following"].append(dims["Instruction Following"])
        avg["Visual Consistency"].append(dims["Visual Consistency"])
        avg["Visual Fidelity"].append(dims["Visual Fidelity"])
        avg["Rule Coherence"].append(dims["Rule Coherence"])

    avg_insruction = sum(avg["Instruction Following"]) / (len(avg["Instruction Following"]))
    avg_visual_consistency = sum(avg["Visual Consistency"]) / len(avg["Visual Consistency"])
    avg_visual_fidelity = sum(avg["Visual Fidelity"]) / len(avg["Visual Fidelity"])
    avg_rule_coherence = sum(avg["Rule Coherence"]) / len(avg["Rule Coherence"])
    total_avg = (avg_insruction + avg_visual_consistency + avg_visual_fidelity + avg_rule_coherence) / 4.0

    # add overall result
    excel_rows.append({"Category": "Overall", "Dimension": "IF", "Score": round(avg_insruction, 2)})
    excel_rows.append({"Category": "", "Dimension": "VC", "Score": round(avg_visual_consistency, 2)})
    excel_rows.append({"Category": "", "Dimension": "VF", "Score": round(avg_visual_fidelity, 2)})
    excel_rows.append({"Category": "", "Dimension": "RC", "Score": round(avg_rule_coherence, 2)})
    excel_rows.append({"Category": "", "Dimension": "Avg", "Score": round(total_avg, 2)})

    df = pd.DataFrame(excel_rows, columns=["Category", "Dimension", "Score"])

    save_path = os.path.join(SAVE_DIR, f"{model_name}_evaluation_summary.xlsx")
    df.to_excel(save_path, index=False)

    print(f"✅ Saved vertical Excel summary to: {save_path}")

if __name__ == "__main__":
    # add arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    model_name = args.model_name

    model_res_dir = os.path.join(EVAL_RESULT_DIR, model_name)
    model_save_dir = os.path.join(SAVE_DIR, model_name)


    # Step1: calculate score per instance and save to json file
    cal_score_per_instance(
        model_res_dir,
        model_save_dir
    )

    # Step 2: report average score per task
    report_acc(
        model_save_dir
    )

"""
python cal_acc.py --model_name veo3_1
"""


                

