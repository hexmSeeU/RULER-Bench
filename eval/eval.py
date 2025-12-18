import os
import json
import argparse
import base64
from openai import OpenAI
import concurrent.futures
import openai
from tqdm import tqdm
from pathlib import Path
import re
import cv2

client = OpenAI(
        base_url="your base_url",
        api_key="your api_key"
)

prefix = "../RULER-Bench/"

# The parent directory of the inference result
res_video_dir = "..."

DATA_JSON_PATTH = "../RULER-Bench/data.jsonl"
SAVE_DIR = "../eval_results"

system_prompt_file = "./system_prompt.txt"
with open(system_prompt_file, "r", encoding="utf-8") as f:
    system_prompt = f.read()


def text_format(text):
    return {"type": "text", "text": text}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def img_format(image_path):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}

def resize_keep_ratio(image, max_side=1024):
    """
    Scale the image so its longest side ≤ max_side, preserving the aspect ratio
    """
    h, w = image.shape[:2]
    long_side = max(h, w)

    # If it is already smaller than max_side, do not resize.
    if long_side <= max_side:
        return image

    scale = max_side / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def encode_image_from_array(image_array):
    """
    Convert an OpenCV image to a base64 string
    """
    # convert to JPEG format
    success, buffer = cv2.imencode('.jpg', image_array)
    if not success:
        raise ValueError("Frame encoding failed.")
    return base64.b64encode(buffer).decode('utf-8')

def img_format_from_array(image_array):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image_from_array(image_array)}"
        }
    }

def extract_frames(video_path, fps=2, max_side=512):
    """
    Extract fps frames per second (default 2), 
    scale them proportionally after extraction, 
    and return a list in the model's input format (base64).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1

    frame_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = resize_keep_ratio(frame, max_side=max_side)

            frame_list.append(img_format_from_array(frame))

        frame_idx += 1

    cap.release()
    return frame_list



def format_check(response, correct_len):
    try:
        response = response.strip()
        # remove markdown code block markers
        response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
        response = re.sub(r"```$", "", response)

        # extract <answer>...</answer> 
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            print("No <answer> tag found.")
            return False

        answer_text = match.group(1).strip()
        # parse json
        sample = json.loads(answer_text)

        print("Parsed sample:", sample)

        if isinstance(sample, list) and len(sample) == correct_len:
            for item in sample:
                if item not in ["Good", "Medium", "Poor"]:
                    print("Invalid item:", item)
                    return False
            return True
        else:
            print(f"Length mismatch: expected {correct_len}, got {len(sample)}")
            return False
    except Exception as e:
        print("Failed to parse GPT output:", e)
        return False
def qwen_vl_processor(sample, model_name):
    content = []
    checklist = sample["checklist"]
    prompt = sample["prompt"]
    input_image_path = sample.get("input_image_path", "")
    input_video_path = sample.get("input_video_path", "")
    gt_image_path = sample.get("gt_image_path", "")
    implicit_explanation = sample.get("implicit_explanation", "")
    if input_image_path:

        input_image_full_path = f"{prefix}{input_image_path}"

        input_image_text = "\n## Input Image:\n"
        content.append(text_format(input_image_text))
        content.append(img_format(input_image_full_path))
    
    prompt_text = f"\n## Input Prompt:\n{prompt}\n"
    content.append(text_format(prompt_text))

    if gt_image_path:
        gt_image_full_path = f"{prefix}{gt_image_path}"

        gt_image_text = "\n## Ground truth Image:\n"
        content.append(text_format(gt_image_text))
        content.append(img_format(gt_image_full_path))
    
    if implicit_explanation:
        implicit_explanation_text = f"\n## Implicit Explanation:\n{implicit_explanation}\n"
        content.append(text_format(implicit_explanation_text))
    
    if input_video_path:
        input_video_full_path = f"{prefix}{input_video_path}"

        input_video_text = "\n## Input Video:\n"
        content.append(text_format(input_video_text))
        content.extend(extract_frames(input_video_full_path, fps=2))

    video_path = os.path.join(res_video_dir, model_name, sample["task_name"], f"{sample['index']}.mp4")

    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return None

    video_text = "\n## Video to evaluate:\n"
    content.append(text_format(video_text))

    content.extend(extract_frames(video_path, fps=2))

    qa_text = "\n## CheckList Questions:\n"
    content.append(text_format(qa_text))

    for idx, check_item in enumerate(checklist):
        key = list(check_item.keys())[0]
        question = check_item[key]

        question_text = f"\n# Question {idx+1}:\n{question}\n"

        content.append(text_format(question_text))
    
    message = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": content}
    ]
    

    response = client.chat.completions.create(
        model="o3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        presence_penalty=0,
    )
    response_text = response.choices[0].message.content

    if not format_check(response_text, len(checklist)):
        print("❌ Response format is incorrect. Skipping save.")
        return None
    else:
        # clean up markdonw code block markers
        response_text = response_text.strip()
        response_text = re.sub(r"^```[a-zA-Z]*\n?", "", response_text)
        response_text = re.sub(r"```$", "", response_text)

        # extract the contents of <think> and <answer> 
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

        think_text = think_match.group(1).strip() if think_match else ""
        answer_text = answer_match.group(1).strip() if answer_match else "[]"

        try:
            eval_results = json.loads(answer_text)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse <answer> JSON:", e)
            return None

        # create save directory if not exists
        task_save_dir = os.path.join(SAVE_DIR, model_name, sample["task_name"])
        os.makedirs(task_save_dir, exist_ok=True)

        # assemble save data
        save_data = sample.copy()
        save_data["reasoning"] = think_text
        save_data["eval_results"] = eval_results

        # save to json file
        save_path = os.path.join(task_save_dir, f"{sample['index']}.json")
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)

        print("✅ Response format is correct. Saved to", save_path)
        return save_data

    


if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    model_name = args.model_name

    samples = []


    with open(DATA_JSON_PATTH, "r", encoding='utf-8') as f:
        for line in f:
            sample_data = json.loads(line)
            task_name = sample_data["task_name"]
            index = sample_data["index"]

            # check whether the sample has already been processed
            to_save_file_path = os.path.join(SAVE_DIR, model_name, task_name, f"{index}.json")
            if not os.path.exists(to_save_file_path):
                samples.append(sample_data)

    print("Total samples to process:", len(samples))

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        list(tqdm(executor.map(qwen_vl_processor, samples, [model_name] * len(samples)), total=len(samples), desc="Processing files"))

"""
python eval.py --model_name your_model_name
"""