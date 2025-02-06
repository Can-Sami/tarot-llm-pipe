from datasets import load_dataset
from huggingface_hub import login
import json


login("HF_TOKEN")

ds = load_dataset("cansani/my-distiset-8be10284", "default")

# Write dataset entries to a JSON file
data = []
for idx, entry in enumerate(ds['train']):
    data.append({
        'entry_id': idx + 1,
        'system_prompt': entry['system_prompt'],
        'user_prompt': entry['prompt'],
        'completion': entry['completion']
    })

with open('dataset_entries.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)


