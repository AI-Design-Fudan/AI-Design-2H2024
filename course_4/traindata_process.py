from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("SkunkworksAI/reasoning-0.01")


def format_entry(entry):
    formatted_entry = [
        {"role": "system", "content": "You are a Yaotelligent, a helpful assistant\n"},
        {"role": "user", "content": f"{entry['instruction']}\n"},
        {"role": "assistant", "content": f"<|reasoning_start|>{entry['reasoning_chains']}<|reasoning_end|>\n{entry['output']}"}
    ]
    return formatted_entry


# Apply formatting to the train dataset
formatted_data = [format_entry(entry) for entry in dataset['train']]

# Save to a JSON file
output_path = "formatted_reasoning_dataset.json"
with open(output_path, 'w') as f:
    json.dump(formatted_data, f, indent=2)

print(f"Formatted data saved to {output_path}")