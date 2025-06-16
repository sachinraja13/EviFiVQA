# Financial Document VQA Dataset

This repository contains a dataset for Visual Question Answering (VQA), specifically designed for tasks involving the interpretation of tables and data from financial documents. Each record in the dataset consists of an image of a document snippet, a question about the image, and a detailed ground truth answer.

The dataset is generated using a Python script that processes source JSON annotations and their corresponding images.
Dataset Structure

The dataset is saved using the Hugging Face datasets library. Each record has the following features:
Feature	Type	Description
image	string	The file path to the PNG image of the document snippet.
question	string	The (LLM-paraphrased) question asked about the content of the image.
difficulty_level	string	A numeric string ("1", "2", etc.) indicating the question's complexity.
ground_truth	string	A pretty-printed JSON string containing the detailed answer information.
The ground_truth Field

The ground_truth column is a JSON formatted string and is the core of the answer data. To use it, you must first parse it into a Python dictionary.
Python


## Example from a loaded dataset
```
import json
record = dataset[0]
gt_string = record['ground_truth']
gt_dictionary = json.loads(gt_string)
```

The resulting dictionary can contain the following keys. Keys are only present if they exist in the source annotation data.
Key	Type	Description
answer	string	The primary, direct answer to the question.
answer_bbox	list[float]	A list of four floats representing the bounding box [x1, y1, x2, y2] of the answer in the image.
individual_answers	list[string]	A list of strings representing component answers that may make up the primary answer. Note: If the original source for this field was a JSON object, it is serialized into a single string element within this list.
individual_answers_bboxes	list[list[float]]	A list of bounding boxes, where each box corresponds to an answer in individual_answers.
answer_key	string	The text from the corresponding row header or stub that provides context for the answer (e.g., "Due after three years").
answer_key_bbox	list[float]	The bounding box for the answer_key text.
How to Use

The dataset is saved in the efficient Apache Arrow format via the save_to_disk method.
### Loading the Dataset

To load the dataset from the saved directory, use the load_from_disk function from the datasets library.
Python
```
from datasets import load_from_disk

## Load the dataset from the directory where it was saved


dataset_path = "./my_vqa_dataset"
dataset = load_from_disk(dataset_path)

print(dataset)
```
    Output:

    Dataset({
        features: ['image', 'question', 'difficulty_level', 'ground_truth'],
        num_rows: 2
    })

### Accessing and Parsing Data

To work with the answer data, you need to select a record and parse the ground_truth string.
Python
```
import json

## Get the first record from the dataset
record = dataset[0]

## The ground_truth is a string
gt_string = record['ground_truth']
print("--- Ground Truth (as string) ---")
print(gt_string)

## Parse the string into a Python dictionary
gt_dictionary = json.loads(gt_string)
print("\n--- Ground Truth (as dictionary) ---")
print(gt_dictionary)

## Now you can access individual keys
answer = gt_dictionary.get('answer')
print(f"\nPrimary Answer: {answer}")
```
### Example Record

Here is an example of what a single record in the dataset looks like.
```

{
  'image': 'val/images/val_sample_0.png',
  'question': 'Could you please detail the total dollar amount (expressed in thousands) representing the investment in common stocks during the calendar year 2017?',
  'difficulty_level': '1',
  'ground_truth': '{\n    "answer": "81855",\n    "answer_bbox": [\n        0.4674,\n        0.3144,\n        0.5552,\n        0.3662\n    ],\n    "individual_answers": [\n        "80000",\n        "1855"\n    ]\n}'
}
```
## Licence

This dataset is licensed under the MIT License.
