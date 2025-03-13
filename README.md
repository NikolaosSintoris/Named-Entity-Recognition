# Named-Entity-Recognition
A text normalization technique that extracts only the writers' names from a given string containing information about authors, production details, and other metadata.

# Goal
The goal of this task is to evaluate the ability to design and implement normalization techniques for textual data. Having the raw composition writer’s information as it has been given, propose a solution that generally can normalize the given text, removing redundant information, and keeping only the writer names in the output. The solution should be generalizable to unseen data while demonstrating proficiency in handling complex and potentially redundant text information.

# Data
The dataset provided is a CSV file containing raw compositions from writers' texts and the normalized version. The description of each column in the dataset is provided below:
- raw_comp_writers_text: Original text containing composer and writer information
- CLEAN_TEXT: The clean normalized text after removing redundant information.

# Data Analysis
Our data consist of 10000 rows and 2 columns.

Some entries in the CLEAN_TEXT column are missing. More specifically 1341 records are missing. Also, the format of raw_comp_writers_text appears inconsistent (some names are separated by slashes ('/'), some by commas (',') and some by special character '&'. More specifically:
- 3998 entries use '/'
- 993 entries use ','
- 961 entries use '&'


# Proposed Solution

## Introduction
We aim to implement a text normalization technique that extracts only the writers' names from a given string containing information about authors, production details, and other metadata. This task involves removing redundant information and focusing solely on identifying names, making it a named entity recognition (NER) problem.

## Named Entity Recognition (NER)
Named Entity Recognition (NER) is a fundamental technique in Natural Language Processing (NLP) designed to identify and classify specific entities within unstructured text. These entities may include names of people, organizations, locations, dates, and other relevant categories. By transforming raw text into structured data, NER enhances machine comprehension and processing. The process involves detecting potential entities and assigning them to predefined categories.

Key Concepts of NER:
- Tokenization: The first step in NER involves breaking down text into smaller units, such as words or phrases. These tokens serve as the building blocks for identifying entities within the text.
- Entity Detection: At this stage, the system scans the tokens to identify potential entities within the text.
- Entity Classification: Once detected, entities are categorized into predefined classes, such as “Person,” “Organization,” “Location,” and others.
- Contextual Analysis: Context plays a crucial role in enhancing recognition accuracy. Since words can have multiple meanings, analyzing the surrounding text helps determine the correct interpretation of an entity.
- Post-processing: The final step involves refining the results by resolving ambiguities, merging multi-word entities, and cross-validating detected entities with external knowledge sources or databases to ensure accuracy.

## Model
We will be using the [Davlan/xlm-roberta-large-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-large-ner-hrl) model from Hugging Face, a NER system designed for ten high-resource languages: Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese. Built on the XLM-RoBERTa large architecture, this model has been fine-tuned to identify three types of entities: Location (LOC), Organization (ORG), and Person (PER). It was trained on a diverse dataset aggregating entity-labeled text from these ten languages, enhancing its multilingual NER capabilities.

## Approach
The approach consists of three main steps: loading the pre-trained model, extracting names from raw text, and processing the dataset. First, the model is loaded from Hugging Face, along with its tokenizer, to perform Named Entity Recognition. The pipeline API from the transformers library is used with an aggregation strategy set to simple to group tokens into meaningful entity groups. This ensures that multi-word entities are correctly identified and classified. Next, for name extraction, the input text is checked for NaN values to handle missing data gracefully. The text is then split into potential names using a regular expression that identifies delimiters such as '/', '|', ',', and '&'. Each extracted token is passed through the NER model, and only those classified as Person (PER) are retained. The detected names are formatted and concatenated using '/' as a separator. Finally, the dataset processing stage involves reading the input CSV file into a Pandas DataFrame, applying the name extraction function to the raw_comp_writers_text column, and saving the results as a new CSV file with an additional column, PREDICTED_TEXT, containing the extracted names.

The code consists of four main components. The load_model() function loads the Davlan/xlm-roberta-large-ner-hrl model using AutoTokenizer and AutoModelForTokenClassification, initializing the NER pipeline. The extract_names() function cleans and tokenizes input text, applies the NER model to detect Person (PER) entities, and returns a formatted string of detected names. The process_dataset() function reads the dataset, applies the name extraction function to each row, and saves the results in a new CSV file. The main execution block defines paths for input and output CSV files, loads the NER model, and processes the dataset to generate the final output.

## Evaluation
Evaluate the performance of the model by comparing its predicted outputs against ground-truth data. The evaluation is based on three key metrics: Precision, Recall, and F1-Score. These metrics provide insights into the model’s ability to correctly identify relevant names while minimizing incorrect detections. The script processes an output CSV file containing both the predicted and actual names, computes the necessary statistics, and prints the results.

In order to evaluate the performance of the model, first I separate the names in both columns into sets using '/' as the delimiter. This allows me to directly compare the predicted and actual names by evaluating the following:
- True Positives (TP): Names that appear in both the predicted and ground-truth sets.
- False Positives (FP): Names that are present in the predicted set but do not exist in the ground-truth set (incorrect extractions).
- False Negatives (FN): Names that are in the ground-truth set but are missing from the predicted set (missed extractions).

Based on the above values, I calculate the key evaluation metrics:
- Precision: Measures the proportion of correctly predicted names out of all extracted names.
- Recall: Measures the proportion of actual names that were successfully identified.
- F1-Score: Provides a harmonic balance between precision and recall, ensuring a comprehensive assessment of the model’s performance.

# Results and Analysis
The results of the model are:
| Metric | Value |
|----------|----------|
| Precision    | 68.54%   |
| Recall    | 66.51%   |
| F1-Score    | 67.51%   |

Based on the above results we can come to the following conclusions:
- A precision of 68.54% means that 68.54% of the names predicted by the model are actually correct.
- A recall of 66.51% means that the model correctly identifies 66.51% of the actual names present in the dataset.
- Since f1-score is closer to both precision and recall, it confirms that the model has moderate performance but struggles to capture all actual names while also making some incorrect predictions.
- The model is not perfect but performs decently. It gets about two-thirds of the names correct. Also it is trained on 10 different languages (including Chinese), not only on English.

# Environment and Run Instructions
Instructions to create an environment:


```bash
python -m venv text\_norm\_env
source envirnmonet\_name/bin/activate
pip install -r requirements.txt
```

Instructions to run the code:
- Execute the data_analysis.py script to get information about the initial dataset.
- Execute the text_normalization.py script to generate output.csv in the data/ folder.
- Run evaluation.py to evaluate the model's performance.
