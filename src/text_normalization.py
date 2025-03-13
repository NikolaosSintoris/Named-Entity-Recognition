import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def load_model() -> pipeline:
    '''
        Load the model from Hugging Face.

        Returns:
            ner: the pre-trained ner model.
    '''
    # Load the multilingual NER model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
    model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner

def extract_names(raw_text: str, ner: pipeline) -> str:
    '''
        From a raw text use ner model to extract the names of persons.

        Args:
            raw_text: the raw text
            ner: the hugging face model

        Return:
            The detected names seperated with the character '/'.

    '''
    # If the raw text is nan return an empty string.
    if not raw_text or pd.isna(raw_text):
        return ""
    
    # Split the string to get the names.
    words_lst = re.split(r"[/\|,&]", raw_text)

    # For each word in the raw text, keep only the words that the model classifies as persons.
    valid_names_lst = []
    for word in words_lst:
        # Remove any whitespaces in the beginning or in the end.
        word = word.strip()

        # Run NER model on the text.
        entities = ner(word)

        # Keep only the person names.
        for ent in entities:
            if ent["entity_group"] == "PER":
                valid_names_lst.append(ent["word"])

    return '/'.join(valid_names_lst)

def process_dataset(data_path: str, input_csv_name: str, ner: pipeline, output_csv_name: str) -> None:
    '''
        A function that process the datset using the model.

        Args:
            csv_path: the path to the csv
            ner: the ner model

        Return:
            Creates a new csv as the initial one, but with a new column that contains the predicted names of the model.
    '''
    # Read the dataset.
    df = pd.read_csv(os.path.join(data_path, input_csv_name))

    # Apply the function to extract the predicted names of the model to each row of the dataset.
    df["PREDICTED_TEXT"] = df["raw_comp_writers_text"].apply(lambda text: extract_names(text, ner))

    # Save the results to a new csv.
    df.to_csv(os.path.join(data_path, output_csv_name), index=False)
    

if __name__ == "__main__":

    # The data path. Fix path.
    data_path = '/home/nsintoris/Documents/Projects/Orfium/data'
    input_csv_name = 'normalization_assesment_dataset_10k.csv'
    output_csv_name = 'output.csv'

    # Load the hugging face model.
    ner = load_model()

    # Process the dataset with the model.
    process_dataset(data_path, input_csv_name, ner, output_csv_name)