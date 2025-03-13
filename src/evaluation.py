import os
import pandas as pd

def evaluate(data_path: str, output_csv_name: str) -> None:
    '''
        A function that evaluates the performance of the model using precision, recall and f1-score.
        Compare the predicted text with the clean text based on the names they contain.

        Args:
            data_path: the path to the data
            output_csv_name: the name of the output csv

        Return:
            Prints the precision, recall and f1-score

    '''

    # Read the output csv.
    output_df = pd.read_csv(os.path.join(data_path, output_csv_name))

    # Compare the predicted text with the clean text and compute precision, recall and f1-score.
    true_positives, false_positives, false_negatives = 0, 0, 0
    for index, row in output_df.iterrows():

        # Handle NaN values: Convert NaN to empty string before splitting
        clean_text = row['CLEAN_TEXT'] if pd.notna(row['CLEAN_TEXT']) else ""
        predicted_text = row['PREDICTED_TEXT'] if pd.notna(row['PREDICTED_TEXT']) else ""

        # Split texts to sets with the names.
        clean_names_set = set(clean_text.split('/'))
        predicted_names_set = set(predicted_text.split('/'))

        # Calculate TP, FP and FN.
        true_positives += len(predicted_names_set & clean_names_set)  # Intersection: names in both sets.
        false_positives += len(predicted_names_set - clean_names_set)  # In predicted but not in clean.
        false_negatives += len(clean_names_set - predicted_names_set)  # In clean but not in predicted.

    # Compute precision, recall and f1-score.
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print the results.
    print(f'Precision: {round(precision*100, 2)}%')
    print(f'Recall: {round(recall*100, 2)}%')
    print(f'F1-Score: {round(f1_score*100, 2)}%')



if __name__ == "__main__":

    # The data path.
    data_path = '/home/nsintoris/Documents/Projects/Orfium/data' 
    output_csv_name = 'output.csv'

    # Evaluate the performance of the model.
    evaluate(data_path, output_csv_name)