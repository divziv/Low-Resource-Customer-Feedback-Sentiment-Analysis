
import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer

from src.config import *



class SentimentDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=MAX_LENGTH
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.max_length = max_length


        if "text" in self.dataframe.columns:

            self.text_column = "text"


        elif "field" in self.dataframe.columns:

            self.text_column = "field"


        else:

            raise ValueError(
                "No text column found"
            )



    def __len__(self):

        return len(self.dataframe)




    def __getitem__(self, index):

        text = str(
            self.dataframe.loc[
                index,
                self.text_column
            ]
        )


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )


        item = {

            "input_ids":
            encoding["input_ids"].squeeze(0),


            "attention_mask":
            encoding["attention_mask"].squeeze(0),

        }



        if "sentiment" in self.dataframe.columns:


            label = str(
                self.dataframe.loc[
                    index,
                    "sentiment"
                ]
            ).strip().lower()



            item["labels"] = torch.tensor(
                LABEL2ID[label],
                dtype=torch.long
            )



        return item






def load_data():

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    valid_df = pd.read_csv(
        VALID_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    unlabelled_df = pd.read_csv(
        UNLABELLED_FILE
    )


    return (
        train_df,
        valid_df,
        test_df,
        unlabelled_df
    )





def get_tokenizer(model_name):

    return AutoTokenizer.from_pretrained(
        model_name
    )





def get_datasets(model_name):


    train_df, valid_df, test_df, unlabelled_df = load_data()


    tokenizer = get_tokenizer(
        model_name
    )



    train_dataset = SentimentDataset(
        train_df,
        tokenizer
    )


    valid_dataset = SentimentDataset(
        valid_df,
        tokenizer
    )


    test_dataset = SentimentDataset(
        test_df,
        tokenizer
    )


    unlabelled_dataset = SentimentDataset(
        unlabelled_df,
        tokenizer
    )



    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        unlabelled_dataset,
        tokenizer,
    )
