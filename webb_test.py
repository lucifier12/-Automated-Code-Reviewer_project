import streamlit as st
import json
from streamlit_lottie import st_lottie
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"   #reduce the amount of console output from TF
import tensorflow as tf
from transformers import *
from datasets import load_dataset
import logging
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
     AdamW,
     T5ForConditionalGeneration,
     T5TokenizerFast as T5Tokenizer
 )
pl.seed_everything(42)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_pumpkin = load_lottiefile("hmm.json")


# PCG By T5-Base


class CodeDataset(Dataset):
  def __init__(
        self,
        data:pd.DataFrame,
        tokenizer:T5Tokenizer,
        text_max_token_len: int = 100,
        code_max_token_len: int = 128
    ):
      self.tokenizer = tokenizer
      self.data = data
      self.text_max_token_len = text_max_token_len
      self.code_max_token_len = code_max_token_len
  def __len__(self):
      return len(self.data)
  def __getitem__(self, index : int):
      data_row = self.data.iloc[index]
      text = data_row["Instruction"]
      text_encoding = tokenizer(
              text,
              max_length = self.text_max_token_len,
              padding = "max_length",
              truncation=True,
              return_attention_mask=True,
              add_special_tokens=True,
              return_tensors="pt"
          )
      code_encoding = tokenizer(
              data_row["Output"],
              max_length = self.code_max_token_len,
              padding = "max_length",
              truncation=True,
              return_attention_mask=True,
              add_special_tokens=True,
              return_tensors="pt"
          )
      labels = code_encoding["input_ids"]
      labels[labels ==0] = -100
      return dict(
              text = text,
              code = data_row["Output"],
              text_input_ids=text_encoding["input_ids"].flatten(),
              text_attention_mask=text_encoding["attention_mask"].flatten(),
              labels=labels.flatten(),
              labels_attention_mask=code_encoding["attention_mask"].flatten()
          )
  

class CodeDataModule(pl.LightningDataModule):
    def __init__(
          self,
          train_df: pd.DataFrame,
          test_df: pd.DataFrame,
          tokenizer: T5Tokenizer,
          batch_size: int = 8,
          text_max_token_len: int = 100,
          code_max_token_len: int = 128
      ):
      super().__init__()
      self.train_df = train_df
      self.test_df = test_df
      self.batch_size = batch_size
      self.tokenizer = tokenizer
      self.text_max_token_len = text_max_token_len
      self.code_max_token_len = code_max_token_len

    def setup(self,stage=None):
        self.train_dataset = CodeDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.code_max_token_len
        )
        self.test_dataset = CodeDataset(
                self.test_df,
                self.tokenizer,
                self.text_max_token_len,
                self.code_max_token_len
            )
    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size = self.batch_size,
          shuffle = True,
          num_workers = 2
        )
    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle = False,
          num_workers = 2
        )
    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size = self.batch_size,
          shuffle = False,
          num_workers = 2
        )
        
MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class TextCodeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
              input_ids,
              attention_mask = attention_mask,
              labels = labels,
              decoder_attention_mask = decoder_attention_mask
            )
        return output.loss, output.logits
    def training_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
              input_ids = input_ids,
              attention_mask = attention_mask,
              decoder_attention_mask=labels_attention_mask,
              labels= labels
              )
        self.log("train_loss",loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
              input_ids = input_ids,
              attention_mask = attention_mask,
              decoder_attention_mask=labels_attention_mask,
              labels= labels
              )
        self.log("val_loss",loss, prog_bar=True, logger=True)
        return loss
    def test_step(self,batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        loss, outputs = self(
              input_ids = input_ids,
              attention_mask = attention_mask,
              decoder_attention_mask=labels_attention_mask,
              labels= labels
              )
        self.log("test_loss",loss, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
    

trained_model = TextCodeModel.load_from_checkpoint(r"C:/Users/AVNEET/Desktop/Code/best-checkpoint_first.ckpt")

def text_to_code(text):
  text_encoding = tokenizer(
    text,
    max_length=100,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
    )

  generated_ids = trained_model.model.generate(
    input_ids=text_encoding["input_ids"],
    attention_mask=text_encoding["attention_mask"],
    max_length= 100,
    num_beams = 2,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
  )
  preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        for gen_id in generated_ids]
  return "".join(preds)


# Page Title
st.title("Project on Python Code Generation")
st.title(" ")


# Sidebar
st.sidebar.subheader("Navigation")
selected_page = st.sidebar.radio("Select a Page", ("Home", "Made By"))

# Home Page
if selected_page == "Home":
    st.header("Welcome to the Python Code generation Project")
    st.header(" ")
    st.write("Here is the real thing: ")
    # Page Title
    st.title("PCG_1")

    user_input = st.text_input("Enter the Query :", "")

    if st.button("Generate"):
        # Ask the user for a string input
        
        # Display the user's input
        m = text_to_code(user_input)

        st.write(f"Pyhton code for the query : {m}")

        # st.write(f"Pyhton code for the query ")

if selected_page == "Made By":
    st.title(" ")
    st.title("Avneet")
    
    st.title(" ")
    st.write("Made as Final Project for _Summer Internship_(2023).")
    st.write("Source Code of this project : https://drive.google.com/file/d/1pSKFhBsxTR_RnIOpZGIxkVfaZL2a6B1y/view?usp=drive_link")
    st.write("You can contact me at avneetsingh1297@gmail.com.")
    
    st_lottie(
        lottie_pumpkin,
        speed=1,
        reverse=False,
        loop=True,
        quality="high"
    )

# Footer
st.text("© 2024 Python code generator -by Avneet")