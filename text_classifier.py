import pandas as pd
import re
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer

def text_cleaner(text):
    # Step 1: Remove empty lines
    lines = [line for line in text.splitlines() if line.strip()]

    # Step 2: Remove lines starting with "from" or "to"
    lines = [line for line in lines if not line.lower().strip().startswith(('from', 'to'))]

    # Step 3: Remove URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F])|#)+')
    lines = [re.sub(url_pattern, '', line) for line in lines]

    # Step 4: Remove emails
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    lines = [re.sub(email_pattern, '', line) for line in lines]

    # Step 5: Remove phone numbers
    phone_pattern = re.compile(r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
    lines = [re.sub(phone_pattern, '', line) for line in lines]

    # Step 6: Remove zip codes (assuming US zip codes for simplicity)
    zip_pattern = re.compile(r'\b\d{5}(?:[-\s]?\d{4})?\b')
    lines = [re.sub(zip_pattern, '', line) for line in lines]

    # Step 7: Remove 'nan'
    lines = [line.replace('nan', '') for line in lines]

    lines = [line.replace('NaN', '') for line in lines]
    
    # Join cleaned lines into a single string
    cleaned_text = '\n'.join(lines)
    

    return cleaned_text

data_file = "PATH_TO_YOUR_DATAFILE"

device = 0 if torch.cuda.is_available() else -1
tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", torch_dtype=torch.bfloat16).to(device)

model_name = "facebook/bart-large-cnn"
tokenizer_bart = BartTokenizer.from_pretrained(model_name)
model_bart = BartForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

c = pd.read_csv(data_file, dtype={'DESCRIPTION' : str , 'NOTES' : str})
c = c.sample(frac=1).reset_index(drop=True)

count = 0
texts = []
clean_texts = []
summaries = []
for index, row in tqdm(c.iterrows(), total = 1000, unit = ''):
    if count == 1000:
        break
    text = str(row['DESCRIPTION']) + " \n " + str(row['NOTES'])
    if text != '': 
        count += 1
        texts.append(text)
        clean_text = text_cleaner(text)
        clean_texts.append(clean_text)

        inputs = tokenizer_bart([clean_text], max_length=1024, return_tensors="pt", truncation=True).to("cuda")
        summary_ids = model_bart.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
        
        summaries.append(summary)

# print(len(texts), len(clean_texts), len(summaries))

labels = []
for summary in tqdm(summaries, total = 1000, unit=''):
    prompt_v2 = f"""
            You are given summaries from cases related to single-family residential properties. Your task is to classify the text based on the presence of any illegal activities, squatter or trespasser activities, property damage, or stolen appliances in vacant homes. 
            
            Classify the text as '1' if it indicates any of the following:
            - Illegal entry (e.g., break-in, forced entry, unauthorized entry, illegal access)
            - Squatter or trespasser activity (e.g., squatter, trespasser, unauthorized person, someone living in)
            - Property damage (e.g., vandalism, broken, damaged, destroyed)
            - Stolen appliances (e.g., theft, stolen, missing appliances, robbed)
            
            Otherwise, classify the text as '0'.
            
            Here are some examples:
            
            Example 1:
            Description: "The front door was broken and several appliances are missing from the kitchen."
            Classification: 1
            
            Example 2:
            Description: "There is a person who appears to be living in the vacant house next door."
            Classification: 1
            
            Example 3:
            Description: "The tenant has requested a plumbing repair in the bathroom."
            Classification: 0
            
            Example 4:
            Description: "Neighbors reported seeing someone climbing through the window of the empty house."
            Classification: 1
            
            Example 5:
            Description: "The garden needs maintenance and the grass is overgrown."
            Classification: 0
            
            Please classify the following text:
            Description: {summary}
            Classification: 
            """
    input_ids = tokenizer_t5(prompt_v2, return_tensors="pt").input_ids.to("cuda")
    outputs = model_t5.generate(input_ids)
    result = int(re.findall(r'\d+', tokenizer_t5.decode(outputs[0]).strip())[0])
    labels.append(result)


# print(len(labels))
    
cases_labels = pd.DataFrame()

cases_labels['Original text'] = texts
cases_labels['Cleaned Text'] = clean_texts
cases_labels['Summarry'] = summaries
cases_labels['label'] = labels

cases_labels.head(10)

cases_labels.to_excel('Predictions.xlsx')