from openprompt.data_utils.text_classification_dataset import SST2Processor, AgnewsProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader

from tqdm import tqdm
import sys

import torch
import time
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

import os, sys

import random

# Set a random seed
seed = 42
random.seed(seed)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



wordnet.synsets

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# experiment configs!


dataset_name = "sst-2"
split = "test"

model = "roberta"
model_config = "roberta-large"

iteration = 5

inference_count = 0

count_from_highest = 16
# shots = 200
# shots = 64
shots = 32

proportional_decrease=0.5
proportional_increase=2

project = f"project_{count_from_highest}_{shots}_{seed}"

if_testing_small_run = None # should be changed to None
if if_testing_small_run:
    project += '_testrun'







print(dataset_name)
print(split)

dataset = {}
# dataset['train'] = SST2Processor().get_train_examples("./datasets/TextClassification/SST-2")
# dataset['validation'] = SST2Processor().get_dev_examples("./datasets/TextClassification/SST-2")


print('load model...')
plm, tokenizer, model_config, WrapperClass = load_plm(model, model_config)
print("model loaded")

savepath = f'./results/{dataset_name}_{project}.txt'

with open(savepath, 'a') as file:
    file.writelines("\n")






initial_targets = list(tokenizer.get_vocab().keys())


# get specific some info from dataset

full_data_len = 0

if dataset_name == "sst-2":
    temp = SST2Processor().get_test_examples("./datasets/TextClassification/SST-2")
    full_data_len = len(temp)
    allowed_pos = 'R'







filtered_targets = []
for unprocessed_word in initial_targets:
    if '<' in unprocessed_word and '>' in unprocessed_word:
        filtered_targets.append(unprocessed_word)
    else:
        try:
            processed_word = unprocessed_word.replace("Ġ",'') 
            w = word_tokenize(processed_word)
            single_tag = pos_tag(w)[0][1]
            if single_tag.startswith(allowed_pos):
                if wordnet.synsets(processed_word):
                    filtered_targets.append(unprocessed_word)
        except:
            print("\n")
            print(unprocessed_word)
            print(processed_word)
            
            
print(len(initial_targets))
print(len(filtered_targets))

classes = [ 
    "negative",
    "positive"
]

label_words = {
        "negative": ["terrible"],
        "positive": ["great"],
    }

dataset['test'] = SST2Processor().get_test_examples("./datasets/TextClassification/SST-2")
dataset['test'] = random.sample(dataset['test'], full_data_len)

def get_score(dataset_name,dataset,shots,candidate):


    if dataset_name == "sst-2":

        
        
        if shots:
            temp_dataset = dataset['test'][:shots]
        
        


        
        the_prompt_to_be_found = candidate
        
        template = '{"placeholder": "text_a"} ' + the_prompt_to_be_found  + ' {"mask"}.'






    promptTemplate = ManualTemplate(
        text = template,
        tokenizer = tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = label_words,
        tokenizer = tokenizer,
    )
    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    )
    promptModel=  promptModel.cuda()
    data_loader = PromptDataLoader(
        dataset = temp_dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )
    promptModel.eval()
    allpreds = []
    alllabels = []
    
    # scorer is evaluation

    for step, inputs in tqdm(enumerate(data_loader)):

        inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())


    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)

    result = f"{the_prompt_to_be_found.strip().ljust(150)}{str(round(acc,4)).ljust(10)}{str(shots).ljust(10)}\n"

    with open(savepath, 'a') as file:
        file.writelines(result)
    return acc




# for test purpose
if if_testing_small_run:
    accumulated_targets = filtered_targets[:if_testing_small_run]
else:
    accumulated_targets = filtered_targets


for i in range(0,1):
    new_targets = []
    results = {} 
    for candidate_prompt in accumulated_targets:
        
        score = get_score(dataset_name,dataset,shots,candidate_prompt) 
        inference_count+=1
        
        results[candidate_prompt] = score
        
    results_sorted = dict(sorted(results.items(), key=lambda item: item[1],reverse=True))
    print(results_sorted)
    results_list = list(results_sorted.keys())[:count_from_highest-1]
    results_list.append(' ') # empty
    
    best_words = results_list
    accumulated_targets = results_list

print(f"initial_result :  {results_list}")

for i in range(1,iteration):

    if proportional_decrease:
        count_from_highest = max(int(count_from_highest * proportional_decrease),5)
        print("\n\nto find")
        print(count_from_highest)    
    if proportional_increase:
        shots = min(int(shots*proportional_increase), full_data_len)
        
        # test all for last iteration. we don't use this to determine and update next step, but just to evaluate results
        if i == iteration-1: 
            shots = full_data_len
        
        print("\n\nshots")
        print(shots)    
    
    temp = []
    for i in accumulated_targets:
        for j in best_words:
            temp.append(i+j)
    accumulated_targets = temp
    
    print("\n\ncandidates")
    print(len(accumulated_targets))
    
    results = {} 
    for candidate_prompt in accumulated_targets:
        
        score = get_score(dataset_name,dataset,shots,candidate_prompt)
        inference_count+=1
        
        results[candidate_prompt] = score
        
    results_sorted = dict(sorted(results.items(), key=lambda item: item[1],reverse=True))
    results_list = list(results_sorted.keys())[:count_from_highest]
    accumulated_targets = results_list