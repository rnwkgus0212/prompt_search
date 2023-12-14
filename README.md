



# discrete prompt search 


## Overview

- iterative hard prompt tuning (HPT) with task-specific part-of-speech(POS) tag heuristics. 
- experiments on SST-2, Agnews dataset.
- project is built on OpenPrompt (https://github.com/thunlp/OpenPrompt) 


## Installation

**Note: Please use Python 3.8+**

### Using Git

Clone the repository from github:

```shell
pip install -r requirements.txt
python setup.py install
```

Modify the code

```
python setup.py develop
```



## Run

### SST-2

Run run_seeded_sc.py for SST-2 sentiment classification

### Agnews

Run run_seeded_tc.py for Agnews topic classification

## Experiment Settings
### Configs
```python
iteration = 5                  # number of discrete tokens to find
count_from_highest = 16        # number of initial prompt candidates
shots = 32                     # number of shots 
proportional_decrease=0.5      # ratio of decreasing number of candidates after an iteration
proportional_increase=2        # ratio of increasing number of shots after an iteration

```

```python
allowed_pos = 'N'              # task-specific
model_config = "roberta-large"
```

### Templates and Verbalizers (default)
```python
# SST-2
classes = [ 
    "negative",
    "positive"
]

label_words = {
        "negative": ["terrible"],
        "positive": ["great"],
    }

template = '{"placeholder": "text_a"} ' + the_prompt_to_be_found  + ' {"mask"}.'
```

```python
# Agnews
classes = [ 
    "World",
    "Sports",
    "Business",
    "Tech"
]

label_words = {
        "World": ["World"],
        "Sports": ["Sports"],
        "Business": ["Business"],
        "Tech": ["Tech"]
    }

template = ' {"mask"} ' +  the_prompt_to_be_found  + ' {"placeholder": "text_a"} '
```


## Records
| Task | Methodology | Prompt  | Test Acc | Test # |   Template | Verbalizer | Model | 
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|:--------:|
| SST-2 | **Manual Prompt** | It was very | **82.02** | 1.8k  | [Input] [Prompt] [MASK]|  {"negative": ["terrible"], "positive": ["great"], } |  roberta-large |
| SST-2 | **rlprompt** | Absolutely VERY absolute VERY absolute | **93.25** | 1.8k  | [Input] [Prompt] [MASK]|  {"negative": ["terrible"], "positive": ["great"], } |  roberta-large |
| SST-2 | **lottery prompt** | I find very | **86.99** | 1.8k  | [Input] [Prompt] [MASK]|  {"negative": ["terrible"], "positive": ["great"], } |  roberta-large |
| SST-2 | **Ours** | ĠgenuinelyĠunequivocallyĠstrongly | **93.19** | 1.8k  | [Input] [Prompt] [MASK]|  {"negative": ["terrible"], "positive": ["great"], } |  roberta-large |

| Task | Methodology | Prompt  | Test Acc | Test # |   Template | Verbalizer | Model | 
|:------:|:------------:|:------------:|:-------:|:---------:|:-----------:|:--------:|:--------:|
| AGNews | **Manual Prompt** | News:  | **76.90** | 7.6k  | [MASK] [Prompt] [Input] |         "World": ["World"], "Sports": ["Sports"],"Business": ["Business"],"Tech": ["Tech"]|  roberta-large |
| AGNews | **rlprompt** | Reviewer Stories | **81.02** | 7.6k  | [MASK] [Prompt] [Input] |       "World": ["World"], "Sports": ["Sports"],"Business": ["Business"],"Tech": ["Tech"] |  roberta-large |
| AGNews | **lottery prompt** | you think other | **77.32** | 7.6k  | [MASK] [Prompt] [Input] |       "World": ["World"], "Sports": ["Sports"],"Business": ["Business"],"Tech": ["Tech"] |  roberta-large |
| AGNews | **Ours** | ĠexplainologyĠDevilCLOSEĠtrick  | **79.54** | 7.6k  | [MASK] [Prompt] [Input] |       "World": ["World"], "Sports": ["Sports"],"Business": ["Business"],"Tech": ["Tech"] |  roberta-large |










<!-- 

1821 (1.8k sst-2 dataset)
Hard Prompt Tuning Baseline
It was very 0.8396
Best single token ?
0.916
RLprompt
Absolutely VERY absolute VERY absolute
0.9325
Lottery Prompt
I find very
0.8699
I find really
0.8402
Ours
ĠgenuinelyĠunequivocallyĠstrongly
0.9319 -->