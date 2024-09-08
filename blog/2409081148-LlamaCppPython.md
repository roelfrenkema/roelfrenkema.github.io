
# Llama Cpp on Python

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q8124DNL)


[Return to Blog](/blog) - [Return to Index](/)

I have used a plethora on UI to access LLM models. But most of them are designed badly or frustrate working with them.
And as the saying goes, if you dont like it do it yourself. 

Now this is what I preach. Do it yourself. Being dependent on others and their wims and changes is always bad. Not like you dont have enough of your own.

So being the cli surfer I am I took to the waves and soon created some python code (while learning Python) that enabled me to access LLM's on the lowest level.

The great thing about this is not only the learning factor, but you are able to do whatever you want with the output and the presentation there of. 

This article that probably will have many updates, starts with the basic script I use. You need to change PATH_TO_YOUR_MODELS and probably the model you want to use.
Depending on your GPU you should alter n_gpu_layers and/or n_ctx as they decide how much of the GPU is used.

TODO: Seting up environment.

```python
#
# LLama CPP Python
#
# Local model call v.1.0.9 by Roelf Renkema
# 

# Updates:
# - Added Timestamps
# - Added Colors
# - set max tokens to 512 to save time with large scripts

from llama_cpp import Llama
from PIL import Image
from prompt_toolkit import PromptSession
from termcolor import colored
import datetime

llm = Llama(
      model_path="PATH_TO_YOUR_MODELS/gemma-2-27b-it-Q4_K_L.gguf",
  n_gpu_layers= 25,
  n_ctx= 4096,
  cache_8bit= False,
  cache_4bit= False,
  threads= 1,
  threads_batch= 1,
  n_batch= 512,
  no_mmap= False,
  mlock= True,
  no_mul_mat_q= False,
  tensor_split= '',
  compress_pos_emb= 1,
  rope_freq_base= 5000000,
  numa= True,
  no_offload_kqv= False,
  row_split= False,
  tensorcores= True,
  flash_attn= True,
  streaming_llm= True,
  attention_sink_size= 5,
  chat_format="chatml",
)

def get_input():
    session = PromptSession()
    print(colored("Enter your prompt (press LEFT-ALT+ENTER to finish):",'white',attrs=["bold"]))

    multiline_input = session.prompt("> ",multiline=True)

    print(colored("Processing. Please wait a moment!",'light_green',attrs=["bold"]))
    return multiline_input

def get_answer(prompt):
    output = llm(
    prompt, # Prompt
    max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
    echo=False # Echo the prompt back in the output
    ) 
    # Output the answer
    multiline_answer = output['choices'][0]['text']
    print(colored(multiline_answer,'light_cyan'))
    return multiline_answer
 
def time_stamp():
    current_time = datetime.datetime.now()
    print(colored(current_time.strftime("%H:%M:%S"),'black', 'on_yellow'))
  
if __name__ == "__main__": 
    while True:
        multiline_input = get_input()
        if multiline_input == 'exit':
            break

        time_stamp()    
        get_answer(multiline_input)
        time_stamp()
 
```

I hope this script is helpfull and can teach you some stuff. You can contact me on Discord @geennaam or Huggingface https://huggingface.co/roelfrenkema if you have any questions. 

With thanks to [⚡straico.com⚡](https://straico.com) for supporting my work. You can support me to by using [this affiliate link](https://platform.straico.com/signup?fpr=roelf14) when subscribing to ⚡Straico⚡

[Return to Blog](/blog) - [Return to Index](/)
