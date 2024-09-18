---                                                                                                                                                                             
layout: default                                                                                                                                                                 
title: Replicate easy training                                                                                                                                                        
---  
# Replicate FLUX1 training

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q8124DNL)


[⬅️](/github) | [🏠](/)

I have been working to automate my worflow in training lora with the replicate API. I now work with a singli directory of images that will be fully automated captioned and send to replicate.

I will not go into the fine detailes. Give you some room to learn and adapt. But I will illustrate the steps needed and some example scripts.

## Step 1. Image collection.

Lets face it most people swear by Google. But did you ever take a look at Bing. It is really amzing in the way you can select messages there, and the choice, size and quality of the images.

You dont need more then 10 to 20 images for FLUX, but make sure that they are divers and show your subject well. As usually pervent repetitions.

Once you collected then you might want to convert to png and resize them as Replicate dont like oversized trainings data. I use a small batch file file for that that I call with the directory of the files as an argument.

The file takes care of conversion to png and then the tedious work to down or upscale in a 1024 box. While doing this it takes care of the quality and size of the image and removes any metadata.

Ofcause next step is to translate this into python ROTFLMAO. and already done. This means we are now close to hooking things up into one file.

```python
from PIL import Image, ImageFilter, ImageEnhance
import os
import argparse

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
args = parser.parse_args()


def convert_to_png(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                new_filename = os.path.splitext(filename)[0] + '.png'
                new_file_path = os.path.join(directory, new_filename)
                img.save(new_file_path, 'PNG')
        except IOError:
            print(f"Cannot convert {file_path}")

def process_images(directory):
    max_size = 1024
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}")

            with Image.open(file_path) as img:
                # Remove metadata
                img = img.copy()

                # Unsharp masking
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                # Calculate new dimensions to maintain aspect ratio
                width, height = img.size
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)

                # Resize the image
                img = img.resize((new_width, new_height), Image.HAMMING)

                # Save the processed image with optimized settings and without metadata
                img.save(file_path, 'PNG', optimize=True)

if __name__ == "__main__":
    directory = args.name
    convert_to_png(directory)
    process_images(directory)
 
```

That finishes the first step.

## Step 2. Captioning.

I know there are plenty of programs helping to capion images. But did you know you dont need them? First of instead of keywords Flux is very happy with natural language. In the following Python script we produce that together with a trigger word.

The script uses a local model named **xtuner_llava-phi-3-mini-hf** You can download it on huggingface and change the line with YOURPATH in it to where you placed the model directory. Ofcause you can adapt the script to other models.

I call the script with two arguments, the first one the directory where your images are. The second one a token that you want to add to the description. It makes identifying your LoRa in your prompt easier.

```python
#
# works with transformers-4.44.2
# but latest github transformers will break it
#
import argparse
import torch
import os
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig, AutoConfig

# Define command line argument parser                                                                                                                                           
parser = argparse.ArgumentParser(description='get dir')                                                                                                                   
parser.add_argument('dir', type=str, help='dir name')                                                                              
parser.add_argument('tok', type=str, help='token name')                                                                              
args = parser.parse_args()                                                                                                                                                      
                                                                                                                                                                                
# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Define the model ID directory
model_id = "/YOURPATH/xtuner_llava-phi-3-mini-hf"

# Define the generation configuration
#generation_config = GenerationConfig(max_new_tokens=100)

# Create the pipeline
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config, "low_cpu_mem_usage": True})

# Define the prompt
prompt = "<|user|>\n<image>\nGive a description of the image.<|end|>\n<|assistant|>\n"

# Loop through all files in the directory
for filename in os.listdir(args.dir):

    # Check if the file is a PNG image
    if filename.endswith(".png"):
        # Open the image
        image = Image.open(f"{args.dir}/{filename}")
        print(f"Image: {filename}")

        # Generate text
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})

        # Get rid of empty lines
        generated_text = outputs[0]['generated_text'].replace("Give a description of the image. ", f"{args.tok}, ")
        lines = generated_text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        result = ' '.join(non_empty_lines)

        #print(f"{result}\n")

        # Save the result to a file
        with open(os.path.splitext(f"{args.dir}/{filename}")[0] + ".txt", "w") as f:
            f.write(result)

```

All straight forward as you can see. After running the script you image directory should have a .txt file for each image. With tags we needed to edit them and see if everything was fitting the image. With the natural language approach this is not realy needed.

## Step3. Zippem Up.

Now you can just zip up all to a zipfile with the name source.zip which should be in the image directory for the next step.

```
import os
import zipfile
import argparse
import shutil

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
args = parser.parse_args()

def zip_files(directory, output_filename):
    zip_file = zipfile.ZipFile(output_filename, 'w')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    
    zip_file.close()

# Use the function
directory_to_zip = args.name
output_filename = "source.zip"

zip_files(directory_to_zip, output_filename)

shutil.move("source.zip", args.name)

```

all ready for training now.

## Step 4. Training.

We are now all set to invoke training on Replicate. The folowing takes care of that with 3 arguments, the image directory, the token and a description. As stuff goes wrong sometimes  after model creation and breaks in the Replicate example because the model exsists I have build a check for that.

To run it you should make sure your Replicate api key is in the enveronment. And while writing I think you should get the Huggingface key from there to if you need it. So I will change the script later. 

Replace owner with your user name. Also if you use Hug do not forget to create a model there before starting to train.

```python
import replicate
import argparse
from replicate import models
from replicate.exceptions import ReplicateError

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name of the global variable for the model')
parser.add_argument('token', type=str, help='The name of the token')
parser.add_argument('description', type=str, help='The description')
args = parser.parse_args()

# Assign command line arguments to global variables
owner = "YOURNAME"

try:
    # Try to create a new model
    model = models.create(
        name=f"{args.name}",
        owner=f"{owner}",
        visibility="public",  # or "private" if you prefer
        hardware="gpu-t4",  # Replicate will override this for fine-tuned models
        description=f"{args.description}"
    )
    print(f"Model created: {args.name}")
except ReplicateError as e:
    if f'A model with that name and owner already exists.' in str(e):
        print("Model already exists, using it for training.")
    else:
        raise  # If this is not a "model already exists" error, re-raise the exception.
	
# Now use this model as the destination for your training

training = replicate.trainings.create(
    version="ostris/flux-dev-lora-trainer:7f53f82066bcdfb1c549245a624019c26ca6e3c8034235cd4826425b61e77bec",
    input={
        "input_images": open(f"{args.name}/source.zip", "rb"),
        "steps": 1000,
        "lora_rank": 16,
        "optimizer": "adamw8bit",
        "batch_size": 1,
        "resolution": "512,768,1024",
        "autocaption": False,
        "trigger_word": f"{args.token}",
        "learning_rate": 0.0004,
        "hf_token": "hf_XXXXXXXXXXXXXXXXXXX",  # optional
        "hf_repo_id": f"{owner}/{args.name}",  # optional
    },
    destination=f"{owner}/{args.name}"
)

print(f"Training started: {training.status}")
print(f"Training URL: https://replicate.com/p/{training.id}")
```

## About this approach.

As you can see the steps are easy to follow. I combine them all in a single batch file. So once I collected the images I call it and it runs. Simple. It could be improved by training yourself. At this moment my rtx3060 is not up to is so no use to me.

At some point the caption script will complain
```
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset                                                  
```
I need to dig into that and learn a little more. I daubt though it makes much sense if you only use a few images like her. You can safely ignore the comment though.

I hope this scripts are helpfull and can teach you some stuff. You can contact me on Discord @geennaam or Huggingface https://huggingface.co/roelfrenkema if you have any questions. 

With thanks to [⚡straico.com⚡](https://straico.com) for supporting my work. You can support me to by using [this affiliate link](https://platform.straico.com/signup?fpr=roelf14) when subscribing to ⚡Straico⚡


## The completed script.

I have completed the script now and it works fine for me. I am going to do some more work and create a github repo
for it. There are ofcause some install steps that need to be documented and I want to do some work on the
configuration of the script itself.

For now read the remarks carefully. You will have to make some changes. The owner name being one, and the
path to the model another. But if you have read the buildingpart the script should have no secrets left.

```python

# Trainplicate a script by Roelf Renkema.
#
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import torch
import argparse
import zipfile
import shutil
import replicate
import datetime
from replicate import models
from replicate.exceptions import ReplicateError
from PIL import Image, ImageFilter, ImageEnhance
from transformers import BitsAndBytesConfig, pipeline
from termcolor import colored

# This should reflect you Replicate, Github and Huggingface username
owner="YOU USER NAME HERE"

# Define command line argument parser
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('name', type=str, help='The name/path of the lora')
parser.add_argument('tok', type=str, help='token name')                                                                              
parser.add_argument('description', type=str, help='desciption of the lora')                                                                              
args = parser.parse_args()

# Step 1 convert all images to PNG and remove anything not png
def convert_to_png(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                new_filename = os.path.splitext(filename)[0] + '.png'
                new_file_path = os.path.join(directory, new_filename)
                img.save(new_file_path, 'PNG')
        except IOError:
            print(f"Cannot convert {file_path}")

    for filename in os.listdir(directory):
        # Create the full file path
        file_path = os.path.join(directory, filename)
    
        # Check if it is a file and does not end with .png
        if os.path.isfile(file_path) and not filename.endswith('.png'):
            # Delete the file
            os.remove(file_path)
            print(f'Deleted: {file_path}')


# Step 2 resize all images to 1024 box
# Routine will resize all images to a 1024 box keeping aspect ratio
# You can change "max_size" for other formats.
def process_images(directory):
    max_size = 1024
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}")

            with Image.open(file_path) as img:
                # Remove metadata
                img = img.copy()

                # Unsharp masking
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                # Calculate new dimensions to maintain aspect ratio
                width, height = img.size
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)

                # Resize the image
                img = img.resize((new_width, new_height), Image.HAMMING)

                # Save the processed image with optimized settings and without metadata
                img.save(file_path, 'PNG', optimize=True)


# Step 3 caption images
# Caption all png files in the directory
# You can change the model, but then you probably have to change the 
# prompt too.
# You can change the prompt, but you will then have to change the output
# replacement too. 
def create_captions(directory, token):

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Define the model ID
    model_id = "PATH TO YOUR MODEL HERE/xtuner_llava-phi-3-mini-hf"

    # Create the pipeline
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config, "low_cpu_mem_usage": True})

    # Define the prompt
    # You can change the prompt, but you will then have to change 
    # the output replacement too. 
    prompt = "<|user|>\n<image>\nGive a description of the image.<|end|>\n<|assistant|>\n"

    # Loop through all files in the directory
    for filename in os.listdir(directory):

        # Check if the file is a PNG image
        if filename.endswith(".png"):
            # Open the image
            image = Image.open(f"{directory}/{filename}")
            print(f"Image: {filename} being captioned.")

        # Generate text
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})

        # Get rid of empty lines and replace the prompt by our token.
        generated_text = outputs[0]['generated_text'].replace("Give a description of the image. ", f"{token}, ")
        lines = generated_text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        result = ' '.join(non_empty_lines)

        # Save the result to a file
        with open(os.path.splitext(f"{directory}/{filename}")[0] + ".txt", "w") as f:
            f.write(result)


# Step 4 zip images and captions(= *.txt)
# There should not be anything else in your directory         
def zip_files(directory, output_filename):
    zip_file = zipfile.ZipFile(output_filename, 'w')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))
    
    zip_file.close()


# Step 5 is a one liner in the main routine


# Step 6 create a model if one does not exist
# Beware if it exists it will use that for training
def create_model(directory,user,description):
    try:
        # Try to create a new model
        model = models.create(
            name=f"{directory}",
            owner=f"{user}",
            visibility="public",  # or "private" if you prefer
            hardware="gpu-t4",  # Replicate will override this for fine-tuned models
            description=f"{description}"
        )
        print(f"Model created: {directory}")
    except ReplicateError as e:
        if f'A model with that name and owner already exists.' in str(e):
            print("Model already exists, using it for training.")
        else:
            raise  # If this is not a "model already exists" error, re-raise the exception.

# Step 7 send it of to Replicate to develop
# Be sure your replicate token is set in your enveronment.
# If you use Huggingface a write key should be available in your environment as INFERENCE_WRITE.
def train_model(directory,user,token):
    hugkey=os.getenv("INFERENCE_WRITE")
    training = replicate.trainings.create(
        version="ostris/flux-dev-lora-trainer:7f53f82066bcdfb1c549245a624019c26ca6e3c8034235cd4826425b61e77bec",
        input={
            "input_images": open(f"{directory}/source.zip", "rb"),
            "steps": 1000,
            "lora_rank": 16,
            "optimizer": "adamw8bit",
            "batch_size": 1,
            "resolution": "512,768,1024",
            "autocaption": False,
            "trigger_word": f"{token}",
            "learning_rate": 0.0004,
            "hf_token": f"{hugkey}",  # optional
            "hf_repo_id": f"{user}/{directory}",  # optional
        },
    destination=f"{user}/{directory}"
    )

    print(f"Training started: {training.status}")
    print(f"Training URL: https://replicate.com/p/{training.id}")


# A little helper routine that prints a text with a timestamp
# Great to follow execution and time.
def time_stamp(text):
    current_time = datetime.datetime.now()
    time_text = current_time.strftime("%H:%M:%S")
    print(colored(f"{time_text} {text} ",'white', 'on_light_red'))
        
if __name__ == "__main__":
    
    # Step 1 convert all images to PNG
    time_stamp("Step 1 convert all images to PNG")
    convert_to_png(args.name)
    # Step 2 resize all images to 1024 box
    time_stamp("Step 2 resize all images to 1024 box")
    process_images(args.name)
    # Step 3 caption images
    time_stamp("Step 3 caption images")
    create_captions(args.name,args.tok)
    # Step 4 zip images and captions
    time_stamp("Step 4 zip images and captions")
    zip_files(args.name, "source.zip")
    # Step 5 move the zip into the lora dir
    time_stamp("Step 5 move the zip into the lora dir")
    shutil.move("source.zip", args.name)
    # Step 6 create a model if one does not exist
    time_stamp("Step 6 create a model if one does not exist")
    create_model(args.name,owner,args.description)
    # Step 7 send it of to Replicate to bake
    time_stamp("Step 7 send it of to Replicate to bake")
    train_model(args.name,owner,args.tok)
    time_stamp("And another beautifull LoRa is baking in the oven!")
```

Enjoy or wait for the whole script to appear on github with install manual.

[⬅️](/github) | [🏠](/)
