'''
    Script for single prediction on an image. It puts result in the folder.
'''

import argparse
import os
import random
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from model import Net
from utils import ConfigS, ConfigL, download_weights

parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='model_l.pt',
    help='Checkpoint name'
)

parser.add_argument(
    '-S', 
    '--size',
    type=str,
    default='L',
    help='Model size [S, L]',
    choices=['S', 'L', 's', 'l']
)

parser.add_argument(
    '-I',
    '--img-path',
    type=str,
    default='',
    help='Path to the image'
)

parser.add_argument(
    '-R',
    '--res-path',
    type=str,
    default='',
    help='Path to the results folder'
)

parser.add_argument(
    '-T', 
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

args = parser.parse_args()

config = ConfigL() if args.size.upper() == 'L' else ConfigS()

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

if __name__ == '__main__':
    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    assert os.path.isfile(args.img_path), 'Image does not exist'
    
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    
    img = Image.open(args.img_path)

    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len,
        device=device
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path, args.size)
        
    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():
        caption, _ = model(img, args.temperature)
    import cv2
    import numpy as np

# Load the image
    image = cv2.imread(args.img_path)

# Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to segment the image into text and non-text regions
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)[1]

# Find the contours of the text regions
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a mask to store the text regions
    mask = np.zeros(gray.shape, np.uint8)

# Fill the text regions in the mask with white pixels
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)

# Count the number of white pixels in the mask
    text_pixels = cv2.countNonZero(mask)
    height, width, channels = image.shape

# Calculate the total number of pixels
    total_pixels = height * width
    textdensity=text_pixels/total_pixels*100

    print("Number of text pixels:", text_pixels)
    print("Total Pixel",total_pixels)
    print("Text density",textdensity)
    
    fl = False
    t = ''
    for c in caption:
        if fl == False and c == '"':
            fl = True
        elif fl == True and c == '"':
            fl = False
        elif fl == False:
            t += c
    caption1=caption
    caption=t
    result=[]
    if(textdensity>0):
        image = cv2.imread(args.img_path)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(args.img_path,paragraph="False")
        k=1
        textocr=""
        for res in result:
            if(k==1):
                textocr=res[1]
                k+=1
            else:
                textocr+=","+res[1]
    else:
        textocr=""
    tokenizer = AutoTokenizer.from_pretrained("RoshanAdhithya/bart-final-image-captioning")

    model = AutoModelForSeq2SeqLM.from_pretrained("RoshanAdhithya/bart-final-image-captioning")
#model = T5ForConditionalGeneration.from_pretrained('./pytorch_model.bin')
#tokenizer = T5Tokenizer.from_pretrained('./tokenizer.json')
    device = torch.device('cpu')

    text =caption+textocr
    print("Input for BART",text)


#preprocess_text = text.strip().replace("\n","")
#t5_prepared_Text = "summarize: "+preprocess_text
#print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)


# summmarize 
    summary_ids = model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=4,
                                    min_length=15,
                                    max_length=100,
                                    early_stopping=False)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print ("\n\nSummarized text: \n",output)
    print(result)
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')

    img_save_path = f'{os.path.split(args.img_path)[-1].split(".")[0]}-R{args.size.upper()}.jpg'
    plt.savefig(os.path.join(args.res_path, img_save_path), bbox_inches='tight')

    plt.clf()
    plt.close()

    print('Generated Caption: "{}"'.format(caption1))
