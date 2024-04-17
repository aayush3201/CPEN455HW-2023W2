'''
This file is to be used to test the model at models/conditional_pixelcnn.pth
and generate all the necessary files.
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
from classification_evaluation import get_label
import argparse
import os
from pytorch_fid.fid_score import calculate_fid_given_paths
from generation_evaluation import my_sample
import numpy as np
NUM_CLASSES = len(my_bidict)

if __name__ == '__main__':
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir='data', 
                                                            mode = 'test', 
                                                            transform=ds_transforms, include_paths=True), 
                                             batch_size=32, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3)
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    hugging_csv = 'id,label\n' 
    logits = None
    for batch_idx, item in enumerate(tqdm(dataloader)):
        model_input, categories, img_path = item
        model_input = model_input.to(device)
        answer, curr_logits = get_label(model, model_input, device, get_logits=True)
        if logits == None:
            logits = curr_logits
        else: logits = torch.cat((logits, curr_logits), 0)
        print(logits.shape)
        # CSV row
        for i in range(len(answer)):
            hugging_csv = hugging_csv + f"{img_path[i].split('/')[-1]},{int(answer[i])}\n"

    # Saving logits        
    with open('test_logits.npy', 'w') as f:
        np.save(f, logits)

    paths = [gen_data_dir, ref_data_dir]
    print("Begin sampling!")
    my_sample(model=model, gen_data_dir=gen_data_dir)
    fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
    hugging_csv = hugging_csv + f"fid,{fid_score}"

    f = open("hugging_face.csv", "w")
    f.write(hugging_csv)
    f.close()
        

