from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir='data', 
                                                            mode = 'test', 
                                                            transform=ds_transforms, include_paths=True), 
                                             batch_size=32, 
                                             shuffle=True, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=40, nr_logistic_mix=5, input_channels=3)
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    hugging_csv = '' 
    for batch_idx, item in enumerate(tqdm(dataloader)):
        model_input, categories, img_path = item
        model_input = model_input.to(device)
        answer = get_label(model, model_input, device)
        for i in range(len(answer)):
            hugging_csv = hugging_csv + f"{img_path[i]}, {answer[i]}\n"
    hugging_csv = hugging_csv + 'fid, 455'
    
    f = open("hugging_face.csv", "w")
    f.write(hugging_csv)
    f.close()
        

