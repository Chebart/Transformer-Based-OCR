from transformers import ViTImageProcessor, AutoTokenizer,  ViTForImageClassification
from datasets import load_dataset
from torchvision.io import read_image
from our_model import Transformer
import torch
import torch.backends.cudnn as cudnn

#initialize the random number generator.
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.benchmark = True
cudnn.deterministic = True

# initialize gpu, tokinizer,
# encoder and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
encoder = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').eval()
tokenizer = AutoTokenizer.from_pretrained("own-tokenizer")
tr = Transformer(encoder).to(device)
tr.load_state_dict(torch.load("./our_model/model.pth"))
tr.eval()

#Function that takes an image,
#reads it, extracts the feature representation and
#and writes it back as a tensor.
def transform(example_batch):
    inputs = feature_extractor([read_image("./imgs/"+x)[:3,:,:] for x in example_batch['name']], return_tensors='pt')
    return inputs

# load and transform train dataset
dataset = load_dataset("csv", data_files="test.csv")
pr_dataset = dataset["train"].with_transform(transform)

# read test images
# and generate sentences
with torch.no_grad():
    for i in range(len(pr_dataset)):
        res = tr.generate(pr_dataset[i]["pixel_values"].unsqueeze(0).to(device))
        print(dataset["train"][i]["name"])
        print(tokenizer.decode(res[0],skip_special_tokens=True))
    
