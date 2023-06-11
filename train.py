from transformers import ViTImageProcessor, AutoTokenizer, ViTForImageClassification
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from torchvision.io import read_image
import torch
from our_model import Transformer
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

#initialize the random number generator.
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.benchmark = True
cudnn.deterministic = True

# initialize gpu, tokinizer,
# encoder and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = ViTForImageClassification.from_pretrained('VIT_130.pth').train()
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = Transformer(encoder).train().to(device)
model.load_state_dict(torch.load("./our_model/model_130.pth"))
tokenizer = AutoTokenizer.from_pretrained("own-tokenizer")

# load train dataset
dataset = load_dataset("csv", data_files="train.csv")
dataset = pd.DataFrame({"text":dataset["train"]["text"],"label":dataset["train"]["label"],
                     "name":dataset["train"]["name"]}).dropna()
dataset = Dataset.from_pandas(dataset)

#Function that takes an image,
#reads it, extracts the feature representation and
#and writes it back as a tensor.
def transform(example_batch):
    inputs = feature_extractor([read_image("./imgs/"+x)[:3,:,:] for x in example_batch['name']], return_tensors='pt')
    # Include the labels
    inputs['labels'] = example_batch['label']
    inputs['text'] = tokenizer(example_batch["text"], padding="max_length", truncation=True, return_tensors='pt',max_length=197).input_ids
    return inputs

# transform dataset
pr_dataset = dataset.with_transform(transform)
pr_dataset = DataLoader(pr_dataset,batch_size=16)

# optimizer
optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-9)
# train hyperparameters
num_epochs = 200
num_training_steps = num_epochs * len(pr_dataset)
# loss function and scheduler
criterion = nn.CrossEntropyLoss(ignore_index=0)
scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

# train loop
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for batch in pr_dataset:
        optimizer.zero_grad()
        id, res = model(batch["pixel_values"].to(device),batch["text"].to(device))

        B, T, C = res.shape
        res = res.view(B*T, C)
        targets = batch["text"].view(B*T).to(device)

        loss = criterion(res,targets)
        loss.backward()
        optimizer.step()
        progress_bar.write(str(loss.item()))
        progress_bar.update(1)
        torch.cuda.empty_cache()

    f = open("train_log.txt","w")
    f.write("epoch{}\n".format(epoch+1))
    scheduler.step()
    torch.save(model.state_dict(), "./model_{}.pth".format(epoch+1))
    encoder.save_pretrained("./VIT_{}.pth".format(epoch+1))
