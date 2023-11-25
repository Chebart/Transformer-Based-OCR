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
encoder = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').train()
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = Transformer(encoder).train().to(device)
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
BATCH_SIZE = 16
pr_dataset = DataLoader(pr_dataset,batch_size=BATCH_SIZE)

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
    mean_batch_loss = 0
    for batch in pr_dataset:
        optimizer.zero_grad()
        id, res = model(batch["pixel_values"].to(device),batch["text"].to(device))

        B, T, C = res.shape
        res = res.view(B*T, C)
        targets = batch["text"].view(B*T).to(device)

        loss = criterion(res,targets)
        loss.backward()
        optimizer.step()
        mean_batch_loss += loss
        progress_bar.update(1)
        torch.cuda.empty_cache()

    mean_batch_loss /= len(pr_dataset)

    f = open("train_log.txt","w")
    f.write(f"epoch{epoch+1}: {mean_batch_loss}\n")
    scheduler.step()
    torch.save(model.state_dict(), f"./model_weights/model_{epoch+1}.pth")
    encoder.save_pretrained(f"./encoder_weights/VIT_{epoch+1}.pth")
