import torch

from torch import optim
from torch.utils.data import DataLoader
from Dataset.dataset import *
from model import stage1_encoder
from loss import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_promotes(args, drug_or_disease):

    now_loss = stage1_loss(args)

    if drug_or_disease == 'drug':
        dataset = stage1_drug(args)
        encoder = stage1_encoder(dataset.Wrname_len, args).to(device)
    elif drug_or_disease == 'disease':
        dataset = stage1_disease(args)
        encoder = stage1_encoder(dataset.Wdname_len, args).to(device)
    else:
        raise ValueError('Invalid drug_or_disease')

    dataloader = DataLoader(dataset, batch_size=args.stage1_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.stage1_epochs):
        encoder.train()
        running_loss = 0.0
        for batch in dataloader:
            feature_drug_1 = batch[0].to(device).type(torch.float)
            feature_drug_2 = batch[1].to(device).type(torch.float)
            label = batch[2].to(device)

            promote_drug_1, promote_drug_2 = encoder(feature_drug_1, feature_drug_2)
            loss = now_loss(promote_drug_1, promote_drug_2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f'epoch={epoch+1}, loss={running_loss}')

        scheduler.step()

    if drug_or_disease == 'drug':
        features = torch.tensor([dataset.drug_sim[i] for i in range(dataset.Wrname_len)])
    elif drug_or_disease == 'disease':
        features = torch.tensor([dataset.disease_sim[i] for i in range(dataset.Wdname_len)])
    else:
        raise ValueError('Invalid drug_or_disease')

    encoder.eval()
    with torch.no_grad():
        features = features.to(device).type(torch.float)
        promotes, _ = encoder(features, features)

    return promotes










