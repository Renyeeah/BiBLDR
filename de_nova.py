from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from Dataset.dataset import denova_drug_train, denova_drug_test, BasicDataset
import argparse, time, warnings

from model import DrugModel_denova
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

results_scores = []
results_labels = []

def getArgs():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='Gdataset', choices=['Cdataset', 'Gdataset', 'Ldataset', 'lrssl'])
    parser.add_argument('-seed', default=125, type=int)
    parser.add_argument('-seqlen_drug_behavior', type=int, default=8)

    # model
    parser.add_argument('-dropout', type=float, default=0.01)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-promote_embedding_dim', type=int, default=512)

    # train
    parser.add_argument('-epochs', default=2, type=int)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-lr', default=0.0001, type=float)
    parser.add_argument('-rating_T', type=float, default=2)

    # test
    parser.add_argument('-eval_margin', default=2, type=int)
    parser.add_argument('-T_score', default=0.50, type=float)

    return parser

def test(model, testLoader, args):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in testLoader:
        # for batch in testLoader:
            prediction = model(batch).squeeze(-1)
            results.append((batch[0], batch[3], prediction, batch[4]))

            results, labels, auc, aupr = evaluate_test(results, args, de_nova=True)
            results_scores.extend(results)
            results_labels.extend(labels)
            return auc, aupr

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = getArgs()
    args = parser.parse_args()
    print_args(args)
    set_random_seed(args.seed)

    if args.dataset == 'Gdataset':
        drug_num = 593
    else:
        raise ValueError("Please set parameter for your dataset")

    dataset = BasicDataset(args)
    similarity_matrix = [dataset.drug_sim, dataset.disease_sim]
    rel_loss = torch.nn.BCEWithLogitsLoss()

    for drug_id in range(0, drug_num):
        print('===== train drug ' + str(drug_id) + ' =====')
        trainset = denova_drug_train(drug_id, args)
        trainset.get_all_sequence()
        trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        testset = denova_drug_test(drug_id, args)
        testset.get_all_sequence()
        testLoader = DataLoader(testset, batch_size=args.batch_size)

        model = DrugModel_denova(dataset.Wrname_len, dataset.Wdname_len, similarity_matrix, args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs+1):
            model.train()
            running_loss = 0.0
            for batch in tqdm(trainLoader, desc="Train epoch " + str(epoch)):
                # for batch in trainLoader:
                prediction = model(batch).squeeze(-1)
                label = batch[4].to(device).type(torch.float)
                loss = rel_loss(prediction, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()

            if epoch % args.eval_margin == 0:
                auroc, aupr = test(model, testLoader, args)
                print(f'Test: epoch={epoch}, auroc={auroc}, aupr={aupr}')
                time.sleep(0.02)

    auc, aupr = evaluate(results_scores, results_labels)
    print(auc, aupr)
    save_experiment_results(args, auc, aupr)



