from torch import optim
from Dataset.dataset import *
import argparse, time, warnings
from torch.utils.data import DataLoader
from model import DrugModel
from utils import *
from torch.utils.tensorboard import SummaryWriter
from Stage1.stage1_train import train_promotes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
results_auroc = []
results_aupr = []

def getArgs():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-dataset', type=str, default='lrssl', choices=['Cdataset', 'Gdataset', 'lrssl'])
    parser.add_argument('-K', type=int, default=10)
    parser.add_argument('-seqlen_drug_behavior', type=int, default=6)
    parser.add_argument('-seqlen_disease_behavior', type=int, default=4)

    # model
    parser.add_argument('-seed', default=125, type=int)
    parser.add_argument('-dropout', type=float, default=0.01)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-promote_embedding_dim', type=int, default=1024)

    # train
    parser.add_argument('-execute_stage1', type=bool, default=False)
    parser.add_argument('-stage1_epochs', default=500, type=int)
    parser.add_argument('-stage1_batch_size', default=1024, type=int)
    parser.add_argument('-stage1_lr', default=0.01, type=float)
    parser.add_argument('-epochs', default=1, type=int)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-lr', default=0.0001, type=float)
    parser.add_argument('-rating_T', type=float, default=3)
    parser.add_argument('-sparse', default=1, type=float)

    # test
    parser.add_argument('-eval_margin', default=1, type=int)
    parser.add_argument('-T_score', default=0.50, type=float)
    return parser

def test(model, testLoader, args):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(testLoader, desc='Test'):
        # for batch in testLoader:
            prediction = model(batch).squeeze(-1)
            results.append((batch[0], batch[3], prediction, batch[6]))

    auc, aupr = evaluate_test(results, args)
    results_auroc.append(auc)
    results_aupr.append(aupr)
    return auc, aupr

def train(args, model, trainLoader, testLoader):
    writer = SummaryWriter('runs/')
    rel_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(trainLoader, desc="Train epoch " + str(epoch)):
        # for batch in trainLoader:
            prediction = model(batch).squeeze(-1)
            label = batch[6].to(device).type(torch.float)
            loss = rel_loss(prediction, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        # print(f'loss={running_loss}')

        if (epoch + 1) % args.eval_margin == 0:
            auroc, aupr = test(model, testLoader, args)
            print(f'epoch={epoch+1}, auroc={auroc}, aupr={aupr}')
            time.sleep(0.02)
            writer.add_scalar('auroc', auroc, epoch / args.eval_margin)
            writer.add_scalar('aupr', aupr, epoch / args.eval_margin)
            writer.add_scalar('loss', running_loss, epoch)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = getArgs()
    args = parser.parse_args()
    print_args(args)

    # stage 1
    if args.execute_stage1:
        print("================= stage 1 =================")
        drug_promotes = train_promotes(args, 'drug')
        disease_promotes = train_promotes(args, 'disease')
        torch.save(drug_promotes, os.path.join('Stage1', args.dataset, str(args.promote_embedding_dim), 'promotes_drug.pt'))
        torch.save(disease_promotes, os.path.join('Stage1', args.dataset, str(args.promote_embedding_dim), 'promotes_disease.pt'))

    # stage 2
    print("================= stage 2 =================")
    seeds = [12, 34, 42, 43, 61, 70, 83, 1024, 2014, 2047]
    for times in range(0, args.K):
        seed = seeds[times]
        args.seed = seed
        set_random_seed(seed)
        dataset = BasicDataset(args)
        similarity_matrix = [dataset.drug_sim, dataset.disease_sim]

        for cv in range(0, args.K):
            print("================= time:" + str(times + 1) + ' cv:' + str(cv + 1) + "===================")

            train_position = dataset.cv_data[cv][0].tolist()
            test_position = dataset.cv_data[cv][1].tolist()

            # Load training data
            traindata = trainDataset(args)
            traindata.get_all_sequence_train(train_position)
            trainLoader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, drop_last=True)

            # Load testing data
            testdata = testDataset(args)
            testdata.get_all_sequence_test(train_position, test_position)
            testLoader = DataLoader(testdata, batch_size=args.batch_size)

            # Load model
            model = DrugModel(dataset.Wrname_len, dataset.Wdname_len, similarity_matrix, args)
            model.to(device)

            # train
            train(args, model, trainLoader, testLoader)

    auroc = np.mean(results_auroc)
    aupr = np.mean(results_aupr)
    print('auroc:', auroc, 'aupr:', aupr)
    save_experiment_results(args, auroc, aupr)