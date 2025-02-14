import torch
from args import args
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_rep_with_label(model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label,clip,clip_moreinf = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep = model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels

def get_freqrep_with_label(freqtime_model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label,clip_moreinf = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep,encoded,xcls = freqtime_model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels

def fit_lr(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe

# THIS IS BEING USED
def get_rep_with_label_with_image_name(model, dataloader):
    reps = []
    clips=[]
    last_reps=[]
    labels = []
    preds= []
    seqs=[]
    scores= []
    image_names=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label, image_name = batch
            # print("seq is", seq)
            # print("label is", label)
            # print("image_name is", image_name)
            seqs+=seq.cpu().numpy().tolist()
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep, encoded, score = model(seq)
            reps+=rep.cpu().numpy().tolist()
            image_names+=list(image_name)
            _, pred = torch.topk(score, 1)
            preds += pred.cpu().numpy().tolist()
            # print(label.shape, len(labels), score.shape, pred.shape, len(preds))
            # print(f'len(reps) - {len(reps)}\t rep.shape - {rep.shape}')
            
    return labels,image_names, preds,seqs, reps,0

def get_rep_with_label_with_image_name_seq(model, dataloader):
    reps = []
    clips=[]
    last_reps=[]
    labels = []
    preds= []
    seqs=[]
    scores= []
    image_names=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label, image_name = batch
            # print("seq is", seq)
            # print("label is", label)
            # print("image_name is", image_name)
            seqs+=seq.cpu().numpy().tolist()
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            score, encoded = model(seq)
            reps+=encoded.cpu().numpy().tolist()
            image_names+=list(image_name)
            _, pred = torch.topk(score, 1)
            preds += pred.cpu().numpy().tolist()
            # print(label.shape, len(labels), score.shape, pred.shape, len(preds))
            # print(f'len(reps) - {len(reps)}\t rep.shape - {rep.shape}')
            
    return labels,image_names, preds,seqs, reps,0