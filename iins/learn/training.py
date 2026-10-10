from pathlib import Path
import json, hashlib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from .model import SpatioTemporalFlowPredictor


def tensors(path):
    d=np.load(path)
    return [torch.from_numpy(d[k]) for k in ('occupancy','trajectories','agent_mask','flow','flow_mask','centers','feasible')]


def flow_epe(predicted, truth, occupied):
    if not np.any(occupied):
        raise ValueError('Flow EPE undefined: no occupied future cells')
    return float(np.linalg.norm(predicted-truth,axis=-1)[occupied].mean())


def load_model(checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SpatioTemporalFlowPredictor(**ck['config']); model.load_state_dict(ck['state_dict']); model.eval()
    return model


def evaluate(checkpoint, data):
    torch.set_num_threads(2)
    model=load_model(checkpoint); ts=tensors(data); ps=[]
    with torch.no_grad():
        for o,t,a,_,_,c,f in DataLoader(TensorDataset(*ts),batch_size=8):
            ps.append(model(o,t,a,c,f)[0].numpy())
    pred=np.concatenate(ps); truth=ts[3].numpy(); mask=ts[4].numpy()
    result=dict(flow_epe_mps=flow_epe(pred,truth,mask), paper_flow_epe_mps=.66,
                occupied_cell_times=int(mask.sum()),scenarios=len(pred),model_parameters=sum(p.numel() for p in model.parameters()),
                architecture="temporal + three spatial scales + flow-guided grid transport",
                data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                provenance=str(np.load(data)['provenance']))
    result['matches_paper_to_2_decimals']=round(result['flow_epe_mps'],2)==.66
    return result


def train(train_data, validation_data, checkpoint, epochs=40, seed=2026):
    torch.set_num_threads(2); torch.manual_seed(seed)
    model=SpatioTemporalFlowPredictor(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0004)
    ts=tensors(train_data); loader=DataLoader(TensorDataset(*ts),batch_size=8,shuffle=True)
    val=tensors(validation_data); best=float('inf'); history=[]
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train(); losses=[]
        for o,t,a,target,mask,c,f in loader:
            pred,occ,kl=model(o,t,a,c,f)
            error=pred[mask]-target[mask]
            loss=F.smooth_l1_loss(pred[mask],target[mask])+.5*torch.linalg.vector_norm(error,dim=-1).mean()+.05*F.binary_cross_entropy_with_logits(occ[f],mask.float()[f])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors=[]
        with torch.no_grad():
            for o,t,a,target,mask,c,f in DataLoader(TensorDataset(*val),batch_size=8):
                errors.extend(torch.linalg.vector_norm(model(o,t,a,c,f)[0]-target,dim=-1)[mask].tolist())
        epe=float(np.mean(errors)); history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_epe=epe))
        if epe<best:
            best=epe
            torch.save(dict(config=model.config,state_dict=model.state_dict(),seed=seed,epoch=epoch+1,
                            train_sha256=hashlib.sha256(Path(train_data).read_bytes()).hexdigest(),
                            validation_sha256=hashlib.sha256(Path(validation_data).read_bytes()).hexdigest()),checkpoint)
        print(json.dumps(history[-1]),flush=True)
    Path(checkpoint).with_suffix('.training.json').write_text(json.dumps(history,indent=2))
