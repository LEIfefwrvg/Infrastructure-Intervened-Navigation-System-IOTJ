from pathlib import Path
import json, hashlib, platform, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from .model import SpatioTemporalFlowPredictor
from .data import load


def tensors(path):
    return [torch.from_numpy(array) for array in load(path)]


def flow_epe(predicted, truth, occupied):
    if not np.any(occupied):
        raise ValueError('Flow EPE undefined: no occupied future cells')
    return float(np.linalg.norm(predicted-truth,axis=-1)[occupied].mean())


def normalized_speed_bce(predicted, target, occupied, speed_scale_mps):
    """Speed BCE on observed scalar magnitudes; never clip out-of-range labels."""
    if not (0 < speed_scale_mps < float('inf')):
        raise ValueError('Speed normalization requires a finite positive scale')
    if not occupied.any():
        raise ValueError('Speed BCE undefined: no occupied cells')
    truth=torch.linalg.vector_norm(target[occupied],dim=-1)/speed_scale_mps
    prediction=torch.linalg.vector_norm(predicted[occupied],dim=-1)/speed_scale_mps
    if not torch.isfinite(truth).all() or (truth>1).any():
        raise ValueError('Speed labels exceed normalization range; do not clip targets')
    if not torch.isfinite(prediction).all() or (prediction>1+1e-6).any():
        raise ValueError('Speed prediction must come from a bounded output head')
    # Only numerical endpoint protection, not a remedy for unbounded predictions.
    return F.binary_cross_entropy(prediction.clamp(1e-7,1-1e-7),truth)


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
                architecture="temporal + three spatial scales + flow-guided grid transport"+(" + Gaussian latent bottleneck" if model.latent_dim else ""),
                speed_scale_mps=model.speed_scale_mps,latent_dim=model.latent_dim,latent_inference="posterior-mean plug-in" if model.latent_dim else "disabled",
                data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),device='cpu'))
    result['speed_mae_mps']=float(np.abs(np.linalg.norm(pred[mask],axis=-1)-np.linalg.norm(truth[mask],axis=-1)).mean())
    if model.speed_scale_mps is not None:
        result['normalized_speed_bce']=float(normalized_speed_bce(torch.from_numpy(pred),ts[3],ts[4],model.speed_scale_mps))
        result['architecture']+=' + bounded scalar speed head'
    result['matches_paper_to_2_decimals']=round(result['flow_epe_mps'],2)==.66
    return result


def train(train_data, checkpoint, epochs=40, seed=2026, latent_dim=0, kl_weight=.001, kl_warmup_epochs=5, reconstruction="normalized-speed-bce", speed_scale_mps=20.):
    if epochs<1: raise ValueError('epochs must be positive')
    if kl_weight<0 or kl_warmup_epochs<0: raise ValueError('KL weight and warmup must be nonnegative')
    if latent_dim and kl_weight<=0: raise ValueError('Latent training requires a positive KL weight')
    if reconstruction not in ('legacy','normalized-speed-bce'):
        raise ValueError('Unknown reconstruction objective')
    speed_bce_enabled=reconstruction=='normalized-speed-bce'
    torch.set_num_threads(2); torch.manual_seed(seed)
    model=SpatioTemporalFlowPredictor(latent_dim=latent_dim,speed_scale_mps=speed_scale_mps if speed_bce_enabled else None); optimizer=torch.optim.AdamW(model.parameters(),lr=.0004)
    ts=tensors(train_data)
    if speed_bce_enabled:
        normalized_speed_bce(ts[3],ts[3],ts[4],speed_scale_mps)
    loader=DataLoader(TensorDataset(*ts),batch_size=8,shuffle=True)
    history=[]
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    protocol=dict(epochs=epochs,seed=seed,initialization='from scratch',optimizer='AdamW',learning_rate=.0004,
                  batch_size=8,latent_dim=latent_dim,kl_weight=kl_weight if latent_dim else 0.,kl_warmup_epochs=kl_warmup_epochs if latent_dim else 0,
                  latent_inference='posterior-mean plug-in' if latent_dim else 'disabled',
                  reconstruction=reconstruction,speed_scale_mps=model.speed_scale_mps,
                  sampling='shuffle',samples_per_epoch=len(ts[0]),
                  selection='final epoch; no validation or test-based selection',
                  train_sha256=hashlib.sha256(Path(train_data).read_bytes()).hexdigest(),
                  runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,device='cpu'))
    checkpoint.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for epoch in range(epochs):
        started=time.monotonic();model.train(); losses=[];recon_losses=[];kl_values=[];speed_losses=[];vector_losses=[];occupancy_losses=[]
        beta=kl_weight*min(1.,(epoch+1)/max(kl_warmup_epochs,1)) if latent_dim else 0.
        for o,t,a,target,mask,c,f in loader:
            pred,occ,kl=model(o,t,a,c,f)
            error=pred[mask]-target[mask]
            speed_loss=normalized_speed_bce(pred,target,mask,speed_scale_mps) if speed_bce_enabled else F.smooth_l1_loss(pred[mask],target[mask])
            vector_loss=torch.linalg.vector_norm(error,dim=-1).mean()
            occupancy_loss=F.binary_cross_entropy_with_logits(occ[f],mask.float()[f])
            reconstruction_loss=speed_loss+.5*vector_loss+.05*occupancy_loss
            speed_losses.append(float(speed_loss.detach()));vector_losses.append(float(vector_loss.detach()));occupancy_losses.append(float(occupancy_loss.detach()))
            loss=reconstruction_loss+beta*kl
            recon_losses.append(float(reconstruction_loss.detach()));kl_values.append(float(kl.detach()))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),reconstruction_loss=float(np.mean(recon_losses)),
                            speed_bce=float(np.mean(speed_losses)) if speed_bce_enabled else None,
                            vector_epe_loss=float(np.mean(vector_losses)),occupancy_bce=float(np.mean(occupancy_losses)),
                            kl=float(np.mean(kl_values)),kl_weight=beta,elapsed_seconds=time.monotonic()-started))
        if epoch+1 == epochs:
            torch.save(dict(config=model.config,state_dict=model.state_dict(),seed=seed,epoch=epoch+1),checkpoint)
        checkpoint.with_suffix('.training.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(history[-1]),flush=True)
