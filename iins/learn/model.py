import torch
from torch import nn
import torch.nn.functional as F


def encoder(width,layers=1):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True),layers,enable_nested_tensor=False)


class GaussianLatentBottleneck(nn.Module):
    """History-conditioned diagonal Gaussian, regularized toward N(0,I).
    """
    def __init__(self, width, latent_dim):
        super().__init__()
        self.mu=nn.Linear(width,latent_dim)
        self.logvar=nn.Linear(width,latent_dim)
        self.decoder=nn.Sequential(nn.Linear(latent_dim,width),nn.GELU(),nn.LayerNorm(width))

    def encode(self, features):
        return self.mu(features),self.logvar(features).clamp(-8.,4.)

    def forward(self, features, mask, sample=False):
        mu,logvar=self.encode(features)
        z=mu+torch.exp(.5*logvar)*torch.randn_like(mu) if sample else mu
        # Sum latent dimensions, then mean only over feasible cell-times.
        per_token=.5*(mu.square()+logvar.exp()-1.-logvar).sum(-1)
        weight=mask.to(per_token.dtype)
        kl=(per_token*weight).sum()/weight.sum().clamp_min(1.)
        decoded=self.decoder(z)*weight[...,None]
        return decoded,kl


class SpatioTemporalFlowPredictor(nn.Module):
    def __init__(self,history=7,future=6,width=48,latent_dim=0,speed_scale_mps=None):
        super().__init__();self.config=dict(history=history,future=future,width=width,latent_dim=latent_dim,speed_scale_mps=speed_scale_mps);self.future=future;self.width=width
        if latent_dim<0: raise ValueError("latent_dim must be nonnegative")
        if speed_scale_mps is not None and not (0 < speed_scale_mps < float("inf")):
            raise ValueError("speed_scale_mps must be finite and positive")
        self.speed_scale_mps=speed_scale_mps
        self.latent_dim=latent_dim
        self.actor_input=nn.Linear(7,width);self.temporal_position=nn.Parameter(torch.randn(1,history,width)*.02)
        self.temporal=encoder(width,2);self.social=encoder(width)
        self.motion=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,future*2))
        self.visual=nn.Sequential(nn.Conv3d(1,12,(3,3,3),padding=1),nn.GELU(),nn.Conv3d(12,12,(history,1,1)),nn.GELU())
        self.map_input=nn.Conv2d(15,width,3,padding=1)
        self.window=encoder(width);self.stage2=encoder(width,2);self.stage3=encoder(width,2)
        self.fusion=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.LayerNorm(width))
        self.query=nn.Linear(width,width);self.key=nn.Linear(width,width)
        self.future_position=nn.Parameter(torch.randn(future,width)*.02)
        self.residual=nn.Linear(width,2);self.occupancy=nn.Linear(width,1)
        self.latent=GaussianLatentBottleneck(width,latent_dim) if latent_dim else None
        self.speed_head=nn.Linear(width,1) if speed_scale_mps is not None else None
        if self.speed_head is not None:
            nn.init.zeros_(self.speed_head.weight);nn.init.zeros_(self.speed_head.bias)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.motion[-1].weight);nn.init.zeros_(self.motion[-1].bias)

    def spatial(self,occupancy,centers,feasible):
        b,h,w,_=occupancy.shape
        visual=self.visual(occupancy.permute(0,3,1,2)[:,None]).squeeze(2)
        relative=(centers-centers.mean((1,2),keepdim=True))/64
        x=self.map_input(torch.cat([visual,relative.permute(0,3,1,2),feasible[:,None].float()],1))
        # Windows avoid quadratic attention across the full-resolution map.
        ph=(-h)%4;pw=(-w)%4;x=F.pad(x,(0,pw,0,ph));hh,ww=x.shape[-2:]
        windows=x.reshape(b,self.width,hh//4,4,ww//4,4).permute(0,2,4,3,5,1).reshape(-1,16,self.width)
        windows=self.window(windows)
        fine=windows.reshape(b,hh//4,ww//4,4,4,self.width).permute(0,5,1,3,2,4).reshape(b,self.width,hh,ww)[...,:h,:w]
        fine=fine*feasible[:,None]
        coarse=F.avg_pool2d(fine,2,ceil_mode=True);hc,wc=coarse.shape[-2:]
        coarse=self.stage2(coarse.flatten(2).transpose(1,2)).transpose(1,2).reshape(b,self.width,hc,wc)
        deep=F.avg_pool2d(coarse,2,ceil_mode=True);hd,wd=deep.shape[-2:]
        deep=self.stage3(deep.flatten(2).transpose(1,2)).transpose(1,2).reshape(b,self.width,hd,wd)
        fused=fine+F.interpolate(coarse,(h,w),mode='bilinear',align_corners=False)+F.interpolate(deep,(h,w),mode='bilinear',align_corners=False)
        return fused.flatten(2).transpose(1,2)

    def forward(self,occupancy,trajectories,agent_mask,centers,feasible,latent_sample=None):
        b,h,w,th=occupancy.shape;n=trajectories.shape[1]
        # Translation invariance; speed differences and per-step displacement remain physical.
        pos=trajectories[...,:2];delta=torch.diff(pos,dim=2,prepend=pos[:,:,:1])/.5
        delta[:,:,0]=delta[:,:,1]
        rel=(pos-pos[:,:,-1:])/20
        features=torch.cat([rel,delta/20,trajectories[...,2:]/20,(pos-centers.mean((1,2))[:,None,None])/64],-1)
        actor=self.temporal(self.actor_input(features).reshape(b*n,th,self.width)+self.temporal_position)[:,-1].reshape(b,n,self.width)
        # Null actor permits genuinely empty scenes without numerical NaNs.
        actor=torch.cat([actor,actor.new_zeros(b,1,self.width)],1)
        valid=torch.cat([agent_mask,torch.ones(b,1,dtype=torch.bool,device=agent_mask.device)],1)
        actor=self.social(actor,src_key_padding_mask=~valid)
        correction=self.motion(actor).reshape(b,n+1,self.future,2)
        measured=torch.cat([delta[:,:,-1],delta.new_zeros(b,1,2)],1)
        # Predict changes to measured world velocity; never a fixed per-cell speed.
        velocity=measured[:,:,None]+correction
        current=torch.cat([pos[:,:,-1],centers.mean((1,2))[:,None]],1)
        future_xy=current[:,:,None]+torch.cumsum(velocity*.5,dim=2)
        visual=self.spatial(occupancy,centers,feasible)
        query=self.query(visual);key=self.key(actor)
        appearance=torch.einsum('bqd,bnd->bqn',query,key)/(self.width**.5)
        cell=centers.flatten(1,2)
        dx=(centers[:,1,0,0]-centers[:,0,0,0]).abs().clamp_min(.5)
        dy=(centers[:,0,1,1]-centers[:,0,0,1]).abs().clamp_min(.5)
        scale=torch.stack([dx,dy],-1)[:,None,None,None]
        dist=((cell[:,:,None,None]-future_xy[:,None])/scale).square().sum(-1)
        logits=appearance[:,:,:,None]*.05-dist*2.
        logits=logits.masked_fill(~valid[:,None,:,None],-1e6)
        logits[:,:,-1]-=30.  # learned empty-scene token, low weight when real actors exist
        attention=logits.softmax(2)
        motion=torch.einsum('bqnt,bntc->bqtc',attention,velocity)
        context=torch.einsum('bqnt,bnd->bqtd',attention,actor)
        # Flow-guided grid-to-grid transport: future query cells attend to their
        # upstream historical features, respecting the irregular feasible mask.
        times=torch.arange(1,self.future+1,device=motion.device,dtype=motion.dtype)*.5
        upstream=cell[:,:,None]-motion*times[None,None,:,None]
        minimum=centers[:,0,0];extent=centers[:,-1,-1]-minimum
        normalized=2*(upstream-minimum[:,None,None])/extent[:,None,None].clamp_min(.1)-1
        # grid_sample axes are (column Y, row X).
        sample=normalized.flip(-1).permute(0,2,1,3).reshape(b*self.future,h,w,2)
        feature=visual.transpose(1,2).reshape(b,self.width,h,w)*feasible[:,None]
        transported=F.grid_sample(feature[:,None].expand(-1,self.future,-1,-1,-1).reshape(b*self.future,self.width,h,w),sample,align_corners=True,padding_mode='zeros')
        transported=transported.reshape(b,self.future,self.width,h*w).permute(0,3,1,2)
        visual_time=visual[:,:,None]+transported
        fused=self.fusion(torch.cat([visual_time,context],-1))+self.future_position
        kl=fused.new_zeros(())
        if self.latent is not None:
            mask=feasible.flatten(1)[:,:,None].expand(-1,-1,self.future)
            fused,kl=self.latent(fused,mask,sample=self.training if latent_sample is None else latent_sample)
        flow=motion+self.residual(fused)
        if self.speed_head is not None:
            # Bounded scalar speed, with direction retained in world coordinates.
            magnitude=torch.linalg.vector_norm(flow,dim=-1,keepdim=True)
            direction=flow/magnitude.clamp_min(1e-8)
            fallback=torch.zeros_like(direction);fallback[...,0]=1.
            direction=torch.where(magnitude>1e-8,direction,fallback)
            prior=(torch.linalg.vector_norm(motion,dim=-1)/self.speed_scale_mps).clamp(1e-4,1-1e-4)
            speed_logits=torch.logit(prior)+self.speed_head(fused).squeeze(-1)
            flow=direction*(self.speed_scale_mps*speed_logits.sigmoid())[...,None]
        occ=self.occupancy(fused).squeeze(-1)-dist.min(2).values
        flow=flow.reshape(b,h,w,self.future,2)*feasible[...,None,None]
        return flow,occ.reshape(b,h,w,self.future),kl
