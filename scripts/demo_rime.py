"""CPU example using generated features, source selection, and held-out prediction."""
import argparse
import numpy as np
import torch
from if_redesign.rime import fit_projection,project,source_bandwidth,prepare_inputs,train_head

def main(epochs=12):
    torch.set_num_threads(2)
    rng=np.random.default_rng(17);n=64
    raw=rng.normal(size=(n,3,4,32)).astype(np.float32)
    valid=np.ones(raw.shape[:-1],dtype=bool);valid[0]=False
    # Empty OCT evidence retains a caller-supplied clinical/semantic anchor.
    y=(raw[:,0,0,0]+rng.normal(size=n)*.7>0).astype(int)
    groups=np.tile(['source_a','source_b','source_c','selection','heldout'],13)[:n]
    tr=np.flatnonzero(~np.isin(groups,['selection','heldout']))
    va=np.flatnonzero(groups=='selection');te=np.flatnonzero(groups=='heldout')
    projection=fit_projection(raw,valid,tr);h=project(raw,valid,projection)
    sigma=source_bandwidth(h,valid,tr)
    context=rng.normal(size=(n,34));anchor=rng.normal(scale=.1,size=n)
    print('Synthetic features only; held-out targets are never passed to training.')
    for pooling,readout in [('attention','moment'),('attention','laplace'),
                            ('measure','moment'),('measure','laplace')]:
        train=prepare_inputs(h,valid,context,anchor,sigma,source_indices=tr,
            indices=tr,pooling=pooling,readout=readout)
        scale=train['coordinate_scale'].numpy()
        selection=prepare_inputs(h,valid,context,anchor,sigma,fitted_scale=scale,
            indices=va,pooling=pooling,readout=readout)
        # Make the held-out labels inaccessible even through the common vector.
        fit_y=y.copy();fit_y[te]=-1
        head,metadata=train_head(train,selection,fit_y,groups,pooling,readout,.3,.003,17,
            max_epochs=epochs,patience=epochs)
        heldout=prepare_inputs(h,valid,context,anchor,sigma,fitted_scale=scale,
            indices=te,pooling=pooling,readout=readout)
        with torch.no_grad():
            logits=head(**{k:v for k,v in heldout.items() if k!='indices'})
        assert torch.isfinite(logits).all()
        print(f'{pooling}/{readout}: epoch={metadata["best_epoch"]}, '
              f'parameters={metadata["parameters"]}, heldout_predictions={len(logits)}')
    empty=prepare_inputs(h,valid,context,anchor,sigma,fitted_scale=scale,indices=[0])
    with torch.no_grad():
        z=head(**{k:v for k,v in empty.items() if k!='indices'})
    torch.testing.assert_close(z,empty['anchor'])
    print('Exact empty-OCT anchor fallback passed.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args();main(args.epochs)
