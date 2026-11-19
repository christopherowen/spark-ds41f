"""Fit a small query-weighted key-clustering metric, entirely out of band.

The metric describes native indexer query directions. It is a clustering proxy,
not a bound on BF16 native scores or a guarantee about final-answer quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

DIMENSION = 32
LAYERS = (2,8,14,20)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def decode(packed,scales):
    assert packed.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert packed.shape[-1] == 64 and scales.shape[-1] == 4
    codes = torch.stack((packed&15,packed>>4),dim=-1).reshape(*packed.shape[:-1],128).long()
    values = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    result = values[codes]*torch.exp2(scales.double().repeat_interleave(32,dim=-1)-127)
    assert bool(result.isfinite().all()),'nonfinite native input requires investigation'
    return result


def covariance(batch):
    q = decode(batch['q'],batch['scales'])
    w = batch['weights'].double()
    assert q.shape[:2] == w.shape and q.shape[1:] == (32,128)
    weighted = (q*w[:,:,None]).reshape(-1,128)
    assert bool(weighted.isfinite().all())
    return weighted.T@weighted,len(weighted)


def projection(matrix,dimension=DIMENSION):
    assert matrix.shape == (128,128) and 0 < dimension <= 128
    values,vectors = torch.linalg.eigh(matrix)
    values = values.clamp(min=0)
    assert float(values.sum()) > 0
    basis = vectors[:,-dimension:].flip(1)
    energy = values[-dimension:].flip(0)
    # One global scale does not change cosine clustering. Retain relative
    # query sensitivity between coordinates while keeping coefficients small.
    projected = basis*torch.sqrt(energy/values.sum())[None,:]
    return projected.float(),basis,float(energy.sum()/values.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replay-receipt',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert not args.out.resolve().is_relative_to(root),'fitted weights must remain outside the Git checkout'
    assert not args.out.exists()
    receipt = json.loads(args.replay_receipt.read_text())
    assert receipt['passed'] and receipt['actual_captures']
    files = sorted(args.inputs.glob('*-rank-0-*.pt'))
    assert len(files) == 12,'three disjoint documents, four native layers, rank 0 only'
    data = {}
    for path in files:
        sha = digest(path)
        assert receipt['files'][path.name]['sha256'] == sha
        batch = torch.load(path,map_location='cpu',weights_only=True)
        case,layer = batch['case_id'],batch['capture_context']['layer_id']
        assert case in ('fit-a','fit-b','validation-a') and layer in LAYERS and batch['rank'] == 0
        assert batch['schema_version'] == 3 and len(batch['q']) == 64 and batch['mode'] == 'prefill'
        assert (case,layer) not in data
        data[case,layer] = (path,sha,batch)
    maps,report = {},{'dimension':DIMENSION,'fit_cases':['fit-a','fit-b'],'validation_case':'validation-a',
        'method':'M = mean over native query heads of w_h^2 q_h q_h^T; leading 32 eigen-directions weighted by sqrt eigenvalue',
        'scope':'key-clustering metric only; no native score bound, completed-answer or speed claim',
        'replay_receipt_sha256':digest(args.replay_receipt),'layers':{}}
    for layer in LAYERS:
        fits = [data[case,layer] for case in ('fit-a','fit-b')]
        covariances = [covariance(row[2]) for row in fits]
        matrix = sum(c for c,n in covariances)/sum(n for c,n in covariances)
        metric,basis,fit_energy = projection(matrix)
        validation = data['validation-a',layer]
        val,n = covariance(validation[2]);val /= n
        val_energy = float(torch.trace(basis.T@val@basis)/torch.trace(val))
        assert torch.isfinite(metric).all() and metric.shape == (128,DIMENSION)
        maps[str(layer)] = metric.contiguous()
        report['layers'][str(layer)] = {'fit_inputs':{row[0].name:row[1] for row in fits},
            'validation_input':{validation[0].name:validation[1]},'fit_retained_query_energy':fit_energy,
            'validation_retained_query_energy':val_energy,'independent_fit_documents':2,
            'independent_validation_documents':1,'replicated_ranks_counted':False}
    args.out.mkdir(parents=True)
    path = args.out/'metrics.pt';torch.save(maps,path)
    report['weights_sha256'] = digest(path)
    (args.out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
