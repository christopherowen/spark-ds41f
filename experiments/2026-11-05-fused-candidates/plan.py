"""Out-of-band compact plan with fused exact candidate construction."""
import math
import torch
from vllm.models.deepseek_v4_1.tilelang.compact_shortlist import CompactShortlistPlan
from finish import FusedCoarse

class Plan(CompactShortlistPlan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.coarse=FusedCoarse(self.width//8,self.page,self.block_budget)

    def _specs(self,rows):
        result=super()._specs(rows)
        result.pop('selected');result.pop('values')
        return result

    def run(self, q, scales, weights, pool, table, lengths, active, candidates=None, candidate_lengths=None,
            *, out=None, candidate_out=None, candidate_out_lengths=None, scratch=None):
        if candidates is not None or candidate_lengths is not None:
            raise ValueError('compact summaries cannot consume reindex candidates')
        if out is None or scratch is None:
            raise ValueError('outputs and scratch must be admitted')
        if self.source and (candidate_out is None or candidate_out_lengths is None):
            raise ValueError('source handoff outputs are required')
        rows = len(q)
        specs = self._specs(rows)
        if len(scratch) != len(specs):
            raise ValueError('workspace differs from admission')
        full = dict(zip(specs, scratch))
        means = pool[:, self.page*68:self.page*68+(self.page//8)*68]
        def buffers(name, plan, b):
            return [b[f'{name}_{i}'] for i in range(len(plan.scratch_specs(1)))]
        for start in range(0, rows, self.row_cap):
            end = min(start+self.row_cap, rows)
            r = end-start
            current = self._specs(r)
            b = {name:t.view(-1)[:math.prod(current[name][0])].view(current[name][0])
                 for name, t in full.items()}
            qr, sr, wr, tr, lr = q[start:end], scales[start:end], weights[start:end], table[start:end], lengths[start:end]
            self.score(qr.view(torch.int8), sr.view(torch.uint32).reshape(r, 32), wr,
                       means.view(torch.int8), means.view(torch.uint32), tr, lr, active, b['scores'], b['counts'])
            self.coarse.run(b['scores'],b['counts'],lr,active,b['candidates'],b['candidate_lengths'],
                            scratch=buffers('coarse',self.coarse,b))
            _, scores = self.final.run(qr, sr, wr, pool, tr, lr, active, b['candidates'], b['candidate_lengths'],
                                      out=out[start:end], scratch=buffers('final', self.final, b))
            if self.source:
                self.select_blocks.run(scores, b['candidate_lengths'], active, b['compact'], b['block_values'],
                                       counts=candidate_out_lengths[start:end], scratch=buffers('blocks', self.select_blocks, b))
                self.remap(b['compact'], b['candidates'], candidate_out[start:end])
        return out, None
