# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-workspace serving adapter for the experimental exact snapshot search.

Rows keep their own page tables. No cache-owner equivalence or persistent
metadata is inferred. At most eight rows use scratch simultaneously; native
attention owns all published outputs. Every kernel is declared before serving.
"""
import os
import math
import torch

from . import prepare
from .clustered_indexer import (CLUSTER, cluster_envelopes, cluster_bounds_tensor,
                                seed_candidates, surviving_candidates)
from .source_indexer import (source_threshold, retain_last_cluster,
                             source_surviving_candidates, map_source_blocks)
from .indexer import (IndexerV41Plan, IndexerTopK, INDEX_TOPK, INDEX_DIM,
                      CANDIDATE_BLOCKS, CANDIDATE_BLOCK, page_bytes)

ROW_CAP = 8


def enabled():
    return os.environ.get('VLLM_DS41_CLUSTERED_SEARCH', '0') == '1'


def use_search(width, candidates, candidate_blocks):
    return enabled() and not candidates and width > (32768 if candidate_blocks else 2048)


def index_plan(width, page_size, candidates, candidate_blocks):
    if use_search(width, candidates, candidate_blocks):
        return ServingClusteredIndexerPlan(width, page_size, candidate_blocks=candidate_blocks)
    return IndexerV41Plan(width, page_size, candidates=candidates, candidate_blocks=candidate_blocks)


def index_calls(width, page_size, candidates, candidate_blocks):
    if use_search(width, candidates, candidate_blocks):
        return ServingClusteredIndexerPlan.calls(width, page_size, candidate_blocks=candidate_blocks)
    return IndexerV41Plan.calls(width, page_size, candidates=candidates, candidate_blocks=candidate_blocks)


class ServingClusteredIndexerPlan:
    def __init__(self, width, page_size, *, candidate_blocks=False):
        self.width, self.page_size, self.source = width, page_size, candidate_blocks
        self.seeds = min(512 if self.source else 32, width // CLUSTER)
        self.envelope = cluster_envelopes(width, page_size)
        self.bounds = cluster_bounds_tensor(width)
        self.coarse = IndexerTopK(width // CLUSTER, self.seeds)
        self.seed = seed_candidates(width, self.seeds)
        self.seed_plan = IndexerV41Plan(self.seeds * CLUSTER, page_size, candidates=True)
        self.final_plan = IndexerV41Plan(width, page_size, candidates=True)
        if self.source:
            self.seed_blocks = IndexerTopK(self.seeds * CLUSTER, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
            self.select_blocks = IndexerTopK(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
            self.threshold = source_threshold(self.seeds * 8)
            self.last = retain_last_cluster(width)
            self.compact = source_surviving_candidates(width, CANDIDATE_BLOCKS)
            self.remap = map_source_blocks(width)
        else:
            self.compact = surviving_candidates(width, INDEX_TOPK)

    @staticmethod
    def calls(width, page_size, *, candidate_blocks=False):
        seeds = min(512 if candidate_blocks else 32, width // CLUSTER)
        calls = [prepare.call(cluster_envelopes, width, page_size),
                 prepare.call(cluster_bounds_tensor, width),
                 *IndexerTopK.calls(width // CLUSTER, seeds),
                 prepare.call(seed_candidates, width, seeds),
                 *IndexerV41Plan.calls(seeds * CLUSTER, page_size, candidates=True),
                 *IndexerV41Plan.calls(width, page_size, candidates=True)]
        if candidate_blocks:
            calls.extend([*IndexerTopK.calls(seeds * CLUSTER, CANDIDATE_BLOCKS, CANDIDATE_BLOCK),
                          *IndexerTopK.calls(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK),
                          prepare.call(source_threshold, seeds * 8),
                          prepare.call(retain_last_cluster, width),
                          prepare.call(source_surviving_candidates, width, CANDIDATE_BLOCKS),
                          prepare.call(map_source_blocks, width)])
        else:
            calls.append(prepare.call(surviving_candidates, width, INDEX_TOPK))
        return calls

    def _specs(self, rows):
        r, c = min(rows, ROW_CAP), self.width // CLUSTER
        specs = {
            'lower': ((r,c,INDEX_DIM),torch.float32),
            'upper': ((r,c,INDEX_DIM),torch.float32),
            'bounds': ((r,c),torch.bfloat16),
            'counts': ((r,),torch.int32),
            'coarse_ids': ((r,self.seeds),torch.int32),
            'coarse_values': ((r,self.seeds),torch.bfloat16),
            'seed_ids': ((r,self.seeds*8),torch.int32),
            'seed_counts': ((r,),torch.int32),
            'seed_out': ((r,INDEX_TOPK),torch.int32),
            'candidates': ((r,self.width//8),torch.int32),
            'candidate_counts': ((r,),torch.int32),
        }
        for prefix, plan in [('coarse',self.coarse),('seed',self.seed_plan),('final',self.final_plan)]:
            for i,spec in enumerate(plan.scratch_specs(r)):
                specs[f'{prefix}_{i}'] = spec
        if self.source:
            for name,dtype,shape in [
                ('seed_block_ids',torch.int32,(r,CANDIDATE_BLOCKS)),
                ('seed_block_values',torch.bfloat16,(r,CANDIDATE_BLOCKS)),
                ('seed_block_counts',torch.int32,(r,)),
                ('thresholds',torch.bfloat16,(r,CANDIDATE_BLOCKS)),
                ('cutoff_keys',torch.int32,(r,)),('cutoff_positions',torch.int32,(r,)),
                ('compact_blocks',torch.int32,(r,CANDIDATE_BLOCKS)),
                ('block_values',torch.bfloat16,(r,CANDIDATE_BLOCKS))]:
                specs[name]=(shape,dtype)
            for prefix,plan in [('seed_blocks',self.seed_blocks),('select_blocks',self.select_blocks)]:
                for i,spec in enumerate(plan.scratch_specs(r)):
                    specs[f'{prefix}_{i}']=spec
        return specs

    def scratch_specs(self, rows):
        return tuple(self._specs(rows).values())

    def run(self, q, scales, weights, pool, table, lengths, active,
            candidates=None, candidate_lengths=None, *, out=None, candidate_out=None,
            candidate_out_lengths=None, scratch=None):
        if candidates is not None or candidate_lengths is not None:
            raise ValueError('dense/source search adapter does not accept reindex input')
        if out is None or scratch is None:
            raise ValueError('serving owns output and reserves all search scratch before profiling')
        if self.source and (candidate_out is None or candidate_out_lengths is None):
            raise ValueError('source search requires native handoff outputs')
        rows = q.shape[0]
        specs = self._specs(rows)
        if len(scratch) != len(specs):
            raise ValueError('reserved search scratch differs from declared workspace')
        full = dict(zip(specs,scratch))
        nbytes = page_bytes(self.page_size)
        def buffers(prefix,plan,b):
            return [b[f'{prefix}_{i}'] for i in range(len(plan.scratch_specs(1)))]
        for start in range(0,rows,ROW_CAP):
            end=min(start+ROW_CAP,rows);r=end-start
            current_specs=self._specs(r)
            b={name:value.view(-1)[:math.prod(current_specs[name][0])].view(current_specs[name][0])
               for name,value in full.items()}
            qr,sr,wr,tr,lr=q[start:end],scales[start:end],weights[start:end],table[start:end],lengths[start:end]
            self.envelope(pool[:, :nbytes],tr,lr,active,b['lower'],b['upper'])
            self.bounds(qr.view(torch.uint8),sr.view(torch.uint8).reshape(r,32,4),wr,
                        b['lower'],b['upper'],lr,active,b['bounds'],b['counts'])
            # counts <= ceil(visible/64) <= visible for nonempty rows, so native
            # Active safely bounds this coarse domain without a device fill.
            self.coarse.run(b['bounds'],b['counts'],active,b['coarse_ids'],b['coarse_values'],
                            scratch=buffers('coarse',self.coarse,b))
            self.seed(b['coarse_ids'],lr,active,b['seed_ids'],b['seed_counts'])
            if self.source:
                self.seed_plan.score(qr.view(torch.int8),sr.view(torch.uint32).reshape(r,-1),wr,
                    pool.view(torch.int8)[:,:nbytes],pool.view(torch.uint32)[:,:nbytes//4],
                    tr,lr,active,b['seed_ids'],b['seed_counts'],b['seed_0'])
                self.seed_blocks.run(b['seed_0'],b['seed_counts'],active,b['seed_block_ids'],
                    b['seed_block_values'],counts=b['seed_block_counts'],
                    scratch=buffers('seed_blocks',self.seed_blocks,b))
                self.threshold(b['seed_block_ids'],b['seed_block_values'],b['seed_counts'],b['seed_ids'],
                    b['cutoff_keys'],b['cutoff_positions'],b['thresholds'])
                self.last(b['bounds'],lr,active)
                self.compact(b['bounds'],b['thresholds'],b['cutoff_keys'],b['cutoff_positions'],
                             lr,active,b['candidates'],b['candidate_counts'])
            else:
                self.seed_plan.run(qr,sr,wr,pool,tr,lr,active,b['seed_ids'],b['seed_counts'],
                                   out=b['seed_out'],
                                   scratch=buffers('seed',self.seed_plan,b))
                self.compact(b['bounds'],b['seed_1'],lr,active,b['candidates'],b['candidate_counts'])
            _,scores=self.final_plan.run(qr,sr,wr,pool,tr,lr,active,b['candidates'],b['candidate_counts'],
                                        out=out[start:end],scratch=buffers('final',self.final_plan,b))
            if self.source:
                self.select_blocks.run(scores,b['candidate_counts'],active,b['compact_blocks'],b['block_values'],
                    counts=candidate_out_lengths[start:end],scratch=buffers('select_blocks',self.select_blocks,b))
                self.remap(b['compact_blocks'],b['candidates'],candidate_out[start:end])
        return out,None
