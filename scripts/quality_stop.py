#!/usr/bin/env python3
"""Predeclared sequential quality rules; never treat repeated fixed-horizon p-values as valid.

Pair evidence: only discordant blinded pairs in one class and one unchanged arm.
Null: either side wins each discordance with probability 1/2. Under the null,
the beta(1,1) mixture likelihood ratio is a nonnegative martingale, so its
supremum crossing 1/alpha has false-alarm probability at most alpha.
Slip evidence: independent first-attempt binary failures for one fixed arm;
Wald log likelihood ratio compares p0 with predeclared p1. Reset only after
an explicit new monitoring epoch, not after a favorable observation.
"""
import argparse
import json
import math


def paired_e(wins, losses, alpha=0.05):
    if min(wins, losses) < 0 or not 0 < alpha < 1:
        raise ValueError('bad counts or alpha')
    n = wins + losses
    log_e = n * math.log(2) + math.lgamma(wins + 1) + math.lgamma(losses + 1) - math.lgamma(n + 2)
    return log_e, log_e >= -math.log(alpha)


def wald(failures, p0, p1, alpha=0.01, beta=0.1):
    if not 0 < p0 < p1 < 1 or not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError('bad predeclared probabilities')
    score = 0.0
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    for index, failed in enumerate(failures, 1):
        if type(failed) is not bool:
            raise ValueError('outcomes must be booleans')
        score += math.log(p1 / p0) if failed else math.log((1 - p1) / (1 - p0))
        if score >= upper:
            return 'SLIP', index, score
        if score <= lower:
            return 'BASELINE', index, score
    return 'CONTINUE', len(failures), score


def selftest():
    assert paired_e(0, 0)[1] is False
    assert paired_e(30, 0)[1] is True
    assert paired_e(15, 15)[1] is False
    assert wald([True] * 10, .1, .3)[0] == 'SLIP'
    assert wald([False] * 30, .1, .3)[0] == 'BASELINE'
    assert wald([], .1, .3)[0] == 'CONTINUE'
    print('quality stop rules: PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--pairs', help='JSON file: {"wins":n,"losses":n}, one unchanged arm/class')
    p.add_argument('--slip', help='JSON file of first-attempt failure booleans in time order')
    p.add_argument('--p0', type=float)
    p.add_argument('--p1', type=float)
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.pairs:
        x = json.load(open(a.pairs))
        print('paired log_e, stop:', paired_e(x['wins'], x['losses']))
    elif a.slip and a.p0 is not None and a.p1 is not None:
        print('slip state, observations, log_lr:', wald(json.load(open(a.slip)), a.p0, a.p1))
    else:
        p.error('give --selftest, --pairs or --slip with --p0 and --p1')


if __name__ == '__main__':
    main()
