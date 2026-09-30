"""Hyperliquid wallet classification (RQ1/RQ2 empirical companion).

Given the L4 order-lifecycle stream, flag market-maker wallets by:
    * two-sided quoting (fraction of time with both a bid and an ask resting)
    * cancel-to-fill ratio
    * passive share (fraction of fills where the wallet was the resting side)
    * inventory mean reversion (AR(1) coefficient of end-of-block inventory)

TODO: implement once data/download.py has been run; schema in data/README.md.
"""
