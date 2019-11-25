"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Hyperparameter range specification.
"""

hp_range = {
    "beta": [0.0, 0.01, 0.02, 0.05, 0.1],
    "emb_dropout_rate": [0, 0.1, 0.2, 0.3],
    "ff_dropout_rate": [0, 0.1, 0.2, 0.3],
    "action_dropout_rate": [0.95],
    "bandwidth": [200, 256, 400, 512],
    "relation_only": [True, False],
}
