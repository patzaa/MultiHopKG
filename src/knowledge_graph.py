"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Knowledge Graph Environment.
"""

import collections
import os
import pickle
from typing import NamedTuple, Dict, List

import torch
import torch.nn as nn

from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda


class Direction(NamedTuple):
    rel: int
    ent: int


class Fork(NamedTuple):
    ent: int
    directions: List[Direction]


def build_forks(num_entities, e1_to_r_to_e2, bandwidth, page_rank_scores) -> List[Fork]:
    def get_action_space(e1):
        directions = [
            Direction(r, e2) for r in e1_to_r_to_e2[e1] for e2 in e1_to_r_to_e2[e1][r]
        ]
        if len(directions) + 1 >= bandwidth:
            # Base graph pruning
            sorted_action_space = sorted(
                directions, key=lambda x: page_rank_scores[x.ent], reverse=True
            )
            directions = sorted_action_space[:bandwidth]

        directions.insert(0, Direction(NO_OP_RELATION_ID, e1))
        return directions

    return [Fork(e1, get_action_space(e1)) for e1 in range(num_entities)]


def vectorize_space(forks: List[Fork], action_space_size, dummy_r, dummy_e):
    bucket_size = len(forks)
    r_space = torch.zeros(bucket_size, action_space_size) + dummy_r
    e_space = torch.zeros(bucket_size, action_space_size) + dummy_e
    action_mask = torch.zeros(bucket_size, action_space_size)
    for i, fork in enumerate(forks):
        for j, direction in enumerate(fork.directions):
            r_space[i, j] = direction.rel
            e_space[i, j] = direction.ent
            action_mask[i, j] = 1
    return ActionSpace(
        forks, int_var_cuda(r_space), int_var_cuda(e_space), var_cuda(action_mask)
    )


def build_buckets(forks: List[Fork], num_entities, args, dummy_r, dummy_e):
    bucketid2forks = collections.defaultdict(list)
    bucket_inbucket_ids = torch.zeros(num_entities, 2).long()
    num_facts_saved_in_action_table = 0
    for fo in forks:
        b_id = int(len(fo.directions) / args.bucket_interval) + 1
        bucket_inbucket_ids[fo.ent, 0] = b_id
        position_in_bucket = len(bucketid2forks[b_id])
        bucket_inbucket_ids[fo.ent, 1] = position_in_bucket
        bucketid2forks[b_id].append(fo)
        num_facts_saved_in_action_table += len(fo.directions)
    print(
        "Sanity check: {} facts saved in action table".format(
            num_facts_saved_in_action_table - num_entities
        )
    )
    bid2ActionSpace = {
        b_id: vectorize_space(forks, b_id * args.bucket_interval, dummy_r, dummy_e)
        for b_id, forks in bucketid2forks.items()
    }
    return bucket_inbucket_ids, bid2ActionSpace


def sanity_checks(e1_to_r_to_e2):
    num_facts = 0
    out_degrees = collections.defaultdict(int)
    for e1 in e1_to_r_to_e2:
        for r in e1_to_r_to_e2[e1]:
            num_facts += len(e1_to_r_to_e2[e1][r])
            out_degrees[e1] += len(e1_to_r_to_e2[e1][r])
    print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
    print("Sanity check: {} facts in knowledge graph".format(num_facts))


def load_page_rank_scores(input_path, entity2id):
    pgrk_scores = collections.defaultdict(float)
    with open(input_path) as f:
        for line in f:
            e, score = line.strip().split(":")
            e_id = entity2id[e.strip()]
            score = float(score)
            pgrk_scores[e_id] = score
    return pgrk_scores


def answers_to_var(d_l):
    d_v = collections.defaultdict(collections.defaultdict)
    for x in d_l:
        for y in d_l[x]:
            v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
            d_v[x][y] = int_var_cuda(v)
    return d_v


class ActionSpace(NamedTuple):
    forks: List[Fork]
    r_space: torch.Tensor
    e_space: torch.Tensor
    action_mask: torch.Tensor

    def get_slice(self, idx):
        d = {k: getattr(self, k)[idx] for k in self._fields if k!='forks'}
        return ActionSpace(**d,forks=[self.forks[i] for i in idx])


class Observation(NamedTuple):
    source_entity: torch.Tensor
    query_relation: torch.Tensor
    target_entity: torch.Tensor
    is_last: bool
    last_relation: torch.Tensor
    seen_nodes: torch.Tensor

    def get_slice(self, idx):
        d = {k: getattr(self, k)[idx] for k in self._fields if k != "is_last"}
        return Observation(**d, is_last=self.is_last)


class KnowledgeGraph(nn.Module):
    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.type2id, self.id2type = {}, {}
        self.entity2typeid = {}
        self.e1_to_r_to_e2 = None
        self.bandwidth = args.bandwidth
        self.args = args

        self.action_space = None
        self.bucketid2ActionSpace: Dict[int, ActionSpace] = None
        self.unique_r_space = None

        # self.train_subjects = None
        # self.train_objects = None
        # self.dev_subjects = None
        self.dev_objects = None
        # self.all_subjects = None
        self.all_objects = None
        self.train_subject_vectors = None
        self.train_object_vectors = None
        # self.dev_subject_vectors = None
        # self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        print("** Create {} knowledge graph **".format(args.model))
        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.num_graph_convolution_layers = args.num_graph_convolution_layers
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None

        self.define_modules()
        self.initialize_modules()

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(
            os.path.join(data_dir, "entity2id.txt")
        )
        print("Sanity check: {} entities loaded".format(len(self.entity2id)))
        self.type2id, self.id2type = load_index(os.path.join(data_dir, "type2id.txt"))
        print("Sanity check: {} types loaded".format(len(self.type2id)))
        with open(os.path.join(data_dir, "entity2typeid.pkl"), "rb") as f:
            self.entity2typeid = pickle.load(f)
        self.relation2id, self.id2relation = load_index(
            os.path.join(data_dir, "relation2id.txt")
        )
        print("Sanity check: {} relations loaded".format(len(self.relation2id)))

        # Load graph structures
        if self.args.model.startswith("point"):
            # Base graph structure used for training and test
            adj_list_path = os.path.join(data_dir, "adj_list.pkl")
            with open(adj_list_path, "rb") as f:
                self.e1_to_r_to_e2 = pickle.load(f)
            self.preprocess_knowledge_graph(data_dir)

    def get_bucket_and_inbucket_ids(self, entities: torch.Tensor):
        entity2bucketid = self._bucket_inbucket_ids[entities.tolist()]
        bucket_ids = entity2bucketid[:, 0]
        inbucket_ids = entity2bucketid[:, 1]
        return bucket_ids, inbucket_ids

    def preprocess_knowledge_graph(self, data_dir):

        sanity_checks(self.e1_to_r_to_e2)

        page_rank_scores = load_page_rank_scores(
            os.path.join(data_dir, "raw.pgrk"), self.entity2id
        )

        forks = build_forks(
            self.num_entities,
            {
                **self.e1_to_r_to_e2,
                **{self.entity2id[k]: {} for k in ["DUMMY_ENTITY", "NO_OP_ENTITY"]},
            },
            self.bandwidth,
            page_rank_scores,
        )
        if self.args.use_action_space_bucketing:
            self._bucket_inbucket_ids, self.bucketid2ActionSpace = build_buckets(
                forks, self.num_entities, self.args, self.dummy_r, self.dummy_e
            )
        else:
            assert False
            # self.build_action_space(list_of_directions)

    # def build_action_space(self, action_spaces_g):
    #     def get_unique_r_space(e1):
    #         if e1 in self.e1_to_r_to_e2:
    #             return list(self.e1_to_r_to_e2[e1].keys())
    #         else:
    #             return []
    #
    #     def vectorize_unique_r_space(
    #         unique_r_space_list, unique_r_space_size, volatile
    #     ):
    #         bucket_size = len(unique_r_space_list)
    #         unique_r_space = (
    #             torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
    #         )
    #         for i, u_r_s in enumerate(unique_r_space_list):
    #             for j, r in enumerate(u_r_s):
    #                 unique_r_space[i, j] = r
    #         return int_var_cuda(unique_r_space)
    #
    #     action_space_list = list(action_spaces_g)
    #     max_num_actions = max([len(a) for a in action_space_list])
    #     print("Vectorizing action spaces...")
    #     self.action_space = vectorize_space(
    #         action_space_list, max_num_actions, self.dummy_r, self.dummy_e
    #     )
    #     if self.args.model.startswith("rule"):
    #         unique_r_space_list = []
    #         max_num_unique_rs = 0
    #         for e1 in sorted(self.e1_to_r_to_e2.keys()):
    #             unique_r_space = get_unique_r_space(e1)
    #             unique_r_space_list.append(unique_r_space)
    #             if len(unique_r_space) > max_num_unique_rs:
    #                 max_num_unique_rs = len(unique_r_space)
    #         self.unique_r_space = vectorize_unique_r_space(
    #             unique_r_space_list, max_num_unique_rs
    #         )

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
        for file_name in ["raw.kb", "train.triples", "dev.triples", "test.triples"]:
            if (
                "NELL" in self.args.data_dir
                and self.args.test
                and file_name == "train.triples"
            ):
                continue
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name in ["raw.kb", "train.triples"]:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(
                                e2, e1, self.get_inv_relation_id(r), train_subjects
                            )
                            add_object(
                                e2, e1, self.get_inv_relation_id(r), train_objects
                            )
                    if file_name in ["raw.kb", "train.triples", "dev.triples"]:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(
                                e2, e1, self.get_inv_relation_id(r), dev_subjects
                            )
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        # self.train_subjects = train_subjects
        # self.train_objects = train_objects
        # self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        # self.all_subjects = all_subjects
        self.all_objects = all_objects

        # self.train_subject_vectors = answers_to_var(train_subjects)
        self.train_object_vectors = answers_to_var(train_objects)
        # self.dev_subject_vectors = answers_to_var(dev_subjects) # TODO(tilo): why unused?
        # self.dev_object_vectors = answers_to_var(dev_objects)
        # self.all_subject_vectors = answers_to_var(all_subjects)
        self.all_object_vectors = answers_to_var(all_objects)

    def load_fuzzy_facts(self):
        # extend current adjacency list with fuzzy facts
        dev_path = os.path.join(self.args.data_dir, "dev.triples")
        test_path = os.path.join(self.args.data_dir, "test.triples")
        with open(dev_path) as f:
            dev_triples = [l.strip() for l in f.readlines()]
        with open(test_path) as f:
            test_triples = [l.strip() for l in f.readlines()]
        removed_triples = set(dev_triples + test_triples)
        theta = 0.5
        fuzzy_fact_path = os.path.join(self.args.data_dir, "train.fuzzy.triples")
        count = 0
        with open(fuzzy_fact_path) as f:
            for line in f:
                e1, e2, r, score = line.strip().split()
                score = float(score)
                if score < theta:
                    continue
                print(line)
                if "{}\t{}\t{}".format(e1, e2, r) in removed_triples:
                    continue
                e1_id = self.entity2id[e1]
                e2_id = self.entity2id[e2]
                r_id = self.relation2id[r]
                if not r_id in self.e1_to_r_to_e2[e1_id]:
                    self.e1_to_r_to_e2[e1_id][r_id] = set()
                if not e2_id in self.e1_to_r_to_e2[e1_id][r_id]:
                    self.e1_to_r_to_e2[e1_id][r_id].add(e2_id)
                    count += 1
                    if count > 0 and count % 1000 == 0:
                        print("{} fuzzy facts added".format(count))

        self.preprocess_knowledge_graph(self.args.data_dir)

    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_all_relation_embeddings(self):
        return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def virtual_step(self, e_set, r):
        """
        Given a set of entities (e_set), find the set of entities (e_set_out) which has at least one incoming edge
        labeled r and the source entity is in e_set.
        """
        batch_size = len(e_set)
        e_set_1D = e_set.view(-1)
        r_space = self.action_space[0][0][e_set_1D]
        e_space = self.action_space[0][1][e_set_1D]
        e_space = (
            r_space.view(batch_size, -1) == r.unsqueeze(1)
        ).long() * e_space.view(batch_size, -1)
        e_set_out = []
        for i in range(len(e_space)):
            e_set_out_b = var_cuda(unique(e_space[i].data))
            e_set_out.append(e_set_out_b.unsqueeze(0))
        e_set_out = ops.pad_and_cat(e_set_out, padding_value=self.dummy_e)
        return e_set_out

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        if not self.args.relation_only:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            if self.args.model == "complex":
                self.entity_img_embeddings = nn.Embedding(
                    self.num_entities, self.entity_dim
                )
            self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        if self.args.model == "complex":
            self.relation_img_embeddings = nn.Embedding(
                self.num_relations, self.relation_dim
            )
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def initialize_modules(self):
        if not self.args.relation_only:
            nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
