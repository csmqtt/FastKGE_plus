from typing import cast

from .BaseModel import *


class LoraKGE_Layers(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE_Layers, self).__init__(args, kg)
        self.lora_ent_embeddings_list = None
        self.lora_rel_embeddings = None

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def expand_embedding_size(self):
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(self.args.device).double()
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)

    def get_new_ordered_entities(self):
        all_new_entities = {}
        for _ in range(self.kg.snapshots[self.args.snapshot].num_ent, self.kg.snapshots[self.args.snapshot + 1].num_ent):
            all_new_entities[_] = (0, 0)
        nodes_ordered_path = f"./data/{self.args.dataset}/{self.args.snapshot + 1}/train_distance_nodes.txt"
        with open(nodes_ordered_path, "r", encoding="utf-8") as f:
            lines = list(f.readlines())
            for line in lines:
                line = line.strip()
                line_list = line.split("\t")
                node, distance, score = int(line_list[0]), int(line_list[1]), float(line_list[2])
                if node in all_new_entities:
                    all_new_entities[node] = (distance, score)
        all_new_entities = dict(sorted(all_new_entities.items(), key = lambda kv:(kv[1][0], kv[1][1])))
        self.all_new_entities = all_new_entities
        all_new_entities = list(all_new_entities.keys())
        return all_new_entities

    def expand_lora_embeddings(self):
        self.new_ordered_entities = self.get_new_ordered_entities()
        new_ent_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_ent - self.kg.snapshots[self.args.snapshot].num_ent
        new_rel_embeddings_len = self.kg.snapshots[self.args.snapshot + 1].num_rel - self.kg.snapshots[self.args.snapshot].num_rel
        self.lora_ent_len = (new_ent_embeddings_len + int(self.args.num_ent_layers) - 1) // int(self.args.num_ent_layers)
        
        tmp_r = self.args.ent_r
        self.args.ent_r = (self.lora_ent_len // 20) if (self.lora_ent_len // 20) > int(self.args.ent_r) else self.args.ent_r
        if self.args.explore:
            self.args.ent_r = tmp_r
        print(self.args.using_various_ranks)
        if self.args.using_various_ranks:
            ent_node_list = []
            for _, v in self.all_new_entities.items():
                ent_node_list.append(v[1])
            self.args.ent_r_list = []
            for i_layer in range(int(self.args.num_ent_layers)):
                self.args.ent_r_list.append(sum(ent_node_list[i_layer * self.lora_ent_len: (i_layer + 1) * self.lora_ent_len]))
            average_nodes = sum(self.args.ent_r_list) / len(self.args.ent_r_list)
            r_threshold = int(int(self.args.ent_r) * 0.9)
            self.args.ent_r_list = [int(self.args.ent_r) * i / average_nodes if int(self.args.ent_r) * i / average_nodes > r_threshold else r_threshold for i in self.args.ent_r_list]
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        elif self.args.using_various_ranks_reverse:
            self.args.ent_r_list = np.linspace(int(self.args.ent_r) // 2, int(self.args.ent_r) // 2 * 3, int(self.args.num_ent_layers)).tolist()
            self.args.ent_r_list = [int(i) for i in self.args.ent_r_list]
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]
            self.args.ent_r_list = self.args.ent_r_list[::-1]
            assert len(self.args.ent_r_list) == int(self.args.num_ent_layers)
        else:
            self.args.ent_r_list = [int(self.args.ent_r) // int(self.args.num_ent_layers)] * int(self.args.num_ent_layers)
            self.args.ent_r_list = [i if i else 1 for i in self.args.ent_r_list]

        lora_ent_embeddings_list = []
        reference_weight = self.ent_embeddings.weight.data.float()
        ref_mean = reference_weight.mean()
        ref_std = reference_weight.std().clamp_min(1e-6)
        init_method = str(getattr(self.args, "lora_init", "pissa")).lower()

        for _ in range(int(self.args.num_ent_layers)):
            r = int(self.args.ent_r_list[_])
            new_ent_embeddings = loralib.Embedding(self.lora_ent_len, self.args.emb_dim, r).to(self.args.device).double()
            if init_method == "xavier":
                xavier_normal_(new_ent_embeddings.weight)
            else:
                with torch.no_grad():
                    temp_w = torch.randn(self.lora_ent_len, self.args.emb_dim, device=self.args.device, dtype=torch.float32) * ref_std + ref_mean
                    U, S, Vh = torch.linalg.svd(temp_w, full_matrices=False)
                    effective_r = min(r, Vh.size(0), U.size(1))
                    lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                    lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                    if lora_a_param is None or lora_b_param is None:
                        lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                    if lora_a_param is not None and lora_b_param is not None:
                        lora_a = cast(torch.Tensor, lora_a_param.data)
                        lora_b = cast(torch.Tensor, lora_b_param.data)
                        us = U[:, :effective_r] @ torch.diag(S[:effective_r])
                        vh = Vh[:effective_r, :]
                        if lora_a.shape[1] == self.args.emb_dim and lora_b.shape[0] == self.lora_ent_len:
                            # Layout-1: A[r, emb_dim], B[num_ent, r]
                            lora_a[:effective_r, :] = vh.clone()
                            lora_b[:, :effective_r] = us.clone()
                        elif lora_a.shape[1] == self.lora_ent_len and lora_b.shape[0] == self.args.emb_dim:
                            # Layout-2: A[r, num_ent], B[emb_dim, r]
                            lora_a[:effective_r, :] = us.T.clone()
                            lora_b[:, :effective_r] = vh.T.clone()
                        else:
                            xavier_normal_(new_ent_embeddings.weight)
                        new_ent_embeddings.weight.data.zero_()
                    else:
                        xavier_normal_(new_ent_embeddings.weight)
            lora_ent_embeddings_list.append(deepcopy(new_ent_embeddings))

        new_rel_embeddings = loralib.Embedding(new_rel_embeddings_len, self.args.emb_dim, int(self.args.rel_r)).to(self.args.device).double()
        if init_method == "xavier":
            xavier_normal_(new_rel_embeddings.weight)
        else:
            with torch.no_grad():
                rel_temp_w = torch.randn(new_rel_embeddings_len, self.args.emb_dim, device=self.args.device, dtype=torch.float32) * ref_std + ref_mean
                U, S, Vh = torch.linalg.svd(rel_temp_w, full_matrices=False)
                rel_r = min(int(self.args.rel_r), Vh.size(0), U.size(1))
                rel_a_param = getattr(new_rel_embeddings, "lora_embedding_A", None)
                rel_b_param = getattr(new_rel_embeddings, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(new_rel_embeddings, "lora_A", None)
                    rel_b_param = getattr(new_rel_embeddings, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    rel_a = cast(torch.Tensor, rel_a_param.data)
                    rel_b = cast(torch.Tensor, rel_b_param.data)
                    rel_us = U[:, :rel_r] @ torch.diag(S[:rel_r])
                    rel_vh = Vh[:rel_r, :]
                    if rel_a.shape[1] == self.args.emb_dim and rel_b.shape[0] == new_rel_embeddings_len:
                        rel_a[:rel_r, :] = rel_vh.clone()
                        rel_b[:, :rel_r] = rel_us.clone()
                    elif rel_a.shape[1] == new_rel_embeddings_len and rel_b.shape[0] == self.args.emb_dim:
                        rel_a[:rel_r, :] = rel_us.T.clone()
                        rel_b[:, :rel_r] = rel_vh.T.clone()
                    else:
                        xavier_normal_(new_rel_embeddings.weight)
                    new_rel_embeddings.weight.data.zero_()
                else:
                    xavier_normal_(new_rel_embeddings.weight)

        return deepcopy(lora_ent_embeddings_list), deepcopy(new_rel_embeddings)

    def switch_snapshot(self):
        if self.lora_ent_embeddings_list is not None:
            assert self.lora_rel_embeddings is not None
            new_ent_embeddings = self.ent_embeddings.weight.data
            new_rel_embeddings = self.rel_embeddings.weight.data
            for lora_id in range(int(self.args.num_ent_layers) - 1):
                start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + lora_id * self.lora_ent_len
                new_ent_embeddings[start_id: start_id + self.lora_ent_len] = Parameter(self.lora_ent_embeddings_list[lora_id].forward(torch.arange(int(self.lora_ent_len)).to(self.args.device)))
            last_start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + (int(self.args.num_ent_layers) - 1) * self.lora_ent_len
            last_lora_id = int(self.args.num_ent_layers) - 1
            new_ent_embeddings[last_start_id:] = Parameter(self.lora_ent_embeddings_list[last_lora_id].forward(torch.arange(len(new_ent_embeddings[last_start_id:])).to(self.args.device)))
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + cast(list[int], self.new_ordered_entities)
            assert len(new_ent_embeddings) == len(ent_indices)
            new_ent_embeddings = new_ent_embeddings[ent_indices]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot - 1].num_rel:] = Parameter(deepcopy(self.lora_rel_embeddings.forward(torch.arange(len(self.lora_rel_embeddings.weight)).to(self.args.device))))
            self.ent_embeddings.weight = Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        self.store_old_parameters()
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(
            self.ent_embeddings.weight.data
        )
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(
            self.rel_embeddings.weight.data
        )
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        self.ent_embeddings.weight.requires_grad = False
        self.rel_embeddings.weight.requires_grad = False
        self.lora_ent_embeddings_list_tmp, self.lora_rel_embeddings = self.expand_lora_embeddings()
        self.lora_ent_embeddings_list = nn.ModuleList(self.lora_ent_embeddings_list_tmp)

    # [提速修改 4：新增 LoRA+ 优化器生成方法]
    def get_lora_plus_optimizer(self, base_lr=1e-3, loraplus_ratio=16.0, weight_decay=0.0):
        param_A_and_others = []
        param_B = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_B' in name or 'lora_embedding_B' in name:
                param_B.append(param)
            else:
                param_A_and_others.append(param)
                
        optimizer = torch.optim.Adam([
            {'params': param_A_and_others, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': param_B, 'lr': base_lr * loraplus_ratio, 'weight_decay': weight_decay}
        ])
        return optimizer


class TransE(LoraKGE_Layers):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label) / head.size(0)

    def score_fun(self, h, r, t):
        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t = self.norm_ent(t)
        return torch.norm(h + r - t, 1, -1)

    def split_pn_score(self, score, label):
        p_score = score[torch.where(label > 0)]
        n_score = (score[torch.where(label < 0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def get_lora_embeddings(self):
        assert self.lora_ent_embeddings_list is not None
        assert self.lora_rel_embeddings is not None
        lora_ent_embeddings = self.lora_ent_embeddings_list[0].forward(torch.arange(int(self.lora_ent_len)).to(self.args.device))
        for lora_id in range(1, int(self.args.num_ent_layers)):
            lora_ent_embeddings = torch.cat((lora_ent_embeddings, self.lora_ent_embeddings_list[lora_id].forward(torch.arange(int(self.lora_ent_len)).to(self.args.device))), dim=0)
        lora_rel_embeddings = self.lora_rel_embeddings.forward(torch.arange(len(self.lora_rel_embeddings.weight)).to(self.args.device))
        return lora_ent_embeddings, lora_rel_embeddings

    def embedding(self, stage=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.snapshot == 0:
            ent_embeddings = self.ent_embeddings.weight
            rel_embeddings = self.rel_embeddings.weight
        else:
            ent_embeddings = cast(torch.Tensor, self.old_data_ent_embeddings_weight)
            rel_embeddings = cast(torch.Tensor, self.old_data_rel_embeddings_weight)
        return ent_embeddings, rel_embeddings

    def predict(self, head, relation, stage='Valid'):
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot_valid].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, relation)
            t_all = ent_embeddings[:num_ent]
        else:
            ent_embeddings, rel_embeddings = self.embedding(stage)
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + cast(list[int], self.new_ordered_entities)
            all_ent_embeddings = all_ent_embeddings[ent_indices]
            h = torch.index_select(all_ent_embeddings, 0, head)
            r = torch.index_select(all_rel_embeddings, 0, relation)
            t_all = all_ent_embeddings[:num_ent]

        h = self.norm_ent(h)
        r = self.norm_rel(r)
        t_all = self.norm_ent(t_all)

        pred_t = h + r
        chunk_size = 2048
        score_chunks = []
        for start in range(0, t_all.size(0), chunk_size):
            end = min(start + chunk_size, t_all.size(0))
            t_chunk = t_all[start:end]
            chunk_score = 9.0 - torch.norm(pred_t.unsqueeze(1) - t_chunk, p=1, dim=2)
            score_chunks.append(torch.sigmoid(chunk_score))
        return torch.cat(score_chunks, dim=1)

    def margin_loss(self, head, rel, tail, label=None):
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, rel)
            t = torch.index_select(ent_embeddings, 0, tail)
        else:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + cast(list[int], self.new_ordered_entities)
            all_ent_embeddings = all_ent_embeddings[ent_indices]
            h = torch.index_select(all_ent_embeddings, 0, head)
            r = torch.index_select(all_rel_embeddings, 0, rel)
            t = torch.index_select(all_ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.get_TransE_loss(head, relation, tail, label)
        return loss