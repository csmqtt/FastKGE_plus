import math
from typing import cast

import torch.nn.functional as F
from torch.profiler import record_function

from .BaseModel import *


def _init_nonempty(fn, tensor, *args, **kwargs):
    """Call a torch.nn.init-style initializer only when the tensor has elements.

    Avoids the `Initializing zero-element tensors is a no-op` UserWarning that
    triggers whenever a snapshot adds new entities but no new relations (or
    vice versa), leaving a (0, emb_dim) / (r, 0) shaped parameter.
    """
    if tensor is None or tensor.numel() == 0:
        return tensor
    return fn(tensor, *args, **kwargs)


class DoRALoraEmbedding(loralib.Embedding):
    """
    DoRA-inspired adapter: keep frozen base embedding + standard LoRA (B @ A),
    multiply the low-rank delta by a learnable per-dimension scale lora_g (shape emb_dim).
    Named lora_g so loralib.lora_state_dict() keeps it in checkpoints.
    merge_weights=False: merged weight path cannot represent g-scaled deltas.
    Optional use_rs_lora: replace loralib's alpha/r scaling with alpha/sqrt(r) (RS-LoRA), DoRA path only.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = False,
        use_rs_lora: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            r,
            lora_alpha=lora_alpha,
            merge_weights=merge_weights,
            **kwargs,
        )
        if r > 0 and use_rs_lora:
            self.scaling = self.lora_alpha / math.sqrt(float(self.r))
        if r > 0:
            self.lora_g = nn.Parameter(
                torch.ones(embedding_dim, dtype=self.weight.dtype, device=self.weight.device)
            )
        else:
            self.register_parameter("lora_g", None)

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            delta = (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            result = result + delta * self.lora_g
            return result
        return nn.Embedding.forward(self, x)


class LoraKGE_Layers(BaseModel):
    def __init__(self, args, kg) -> None:
        super(LoraKGE_Layers, self).__init__(args, kg)
        self.lora_ent_embeddings_list = None
        # P0 Graph Layering (rel-side): always a ModuleList. When num_rel_layers==1
        # the list has a single module and the pipeline is numerically identical
        # to the pre-P0 single-module path.
        self.lora_rel_embeddings_list = None
        self.lora_rel_len = 0
        self.new_rel_embeddings_len = 0
        # P1/P2 efficiency caches: precomputed once per snapshot transition and
        # reused across every batch of that snapshot. Without these, margin_loss
        # / predict / get_lora_embeddings were rebuilding the same tensors per
        # batch (torch.arange(N).to(device), list(range(old_num_ent)) +
        # new_ordered_entities, python-list fancy index).
        self._ent_arange_cache = None     # torch.arange(lora_ent_len) on device
        self._rel_arange_list = None      # list of torch.arange(rows) per rel layer
        self._ent_canonical_perm = None   # maps canonical-new-ent-idx -> lora row

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def expand_embedding_size(self):
        _dtype = model_dtype(self.args)
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(self.args.device, dtype=_dtype)
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(self.args.device, dtype=_dtype)
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return ent_embeddings, rel_embeddings

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
            if init_method == "dora":
                new_ent_embeddings = DoRALoraEmbedding(
                    self.lora_ent_len,
                    self.args.emb_dim,
                    r,
                    merge_weights=False,
                    use_rs_lora=bool(getattr(self.args, "use_rs_lora", False)),
                ).to(self.args.device, dtype=model_dtype(self.args))
                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
            else:
                new_ent_embeddings = loralib.Embedding(self.lora_ent_len, self.args.emb_dim, r).to(self.args.device, dtype=model_dtype(self.args))
                if init_method == "xavier":
                    _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                elif init_method == "zero_b":
                    # Canonical LoRA init: A ~ kaiming, B = 0 → delta = 0 at step 0,
                    # and grad_B ∝ A is immediately non-zero so LoRA+ (lr_B > lr_A) is meaningful.
                    _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                    with torch.no_grad():
                        lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                        if lora_a_param is None or lora_b_param is None:
                            lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                            lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                        if lora_a_param is not None and lora_b_param is not None:
                            _init_nonempty(
                                nn.init.kaiming_uniform_,
                                cast(torch.Tensor, lora_a_param.data),
                                a=math.sqrt(5),
                            )
                            _init_nonempty(nn.init.zeros_, cast(torch.Tensor, lora_b_param.data))
                elif init_method == "qr":
                    with torch.no_grad():
                        effective_r = min(r, self.lora_ent_len, self.args.emb_dim)
                        left_rand = torch.randn(self.lora_ent_len, effective_r, device=self.args.device, dtype=torch.float32)
                        right_rand = torch.randn(self.args.emb_dim, effective_r, device=self.args.device, dtype=torch.float32)
                        left_q, _ = torch.linalg.qr(left_rand, mode='reduced')
                        right_q, _ = torch.linalg.qr(right_rand, mode='reduced')
                        us = left_q[:, :effective_r] * ref_std
                        vh = right_q[:, :effective_r].T
                        lora_a_param = getattr(new_ent_embeddings, "lora_embedding_A", None)
                        lora_b_param = getattr(new_ent_embeddings, "lora_embedding_B", None)
                        if lora_a_param is None or lora_b_param is None:
                            lora_a_param = getattr(new_ent_embeddings, "lora_A", None)
                            lora_b_param = getattr(new_ent_embeddings, "lora_B", None)
                        if lora_a_param is not None and lora_b_param is not None:
                            lora_a = cast(torch.Tensor, lora_a_param.data)
                            lora_b = cast(torch.Tensor, lora_b_param.data)
                            if lora_a.shape[1] == self.args.emb_dim and lora_b.shape[0] == self.lora_ent_len:
                                lora_a[:effective_r, :] = vh.clone()
                                lora_b[:, :effective_r] = us.clone()
                            elif lora_a.shape[1] == self.lora_ent_len and lora_b.shape[0] == self.args.emb_dim:
                                lora_a[:effective_r, :] = us.T.clone()
                                lora_b[:, :effective_r] = vh.T.clone()
                            else:
                                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                            new_ent_embeddings.weight.data.zero_()
                        else:
                            _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
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
                                lora_a[:effective_r, :] = vh.clone()
                                lora_b[:, :effective_r] = us.clone()
                            elif lora_a.shape[1] == self.lora_ent_len and lora_b.shape[0] == self.args.emb_dim:
                                lora_a[:effective_r, :] = us.T.clone()
                                lora_b[:, :effective_r] = vh.T.clone()
                            else:
                                _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
                            new_ent_embeddings.weight.data.zero_()
                        else:
                            _init_nonempty(xavier_normal_, new_ent_embeddings.weight)
            lora_ent_embeddings_list.append(new_ent_embeddings)

        # ---- Relation side: P0 Graph Layering --------------------------------
        # Split new relations into `num_rel_layers` chunks in canonical insertion
        # order; each chunk owns its own LoRA adapter. When num_rel_layers == 1
        # this loop produces a single module of shape (new_rel_len, emb_dim, rel_r)
        # which is bit-identical to the pre-P0 path (same rows, same rank, same
        # RNG sequence for init).
        num_rel_layers = max(1, int(self.args.num_rel_layers))
        self.lora_rel_len = (new_rel_embeddings_len + num_rel_layers - 1) // num_rel_layers
        self.new_rel_embeddings_len = new_rel_embeddings_len
        # Per-layer rank. Uniform split mirrors the ent-side default path when
        # using_various_ranks is off; each layer gets rel_r // num_rel_layers rank.
        # For num_rel_layers==1 this is just [rel_r], matching current behavior.
        base_rel_r = int(self.args.rel_r)
        if num_rel_layers == 1:
            rel_r_list = [base_rel_r]
        else:
            rel_r_list = [max(1, base_rel_r // num_rel_layers)] * num_rel_layers
        self.args.rel_r_list = rel_r_list

        lora_rel_embeddings_list = []
        for layer_i in range(num_rel_layers):
            layer_rows = self.lora_rel_len if num_rel_layers > 1 else new_rel_embeddings_len
            layer_r = int(rel_r_list[layer_i])
            lora_rel_embeddings_list.append(
                self._build_rel_lora_module(layer_rows, layer_r, init_method, ref_mean, ref_std)
            )

        return list(lora_ent_embeddings_list), lora_rel_embeddings_list

    def _build_rel_lora_module(self, rows: int, r: int, init_method: str, ref_mean, ref_std):
        """Create one relation-side LoRA adapter of shape (rows, emb_dim, r) and
        initialize it with the requested scheme.

        Extracted from the original inline block in expand_lora_embeddings so the
        layered (num_rel_layers>1) path can construct each layer independently.
        For num_rel_layers==1 with rows=new_rel_embeddings_len and r=rel_r this is
        numerically identical to the pre-P0 initialization sequence.
        """
        if init_method == "dora":
            mod = DoRALoraEmbedding(
                rows,
                self.args.emb_dim,
                r,
                merge_weights=False,
                use_rs_lora=bool(getattr(self.args, "use_rs_lora", False)),
            ).to(self.args.device, dtype=model_dtype(self.args))
            _init_nonempty(xavier_normal_, mod.weight)
            return mod

        mod = loralib.Embedding(rows, self.args.emb_dim, r).to(self.args.device, dtype=model_dtype(self.args))
        if init_method == "xavier":
            _init_nonempty(xavier_normal_, mod.weight)
        elif init_method == "zero_b":
            _init_nonempty(xavier_normal_, mod.weight)
            with torch.no_grad():
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    _init_nonempty(
                        nn.init.kaiming_uniform_,
                        cast(torch.Tensor, rel_a_param.data),
                        a=math.sqrt(5),
                    )
                    _init_nonempty(nn.init.zeros_, cast(torch.Tensor, rel_b_param.data))
        elif init_method == "qr":
            with torch.no_grad():
                eff_r = min(r, rows, self.args.emb_dim)
                left_rand = torch.randn(rows, eff_r, device=self.args.device, dtype=torch.float32)
                right_rand = torch.randn(self.args.emb_dim, eff_r, device=self.args.device, dtype=torch.float32)
                left_q, _ = torch.linalg.qr(left_rand, mode='reduced')
                right_q, _ = torch.linalg.qr(right_rand, mode='reduced')
                rel_us = left_q[:, :eff_r] * ref_std
                rel_vh = right_q[:, :eff_r].T
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    rel_a = cast(torch.Tensor, rel_a_param.data)
                    rel_b = cast(torch.Tensor, rel_b_param.data)
                    if rel_a.shape[1] == self.args.emb_dim and rel_b.shape[0] == rows:
                        rel_a[:eff_r, :] = rel_vh.clone()
                        rel_b[:, :eff_r] = rel_us.clone()
                    elif rel_a.shape[1] == rows and rel_b.shape[0] == self.args.emb_dim:
                        rel_a[:eff_r, :] = rel_us.T.clone()
                        rel_b[:, :eff_r] = rel_vh.T.clone()
                    else:
                        _init_nonempty(xavier_normal_, mod.weight)
                    mod.weight.data.zero_()
                else:
                    _init_nonempty(xavier_normal_, mod.weight)
        else:
            with torch.no_grad():
                rel_temp_w = torch.randn(rows, self.args.emb_dim, device=self.args.device, dtype=torch.float32) * ref_std + ref_mean
                U, S, Vh = torch.linalg.svd(rel_temp_w, full_matrices=False)
                eff_r = min(r, Vh.size(0), U.size(1))
                rel_a_param = getattr(mod, "lora_embedding_A", None)
                rel_b_param = getattr(mod, "lora_embedding_B", None)
                if rel_a_param is None or rel_b_param is None:
                    rel_a_param = getattr(mod, "lora_A", None)
                    rel_b_param = getattr(mod, "lora_B", None)
                if rel_a_param is not None and rel_b_param is not None:
                    rel_a = cast(torch.Tensor, rel_a_param.data)
                    rel_b = cast(torch.Tensor, rel_b_param.data)
                    rel_us = U[:, :eff_r] @ torch.diag(S[:eff_r])
                    rel_vh = Vh[:eff_r, :]
                    if rel_a.shape[1] == self.args.emb_dim and rel_b.shape[0] == rows:
                        rel_a[:eff_r, :] = rel_vh.clone()
                        rel_b[:, :eff_r] = rel_us.clone()
                    elif rel_a.shape[1] == rows and rel_b.shape[0] == self.args.emb_dim:
                        rel_a[:eff_r, :] = rel_us.T.clone()
                        rel_b[:, :eff_r] = rel_vh.T.clone()
                    else:
                        _init_nonempty(xavier_normal_, mod.weight)
                    mod.weight.data.zero_()
                else:
                    _init_nonempty(xavier_normal_, mod.weight)
        return mod

    def switch_snapshot(self):
        if self.lora_ent_embeddings_list is not None:
            assert self.lora_rel_embeddings_list is not None
            new_ent_embeddings = self.ent_embeddings.weight.data
            new_rel_embeddings = self.rel_embeddings.weight.data
            for lora_id in range(int(self.args.num_ent_layers) - 1):
                start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + lora_id * self.lora_ent_len
                new_ent_embeddings[start_id: start_id + self.lora_ent_len] = Parameter(
                    self.lora_ent_embeddings_list[lora_id](torch.arange(int(self.lora_ent_len)).to(self.args.device)).detach().clone()
                )
            last_start_id = self.kg.snapshots[self.args.snapshot - 1].num_ent + (int(self.args.num_ent_layers) - 1) * self.lora_ent_len
            last_lora_id = int(self.args.num_ent_layers) - 1
            new_ent_embeddings[last_start_id:] = Parameter(
                self.lora_ent_embeddings_list[last_lora_id](torch.arange(len(new_ent_embeddings[last_start_id:])).to(self.args.device)).detach().clone()
            )
            ent_indices = list(range(self.kg.snapshots[self.args.snapshot - 1].num_ent)) + cast(list[int], self.new_ordered_entities)
            assert len(new_ent_embeddings) == len(ent_indices)
            new_ent_embeddings = new_ent_embeddings[ent_indices]
            # Layered rel writeback: concatenate each layer's output in canonical
            # order and trim phantom rows from the last layer (when
            # new_rel_len % num_rel_layers != 0). For num_rel_layers==1 this
            # collapses to the original single-module path.
            num_rel_layers = len(self.lora_rel_embeddings_list)
            rel_parts = []
            for i in range(num_rel_layers):
                layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[i])
                rows = int(cast(torch.Tensor, layer_mod.weight).shape[0])  # type: ignore[attr-defined]
                rel_parts.append(
                    layer_mod(torch.arange(rows).to(self.args.device)).detach().clone()
                )
            rel_concat = torch.cat(rel_parts, dim=0)[: self.new_rel_embeddings_len]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot - 1].num_rel:] = Parameter(rel_concat)
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
        lora_ent_list_tmp, lora_rel_list_tmp = self.expand_lora_embeddings()
        self.lora_ent_embeddings_list = nn.ModuleList(lora_ent_list_tmp)
        self.lora_rel_embeddings_list = nn.ModuleList(lora_rel_list_tmp)
        self._rebuild_lora_cache()

    def _rebuild_lora_cache(self):
        """Precompute per-snapshot tensors used in the hot margin_loss /
        predict / get_lora_embeddings path.

        Before this cache, every batch rebuilt three things that are actually
        constant throughout a snapshot:
          1. `torch.arange(self.lora_ent_len).to(device)`  (one per ent layer)
          2. `list(range(old_num_ent)) + self.new_ordered_entities`  (Python
             list of size ~70k on HYBRID S3)
          3. `all_ent[ent_indices]` with that Python list as fancy index
             (forces a host->device index transfer each batch)

        Observation: `ent_indices[:old_num_ent]` is identity on the old /
        frozen part, so the permutation truly only acts on the new-entity
        section. We collapse it into a small `new_num_ent`-sized long tensor
        `_ent_canonical_perm` applied directly to `lora_ent_embeddings` inside
        `get_lora_embeddings` -- margin_loss can then just `torch.cat` + gather
        with no Python hop per batch.
        """
        if self.lora_ent_embeddings_list is None or self.lora_rel_embeddings_list is None:
            self._ent_arange_cache = None
            self._rel_arange_list = None
            self._ent_canonical_perm = None
            return

        device = self.args.device

        # (1) entity-side arange: all ent layers share the same lora_ent_len
        # (last layer may hold padding rows that are trimmed via the canonical
        # perm below, so this arange covers the full padded length).
        ent_len = int(self.lora_ent_len)
        self._ent_arange_cache = torch.arange(ent_len, device=device, dtype=torch.long)

        # (2) relation-side arange per layer: last layer may have fewer rows
        # when new_rel_len % num_rel_layers != 0.
        rel_arange_list = []
        for i in range(len(self.lora_rel_embeddings_list)):
            layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[i])
            rows = int(cast(torch.Tensor, layer_mod.weight).shape[0])  # type: ignore[attr-defined]
            rel_arange_list.append(torch.arange(rows, device=device, dtype=torch.long))
        self._rel_arange_list = rel_arange_list

        # (3) canonical-order permutation for new entities. switch_snapshot is
        # called at end of snap s iteration, so args.snapshot = s here and
        # `self.new_ordered_entities` holds global ids in
        # [num_ent_s, num_ent_{s+1}) sorted by (distance, centrality). At
        # training time (snap s+1) margin_loss does the equivalent of
        # `all_ent[new_ordered_entities[j]]` i.e. row
        # `(new_ordered_entities[j] - num_ent_s)` of the lora layer output.
        old_num_ent = int(self.kg.snapshots[int(self.args.snapshot)].num_ent)
        perm = [int(gid) - old_num_ent for gid in self.new_ordered_entities]
        self._ent_canonical_perm = torch.tensor(perm, device=device, dtype=torch.long)

    # LoRA+ optimizer: separate lr for A / B, independent weight decay per group,
    # and keep DoRA scale parameter lora_g in a dedicated group (no weight decay).
    def get_lora_plus_optimizer(
        self,
        base_lr=1e-3,
        loraplus_ratio=16.0,
        weight_decay=0.0,
        lora_wd_a=0.0,
        lora_wd_b=0.0,
    ):
        param_A = []
        param_B = []
        param_g = []
        param_other = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_B' in name or 'lora_embedding_B' in name:
                param_B.append(param)
            elif 'lora_A' in name or 'lora_embedding_A' in name:
                param_A.append(param)
            elif name.endswith('lora_g') or '.lora_g' in name:
                param_g.append(param)
            else:
                param_other.append(param)

        param_groups = []
        if param_A:
            param_groups.append({'params': param_A, 'lr': base_lr, 'weight_decay': lora_wd_a})
        if param_B:
            param_groups.append({'params': param_B, 'lr': base_lr * loraplus_ratio, 'weight_decay': lora_wd_b})
        if param_g:
            param_groups.append({'params': param_g, 'lr': base_lr, 'weight_decay': 0.0})
        if param_other:
            param_groups.append({'params': param_other, 'lr': base_lr, 'weight_decay': weight_decay})

        if not param_groups:
            # Fallback: ensure the optimizer has at least one (possibly empty) group
            # so callers can still call .step(). AdamW requires >=1 group.
            param_groups.append({'params': [], 'lr': base_lr, 'weight_decay': weight_decay})

        # AdamW with wd=0 is numerically equivalent to Adam; with wd>0 we get
        # decoupled weight decay (better behaved for LoRA B norm growth).
        optimizer = torch.optim.AdamW(param_groups)
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
        """Forward each LoRA layer on its cached arange tensor and return the
        per-entity / per-relation deltas already re-ordered to match the
        canonical global-id layout used by the rest of the pipeline.

        Ent output shape: (new_ent_embeddings_len, emb_dim). Row j corresponds
        to the j-th new entity in canonical order (i.e. global id
        old_num_ent + j). Rel output shape: (new_rel_embeddings_len, emb_dim),
        already in canonical order since rel has no centrality sort.
        """
        assert self.lora_ent_embeddings_list is not None
        assert self.lora_rel_embeddings_list is not None
        assert self._ent_arange_cache is not None, "LoRA cache not built; did switch_snapshot run?"
        assert self._rel_arange_list is not None
        assert self._ent_canonical_perm is not None

        with record_function("lora/ent_layers_forward"):
            num_ent_layers = int(self.args.num_ent_layers)
            arange_ent = self._ent_arange_cache
            if num_ent_layers == 1:
                lora_ent_concat = self.lora_ent_embeddings_list[0].forward(arange_ent)
            else:
                ent_parts = [
                    cast(nn.Module, self.lora_ent_embeddings_list[i]).forward(arange_ent)
                    for i in range(num_ent_layers)
                ]
                lora_ent_concat = torch.cat(ent_parts, dim=0)
            # Canonical reordering (drops phantom rows from the last padded
            # layer as a side effect since perm only references valid rows).
            lora_ent_embeddings = lora_ent_concat.index_select(0, self._ent_canonical_perm)

        with record_function("lora/rel_layers_forward"):
            num_rel_layers = len(self.lora_rel_embeddings_list)
            if num_rel_layers == 1:
                layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[0])
                lora_rel_embeddings = layer_mod(self._rel_arange_list[0])
            else:
                rel_parts = []
                for i in range(num_rel_layers):
                    layer_mod = cast(nn.Module, self.lora_rel_embeddings_list[i])
                    rel_parts.append(layer_mod(self._rel_arange_list[i]))
                lora_rel_embeddings = torch.cat(rel_parts, dim=0)[: self.new_rel_embeddings_len]
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
            # lora_* already in canonical order; see get_lora_embeddings.
            all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
            all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
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

    def _lookup_ent_with_lora(
        self,
        idx: torch.Tensor,
        ent_old: torch.Tensor,
        lora_new: torch.Tensor,
    ) -> torch.Tensor:
        """P2 sparse-gather: return `torch.cat([ent_old, lora_new])[idx]` without
        ever materializing the (old_num+new_num, D) concat.

        On HYBRID S3 (old=70k, new=~1000, D=200) the cat was ~56 MB per batch;
        a batch only touches batch_size * (1+neg_ratio) ~ 11k rows, so >99% of
        the cat was wasted. We gather from ent_old and lora_new independently
        and merge with torch.where on a bool mask.

        Correctness: for any row i with idx[i] < old_num the returned row equals
        ent_old[idx[i]] (is_new == False branch); for idx[i] >= old_num it
        equals lora_new[idx[i] - old_num] (is_new == True branch). The clamps
        keep the "wrong side" gather on an in-range index so it doesn't fault;
        torch.where then discards that garbage row.
        """
        old_num = ent_old.size(0)
        is_new = idx >= old_num
        safe_old_idx = torch.clamp(idx, max=old_num - 1)
        safe_new_idx = torch.clamp(idx - old_num, min=0)
        h_old = ent_old.index_select(0, safe_old_idx)
        h_new = lora_new.index_select(0, safe_new_idx)
        return torch.where(is_new.unsqueeze(-1), h_new, h_old)

    def margin_loss(self, head, rel, tail, label=None):
        if self.args.snapshot == 0:
            ent_embeddings, rel_embeddings = self.embedding('Train')
            h = torch.index_select(ent_embeddings, 0, head)
            r = torch.index_select(rel_embeddings, 0, rel)
            t = torch.index_select(ent_embeddings, 0, tail)
        else:
            with record_function("lora/embedding_base"):
                ent_embeddings, rel_embeddings = self.embedding('Train')
            with record_function("lora/get_lora_embeddings"):
                lora_ent_embeddings, lora_rel_embeddings = self.get_lora_embeddings()
            # lora_ent_embeddings is already in canonical new-entity order
            # (see get_lora_embeddings). Relation table is small (a few hundred
            # rows) so the cat on the relation side is cheap -- we only skip
            # the entity-side cat which is the one that costs memory.
            use_sparse = str(getattr(self.args, "lora_sparse_gather", "on")) == "on"
            if use_sparse:
                with record_function("lora/gather_ent_mixed"):
                    h = self._lookup_ent_with_lora(head, ent_embeddings, lora_ent_embeddings)
                    t = self._lookup_ent_with_lora(tail, ent_embeddings, lora_ent_embeddings)
                with record_function("lora/gather_rel"):
                    all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
                    r = torch.index_select(all_rel_embeddings, 0, rel)
            else:
                with record_function("lora/assemble_full_embeddings"):
                    all_ent_embeddings = torch.cat([ent_embeddings, lora_ent_embeddings], dim=0)
                    all_rel_embeddings = torch.cat([rel_embeddings, lora_rel_embeddings], dim=0)
                with record_function("lora/index_select"):
                    h = torch.index_select(all_ent_embeddings, 0, head)
                    r = torch.index_select(all_rel_embeddings, 0, rel)
                    t = torch.index_select(all_ent_embeddings, 0, tail)
        with record_function("lora/score_fun"):
            score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y)

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def loss(self, head, relation, tail=None, label=None):
        loss = self.get_TransE_loss(head, relation, tail, label)
        return loss