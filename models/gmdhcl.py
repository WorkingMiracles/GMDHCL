# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from models.graph_transformer import GraphTransformerLayer


class GMDHCL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GMDHCL, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.cl_layer = config['cl_layer']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']

        self.eps = config['eps']
        self.depth_cl = config['depth_cl']
        self.gama = config['gama']
        self.beta = config['beta']
        self.tau = 0.2
        self.n_nodes = self.n_users + self.n_items

        # ============================================================
        # Geometric fusion and hierarchical contrastive module switches
        # ============================================================
        self.use_gmd = config['use_gmd'] if config['use_gmd'] is not None else False
        self.use_hcl = config['use_hcl'] if config['use_hcl'] is not None else False
        self.use_saher = config['use_saher'] if config['use_saher'] is not None else False
        self.use_multiscale_perturb = False

        # ---- GMD Ablation Switches ----

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()

        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)

        # graph transformer
        self.graph_transformer = GraphTransformerLayer(config).to(self.device)
        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))

        # ============================================================
        # GMD: geometric modality decomposition and reliability weighting
        # ============================================================
        if self.use_gmd:
            self.gmd_tau_u = config['gmd_tau_u']
            self.gmd_drop_rate = config['gmd_drop_rate']
            self.gmd_align_weight = config['gmd_align_weight']
            self.synergy_weight = config['synergy_weight']

            # Precompute item degrees for cold-warm continuous weighting
            item_degrees = torch.tensor(
                np.array(self.interaction_matrix.sum(axis=0))
            ).flatten().float()
            item_degrees = torch.log(item_degrees + 1.0)
            item_degrees = item_degrees / (item_degrees.max() + 1e-7)
            self.register_buffer('item_degree_norm', item_degrees)

            # --- Direction 1: Spectral-Guided Perturbation ---
            self.use_spectral_perturb = config['use_spectral_perturb'] if config['use_spectral_perturb'] is not None else False
            if self.use_spectral_perturb:
                self.spectral_update_interval = config['spectral_update_interval']
                self.spectral_p_min = config['spectral_p_min']
                self.register_buffer('svd_v', torch.zeros(self.feat_embed_dim))
                self.register_buffer('svd_t', torch.zeros(self.feat_embed_dim))
                self._svd_update_counter = 0

            # --- Direction 2: Adaptive Temperature ---
            self.use_optimal_tau = config['use_optimal_tau'] if config['use_optimal_tau'] is not None else False
            if self.use_optimal_tau:
                self.gmd_tau_base = config['gmd_tau_base']
                self.gmd_tau_scale = config['gmd_tau_scale']

            # --- Direction 3: Multi-Scale Perturbation ---
            self.use_multiscale_perturb = config['use_multiscale_perturb'] if config['use_multiscale_perturb'] is not None else False
            if self.use_multiscale_perturb:
                self.ms_alpha = config['ms_alpha']
                self.ms_beta = config['ms_beta']
                self.ms_gamma = config['ms_gamma']

        # ============================================================
        # BPR-EMA-driven outer scaling of the hierarchical losses
        # ============================================================
        if self.use_hcl:
            self.hcl_ema_beta = config['hcl_ema_beta']
            self.hcl_scale_max = config['hcl_scale_max']
            self.register_buffer('bpr_ema', torch.tensor(0.0))
            self.register_buffer('bpr_init', torch.tensor(0.0))
            self.register_buffer('bpr_init_stored', torch.tensor(0.0))

        # ============================================================
        # Spectral Anti-Collapse Regularization
        # ============================================================
        self.use_rank_reg = config['use_rank_reg'] if config['use_rank_reg'] is not None else False
        if self.use_rank_reg:
            self.rank_reg_weight = config['rank_reg_weight']

        # ============================================================
        # HCL: Multi-Level Hierarchical Contrastive Learning
        # ============================================================
        if self.use_hcl:
            self.hcl_l1_weight = config['hcl_l1_weight']
            self.hcl_l3_weight = config['hcl_l3_weight']
            self.hcl_hard_neg_k = config['hcl_hard_neg_k']
            self.hcl_use_adaptive_weight = config['hcl_use_adaptive_weight']

            if self.hcl_use_adaptive_weight:
                # Kendall uncertainty weighting: log(sigma^2) for each level
                self.log_sigma_l1 = nn.Parameter(torch.zeros(1))
                self.log_sigma_l2 = nn.Parameter(torch.zeros(1))
                self.log_sigma_l3 = nn.Parameter(torch.zeros(1))

        # ============================================================
        # SAHER: Structure-Aware Hypergraph with Entropy Regularization
        # ============================================================
        if self.use_saher:
            self.saher_gamma = config['saher_gamma']
            self.saher_entropy_weight = config['saher_entropy_weight']
            self.saher_diversity_weight = config['saher_diversity_weight']

            # Structure-aware hyperedge projection parameters
            if self.v_feat is not None:
                self.v_hyper_struct = nn.Parameter(
                    nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.hyper_num)))
            if self.t_feat is not None:
                self.t_hyper_struct = nn.Parameter(
                    nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.hyper_num)))

            # Precompute item-user block of norm_adj (sparse slicing unsupported on CUDA)
            indices = self.norm_adj._indices()
            values = self.norm_adj._values()
            mask = (indices[0] >= self.n_users) & (indices[1] < self.n_users)
            iu_rows = indices[0, mask] - self.n_users
            iu_cols = indices[1, mask]
            iu_vals = values[mask]
            self.norm_adj_iu = torch.sparse_coo_tensor(
                torch.stack([iu_rows, iu_cols]), iu_vals,
                (self.n_items, self.n_users)
            ).coalesce()

    def update_spectral_weights(self):
        """Update SVD-based importance weights for spectral perturbation."""
        with torch.no_grad():
            if self.v_feat is not None and self._svd_update_counter % self.spectral_update_interval == 0:
                _, S_v, _ = torch.linalg.svd(self.item_image_trs, full_matrices=False)
                self.svd_v = S_v / (S_v.sum() + 1e-7)
            if self.t_feat is not None and self._svd_update_counter % self.spectral_update_interval == 0:
                _, S_t, _ = torch.linalg.svd(self.item_text_trs, full_matrices=False)
                self.svd_t = S_t / (S_t.sum() + 1e-7)
        self._svd_update_counter += 1

    def spectral_dropout(self, item_feats, sv_weights):
        """Differentiated dropout: high-singularity dims get higher dropout rate."""
        p = self.spectral_p_min + sv_weights * (1.0 - self.spectral_p_min)
        p = p.unsqueeze(0).clamp(0.05, 0.9)
        mask = torch.bernoulli(1 - p)
        return item_feats * mask / (1 - p + 1e-7)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse_coo_tensor(i, data, shape, device=self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        rows, cols = zip(*data_dict)
        A = sp.coo_matrix((list(data_dict.values()), (rows, cols)), shape=A.shape, dtype=np.float32)
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)

        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs

    # modality graph embedding
    def mge(self, perturbed, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)

        # ---- GMD: feature-level dropout BEFORE graph propagation ----
        if perturbed and self.use_gmd:
            if self.use_spectral_perturb:
                sv = self.svd_v if str == 'v' else self.svd_t
                item_feats_p = self.spectral_dropout(item_feats, sv)
            else:
                item_feats_p = F.dropout(item_feats, p=self.gmd_drop_rate, training=True)
            item_feats_c = item_feats
        else:
            item_feats_p = item_feats
            item_feats_c = item_feats

        # Build initial node embeddings (users from adj, items directly)
        user_feats_p = torch.sparse.mm(self.adj, item_feats_p) * self.num_inters[:self.n_users]
        mge_feats_p = torch.concat([user_feats_p, item_feats_p], dim=0)

        if self.use_gmd or self.use_hcl:
            user_feats_c = torch.sparse.mm(self.adj, item_feats_c) * self.num_inters[:self.n_users]
            mge_feats_c = torch.concat([user_feats_c, item_feats_c], dim=0)

        all_embeddings = []
        clean_embeddings = []
        ms_uncertainties = [] if self.use_multiscale_perturb else None

        for k in range(self.cl_layer):
            mge_feats_p = torch.sparse.mm(self.norm_adj, mge_feats_p)

            if self.use_gmd or self.use_hcl:
                mge_feats_c = torch.sparse.mm(self.norm_adj, mge_feats_c)
                clean_embeddings.append(mge_feats_c)

                # Multi-scale: collect per-layer uncertainty
                if self.use_multiscale_perturb:
                    ms_uncertainties.append(
                        torch.norm(mge_feats_p - mge_feats_c, dim=-1, keepdim=True))

            # Original per-layer noise (only when GMD is off, for HCL)
            if perturbed and not self.use_gmd:
                random_noise = torch.rand_like(mge_feats_p)
                mge_feats_p += torch.sign(mge_feats_p) * F.normalize(random_noise, dim=-1) * self.eps

            all_embeddings.append(mge_feats_p)

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        if perturbed and (self.use_gmd or self.use_hcl):
            final_clean = torch.stack(clean_embeddings, dim=1)
            final_clean = torch.mean(final_clean, dim=1)
            if self.use_multiscale_perturb and self.use_gmd:
                return final_embeddings, final_clean, ms_uncertainties
            return final_embeddings, final_clean
        else:
            return final_embeddings

    # ---- Geometric Information Decomposition Fusion ----
    def geometric_fusion(self, v_feats, v_feats_clean, t_feats, t_feats_clean,
                   v_ms=None, t_ms=None, *, return_diagnostics=False):
        """Geometric information decomposition for multimodal fusion.

        Cosine similarity separates shared (R) vs. unique (U_v, U_t) information.
        Dropout perturbation provides zero-parameter uncertainty estimation.
        """
        v_n = F.normalize(v_feats, dim=-1)
        t_n = F.normalize(t_feats, dim=-1)

        # --- Geometric information decomposition ---
        cos = (v_n * t_n).sum(dim=-1, keepdim=True).clamp(-1, 1)

        R = cos.clamp(min=0) * (v_n + t_n) / 2.0

        U_v = cos.clamp(max=0).abs() * (v_n - cos * t_n)
        U_t = cos.clamp(max=0).abs() * (t_n - cos * v_n)

        S = F.normalize(v_feats * t_feats, dim=-1)

        # --- Uncertainty estimation ---
        u_v = torch.norm(v_feats - v_feats_clean, dim=-1, keepdim=True)
        u_t = torch.norm(t_feats - t_feats_clean, dim=-1, keepdim=True)
        u_v_single, u_t_single = u_v, u_t

        # Multi-scale uncertainty: weighted sum across propagation layers
        if self.use_multiscale_perturb and v_ms is not None:
            u_v = self.ms_alpha * u_v
            u_t = self.ms_alpha * u_t
            for k, (u_v_k, u_t_k) in enumerate(zip(v_ms, t_ms)):
                w = [self.ms_beta, self.ms_gamma][k] if k < 2 else self.ms_gamma
                u_v = u_v + w * u_v_k
                u_t = u_t + w * u_t_k

        # Adaptive temperature: variance-driven
        if self.use_optimal_tau:
            var_sum = (u_v.var() + u_t.var()).detach()
            tau = self.gmd_tau_base / (1.0 + self.gmd_tau_scale * var_sum)
        else:
            tau = self.gmd_tau_u

        rel_v = torch.exp(-u_v / tau)
        rel_t = torch.exp(-u_t / tau)
        rel_sum = rel_v + rel_t + 1e-7
        raw_v, raw_t = rel_v, rel_t

        # --- Cold-warm soft transition ---
        lam = torch.ones_like(rel_v)
        lam[self.n_users:] = (1.0 - self.item_degree_norm).unsqueeze(-1)

        rel_v = lam * (rel_v / rel_sum) + (1.0 - lam) * 0.5
        rel_t = lam * (rel_t / rel_sum) + (1.0 - lam) * 0.5

        fused = R + rel_v * U_v + rel_t * U_t + self.synergy_weight * S
        if return_diagnostics:
            # Observe the actual calculation, without modifying the training path.
            diagnostics = {
                'u_single': torch.cat((u_v_single, u_t_single), dim=-1),
                'u_used': torch.cat((u_v, u_t), dim=-1),
                'reliability_raw': torch.cat((raw_v, raw_t), dim=-1),
                'weight_before': torch.cat((raw_v / rel_sum, raw_t / rel_sum), dim=-1),
                'weight_after': torch.cat((rel_v, rel_t), dim=-1),
                'lambda': lam.squeeze(-1),
                'cosine_vt': cos.squeeze(-1),
                'unique_norm': torch.stack((U_v.norm(dim=-1), U_t.norm(dim=-1)), dim=-1),
                'clean_norm': torch.stack((v_feats_clean.norm(dim=-1), t_feats_clean.norm(dim=-1)), dim=-1),
                'tau': torch.as_tensor(tau, device=v_feats.device, dtype=v_feats.dtype),
            }
            return fused, diagnostics
        return fused


    def gmd_alignment_loss(self, item_embs):
        """Optional cold-warm distribution alignment in the local implementation.

        Prevents cold-item representations from collapsing to the margin
        of the embedding space. Lightweight MMD between cold and warm groups.
        """
        i_embs = item_embs[self.n_users:]  # item embeddings only
        cold_mask = (self.item_degree_norm < 0.3)
        warm_mask = ~cold_mask

        if cold_mask.sum() < 2 or warm_mask.sum() < 2:
            return torch.tensor(0.0, device=i_embs.device)

        cold_mean = i_embs[cold_mask].mean(dim=0)
        warm_mean = i_embs[warm_mask].mean(dim=0)
        cold_std = i_embs[cold_mask].std(dim=0)
        warm_std = i_embs[warm_mask].std(dim=0)

        # MMD between cold and warm Gaussian approximations
        mean_diff = (cold_mean - warm_mean).pow(2).sum()
        std_diff = (cold_std - warm_std).pow(2).sum()
        return self.gmd_align_weight * (mean_diff + std_diff)

    # ---- Spectral Anti-Collapse Regularization ----
    def compute_effective_rank(self, embeddings):
        """Effective rank from spectral entropy.

        SVD of the covariance matrix D×D (D=256) is cheap.
        Effective rank = exp(entropy of singular value distribution).
        Low rank → representations collapsed to few eigendirections.
        """
        emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = emb_centered.T @ emb_centered / (embeddings.shape[0] - 1)
        S = torch.linalg.svdvals(cov)
        p = S / (S.sum() + 1e-7)
        entropy = -(p * torch.log(p + 1e-7)).sum()
        return torch.exp(entropy)

    def modality_rank_loss(self, v_embs, t_embs):
        """Optional effective-rank regularization from the local implementation."""
        v_items = v_embs[self.n_users:]
        t_items = t_embs[self.n_users:]
        rank_v = self.compute_effective_rank(v_items)
        rank_t = self.compute_effective_rank(t_items)
        max_rank = min(v_items.shape[0], v_items.shape[1])
        return -self.rank_reg_weight * (rank_v + rank_t) / (2.0 * max_rank)

    # ---- HCL: Multi-Level Contrastive Losses ----
    def ssl_intra_modal_loss(self, emb_perturbed, emb_clean, num_neg=2048):
        """L1: Intra-modal contrastive with random negative sampling.

        Full NxN would OOM (N ~ 62k). Instead sample num_neg random nodes
        as negatives per positive, SimCLR-style.
        """
        N, D = emb_perturbed.shape
        device = emb_perturbed.device
        n_sample = min(N, num_neg + 1)

        # Sample subset of nodes for this batch
        idx = torch.randperm(N, device=device)[:n_sample]
        p_sample = F.normalize(emb_perturbed[idx])  # [S, D]
        c_sample = F.normalize(emb_clean[idx])       # [S, D]

        # Positive: same node perturbed vs clean
        pos = torch.exp((p_sample * c_sample).sum(dim=1) / self.tau)  # [S]

        # Negatives: all other nodes in the sample
        sim = p_sample @ c_sample.T / self.tau      # [S, S]
        mask = torch.eye(n_sample, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        neg = torch.exp(sim).sum(dim=1)               # [S]

        return -torch.log(pos / (pos + neg + 1e-7)).mean()

    def ssl_cross_view_loss(self, struct_emb, modal_emb, num_neg=2048):
        """L3: Cross-view contrastive with random negative sampling.

        Positive: same node's structure vs multimodal embedding.
        Negatives: randomly sampled other nodes.
        """
        N, D = struct_emb.shape
        device = struct_emb.device
        n_sample = min(N, num_neg + 1)

        idx = torch.randperm(N, device=device)[:n_sample]
        s_sample = F.normalize(struct_emb[idx])  # [S, D]
        m_sample = F.normalize(modal_emb[idx])    # [S, D]

        pos = torch.exp((s_sample * m_sample).sum(dim=1) / self.tau)  # [S]

        sim = s_sample @ m_sample.T / self.tau      # [S, S]
        mask = torch.eye(n_sample, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        neg = torch.exp(sim).sum(dim=1)               # [S]

        return -torch.log(pos / (pos + neg + 1e-7)).mean()

    def ssl_triple_loss(self, emb1, emb2, all_emb, hard_neg_k=0):
        """L2: Cross-modal contrastive with optional hard negative mining."""
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)

        if hard_neg_k > 0 and hard_neg_k < all_emb.shape[0]:
            sim_matrix = torch.matmul(norm_emb1, norm_all_emb.T) / self.tau
            topk_sim, _ = torch.topk(sim_matrix, hard_neg_k, dim=1)
            ttl_score = topk_sim.exp().sum(dim=1)
        else:
            ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / (ttl_score + 1e-7)).mean()
        return ssl_loss

    # ---- SAHER: Hypergraph Regularization ----
    def saher_entropy_loss(self, i_hyper):
        """Encourage balanced hyperedge assignment across items."""
        p_hyper = i_hyper.mean(dim=0) + 1e-7  # [hyper_num]
        p_hyper = p_hyper / p_hyper.sum()
        entropy = -(p_hyper * torch.log(p_hyper)).sum()
        return -entropy  # negative entropy → minimize to maximize entropy

    def saher_diversity_loss(self, hyper_protos):
        """Encourage hyperedge prototypes to be orthogonal (diverse)."""
        # hyper_protos: [feat_dim, hyper_num]
        proto_norm = F.normalize(hyper_protos, dim=0)
        sim_matrix = torch.mm(proto_norm.T, proto_norm)  # [hyper_num, hyper_num]
        # Remove diagonal (self-similarity)
        eye = torch.eye(self.hyper_num, device=sim_matrix.device)
        off_diag = (sim_matrix - eye) ** 2
        return off_diag.sum()

    def forward(self, perturbed=False):
        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)

            # ---- SAHER: Structure-aware hyperedge injection ----
            if self.use_saher:
                # Project image features then propagate through graph structure (item→user→item)
                item_feats_v = torch.mm(self.image_embedding.weight, self.item_image_trs)
                user_feats_v = torch.sparse.mm(self.adj, item_feats_v)
                item_struct_v = torch.sparse.mm(self.norm_adj_iu, user_feats_v)
                iv_hyper = iv_hyper + self.saher_gamma * torch.mm(
                    item_struct_v, self.v_hyper_struct)

            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)

        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)

            # ---- SAHER: Structure-aware hyperedge injection ----
            if self.use_saher:
                # Project text features then propagate through graph structure (item→user→item)
                item_feats_t = torch.mm(self.text_embedding.weight, self.item_text_trs)
                user_feats_t = torch.sparse.mm(self.adj, item_feats_t)
                item_struct_t = torch.sparse.mm(self.norm_adj_iu, user_feats_t)
                it_hyper = it_hyper + self.saher_gamma * torch.mm(
                    item_struct_t, self.t_hyper_struct)

            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        inter_embs = self.cge()

        if self.v_feat is not None and self.t_feat is not None:
            # ---- Spectral weight update (lazy, every N epochs) ----
            if self.use_gmd and self.use_spectral_perturb:
                self.update_spectral_weights()

            # ---- GMD / HCL: Get modality embeddings with optional clean versions ----
            v_ms, t_ms = None, None
            if perturbed and (self.use_gmd or self.use_hcl):
                mge_v = self.mge(perturbed, 'v')
                mge_t = self.mge(perturbed, 't')
                if self.use_gmd and self.use_multiscale_perturb:
                    v_feats, v_feats_clean, v_ms = mge_v
                    t_feats, t_feats_clean, t_ms = mge_t
                else:
                    v_feats, v_feats_clean = mge_v
                    t_feats, t_feats_clean = mge_t
            else:
                v_feats = self.mge(perturbed, 'v')
                t_feats = self.mge(perturbed, 't')

            # ---- GMD: Geometric decomposition fusion ----
            if self.use_gmd:
                if perturbed:
                    mcl_embs = self.geometric_fusion(v_feats, v_feats_clean, t_feats, t_feats_clean,
                                               v_ms=v_ms, t_ms=t_ms)
                else:
                    mcl_embs = F.normalize(v_feats) + F.normalize(t_feats)
            else:
                mcl_embs = F.normalize(v_feats) + F.normalize(t_feats)

            # GHE: global hypergraph embedding
            user_indices = torch.arange(self.n_users, dtype=torch.long, device=self.device)
            user_tensor = self.user_embedding(user_indices)
            item_indices = torch.arange(self.n_items, dtype=torch.long, device=self.device)
            item_tensor = self.item_id_embedding(item_indices)

            embs = torch.cat([user_tensor, item_tensor], dim=0)
            glb, _ = self.graph_transformer(self.norm_adj, embs)

            str_emb = self.alpha * F.normalize(glb) + inter_embs

            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper),
                                                          inter_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper),
                                                          inter_embs[self.n_users:])
            av_hyper_embs = torch.cat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.cat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs

            mm_emb = self.beta * F.normalize(ghe_embs) + mcl_embs

            all_embs = str_emb + mm_emb
        else:
            all_embs = inter_embs
        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        # Build return tuple with optional extra info
        extra = {}
        if self.use_hcl and perturbed:
            extra['v_feats_clean'] = v_feats_clean
            extra['t_feats_clean'] = t_feats_clean
            extra['inter_embs'] = inter_embs  # pure CGE for cross-view contrastive
        if self.use_saher:
            extra['iv_hyper_raw'] = iv_hyper
            extra['it_hyper_raw'] = it_hyper

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs], [v_feats, t_feats], extra

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction, sign):
        ua_embeddings, ia_embeddings, hyper_embeddings, mm_embeddings, extra = self.forward(sign)
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        [v_embs, t_embs] = mm_embeddings
        v_u, v_i = torch.split(v_embs, [self.n_users, self.n_items], dim=0)
        t_u, t_i = torch.split(t_embs, [self.n_users, self.n_items], dim=0)

        # ---- L2: Cross-modal hypergraph contrast from the inherited backbone ----
        hard_neg_k = self.hcl_hard_neg_k if self.use_hcl else 0
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs, hard_neg_k) \
            + self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs, hard_neg_k)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss + self.cl_weight * batch_hcl_loss

        # ============================================================
        # HCL: BPR-driven adaptive weighting (L2 handled here)
        # ============================================================
        if self.use_hcl:
            bpr_val = batch_bpr_loss.detach()
            if self.bpr_init_stored < 1e-6:
                self.bpr_init = bpr_val.clone()
                self.bpr_init_stored = torch.tensor(1.0, device=bpr_val.device)
            self.bpr_ema = self.hcl_ema_beta * self.bpr_ema + (1 - self.hcl_ema_beta) * bpr_val

            adaptive_scale = self.bpr_init / (self.bpr_ema + 1e-7)
            adaptive_scale = adaptive_scale.clamp(0.5, self.hcl_scale_max)

            v_feats_clean = extra.get('v_feats_clean', v_embs)
            t_feats_clean = extra.get('t_feats_clean', t_embs)

            l1_v = self.ssl_intra_modal_loss(v_embs, v_feats_clean)
            l1_t = self.ssl_intra_modal_loss(t_embs, t_feats_clean)

            cge_embs = extra.get('inter_embs', torch.cat([ua_embeddings, ia_embeddings], dim=0))
            mm_full = torch.cat([v_u, v_i], dim=0)
            l3_loss = self.ssl_cross_view_loss(cge_embs, mm_full)

            if self.hcl_use_adaptive_weight:
                precision_l1 = torch.exp(-self.log_sigma_l1)
                precision_l2 = torch.exp(-self.log_sigma_l2)
                precision_l3 = torch.exp(-self.log_sigma_l3)
                weighted_l1 = adaptive_scale * (precision_l1 * (l1_v + l1_t) + self.log_sigma_l1)
                weighted_l2 = adaptive_scale * (precision_l2 * batch_hcl_loss + self.log_sigma_l2)
                weighted_l3 = adaptive_scale * (precision_l3 * l3_loss + self.log_sigma_l3)
                loss = loss + weighted_l2 + weighted_l1 + weighted_l3
            else:
                loss = loss + adaptive_scale * (self.cl_weight * batch_hcl_loss
                    + self.hcl_l1_weight * (l1_v + l1_t) + self.hcl_l3_weight * l3_loss)

        # ============================================================
        # GMD: Cold-warm distribution alignment
        # ============================================================
        if self.use_gmd:
            loss = loss + self.gmd_alignment_loss(torch.cat([ua_embeddings, ia_embeddings], dim=0))

        # ============================================================
        # Spectral Anti-Collapse
        # ============================================================
        if self.use_rank_reg:
            loss = loss + self.modality_rank_loss(v_embs, t_embs)

        # ============================================================
        # SAHER: Hypergraph structure regularization
        # ============================================================
        saher_loss = torch.tensor(0.0, device=loss.device)
        if self.use_saher:
            iv_hyper_raw = extra.get('iv_hyper_raw', None)
            it_hyper_raw = extra.get('it_hyper_raw', None)

            if iv_hyper_raw is not None:
                saher_loss = saher_loss + self.saher_entropy_weight * self.saher_entropy_loss(iv_hyper_raw)
                saher_loss = saher_loss + self.saher_diversity_weight * self.saher_diversity_loss(self.v_hyper)
            if it_hyper_raw is not None:
                saher_loss = saher_loss + self.saher_entropy_weight * self.saher_entropy_loss(it_hyper_raw)
                saher_loss = saher_loss + self.saher_diversity_weight * self.saher_diversity_loss(self.t_hyper)

            loss = loss + saher_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer

    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret
