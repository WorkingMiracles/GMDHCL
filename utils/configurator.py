"""Built-in runtime defaults; no dataset or configuration files are distributed."""
import copy
import torch

_COMMON_PARAMETERS = {'use_gpu': True,
 'seed': [999],
 'data_path': './data/',
 'inter_splitting_label': 'x_label',
 'filter_out_cod_start_users': True,
 'is_multimodal_model': True,
 'checkpoint_dir': 'saved',
 'save_recommended_topk': True,
 'recommend_topk': 'recommend_topk/',
 'embedding_size': 256,
 'weight_decay': 0.0,
 'req_training': True,
 'epochs': 1000,
 'stopping_step': 20,
 'train_batch_size': 2048,
 'learner': 'adam',
 'learning_rate': 0.001,
 'learning_rate_scheduler': [1.0, 50],
 'eval_step': 1,
 'training_neg_sample_num': 1,
 'use_neg_sampling': True,
 'use_full_sampling': False,
 'NEG_PREFIX': 'neg__',
 'USER_ID_FIELD': 'userID',
 'ITEM_ID_FIELD': 'itemID',
 'TIME_FIELD': 'timestamp',
 'field_separator': '\t',
 'metrics': ['Recall', 'NDCG'],
 'topk': [5, 10, 20, 50],
 'valid_metric': 'Recall@20',
 'eval_batch_size': 4096,
 'use_raw_features': False,
 'max_txt_len': 32,
 'max_img_size': 256,
 'vocab_size': 30522,
 'type_vocab_size': 2,
 'hidden_size': 4,
 'pad_token_id': 0,
 'max_position_embeddings': 512,
 'layer_norm_eps': 1e-12,
 'hidden_dropout_prob': 0.1,
 'end2end': False,
 'hyper_parameters': ['n_ui_layers',
                      'cl_layer',
                      'n_hyper_layer',
                      'hyper_num',
                      'keep_rate',
                      'alpha',
                      'cl_weight',
                      'reg_weight',
                      'K',
                      'beta',
                      'eps',
                      'depth_cl',
                      'seed'],
 'vision_feature_file': 'image_feat.npy',
 'text_feature_file': 'text_feat.npy',
 'n_ui_layers': [4],
 'cl_weight': [0.0001],
 'reg_weight': [1e-06],
 'beta': [0.3],
 'K': [4],
 'depth_cl': [1],
 'feat_embed_dim': 256,
 'cf_model': 'lightgcn',
 'use_gmd': True,
 'use_hcl': True,
 'use_saher': False,
 'use_rank_reg': True,
 'gmd_tau_u': 2.0,
 'gmd_drop_rate': 0.3,
 'gmd_align_weight': 0.01,
 'synergy_weight': 1.0,
 'use_spectral_perturb': False,
 'spectral_update_interval': 5,
 'spectral_p_min': 0.1,
 'use_optimal_tau': False,
 'gmd_tau_base': 2.0,
 'gmd_tau_scale': 1.0,
 'use_multiscale_perturb': True,
 'ms_alpha': 1.0,
 'ms_beta': 0.5,
 'ms_gamma': 0.3,
 'hcl_l1_weight': 1.0,
 'hcl_l3_weight': 0.5,
 'hcl_hard_neg_k': 16,
 'hcl_use_adaptive_weight': True,
 'hcl_ema_beta': 0.99,
 'hcl_scale_max': 10.0,
 'rank_reg_weight': 0.01,
 'saher_gamma': 0.1,
 'saher_entropy_weight': 0.01,
 'saher_diversity_weight': 0.01,
 'gpu_id': 0}

_DATASET_PARAMETERS = {'baby': {'inter_file_name': 'baby.inter',
          'cl_layer': [4],
          'n_hyper_layer': [1],
          'hyper_num': [4],
          'keep_rate': [0.5],
          'alpha': [0.1],
          'eps': [0.15]},
 'sports': {'inter_file_name': 'sports.inter',
            'cl_layer': [3],
            'n_hyper_layer': [1],
            'hyper_num': [4],
            'keep_rate': [0.4],
            'alpha': [0.6],
            'eps': [0.2]},
 'clothing': {'inter_file_name': 'clothing.inter',
              'cl_layer': [3],
              'n_hyper_layer': [2],
              'hyper_num': [64],
              'keep_rate': [0.2],
              'alpha': [0.2],
              'eps': [0.25]},
 'tiktok': {'inter_file_name': 'tiktok.inter',
            'cardinality_file': 'audit.json',
            'cl_layer': [3],
            'n_hyper_layer': [2],
            'hyper_num': [64],
            'keep_rate': [0.2],
            'alpha': [0.2],
            'eps': [0.25]}}


class Config(object):
    """Load built-in GMDHCL defaults and optional caller-supplied overrides.

    The configuration interface is inherited from the local RecBole-style
    infrastructure. Defaults are deep-copied so independent runs cannot mutate
    each other's lists or dictionaries. No files are read by this class.
    """

    def __init__(self, model=None, dataset=None, config_dict=None):
        if model != 'GMDHCL':
            raise ValueError('The supported model is GMDHCL')
        if dataset not in _DATASET_PARAMETERS:
            raise ValueError('Unsupported dataset: ' + str(dataset))
        overrides = copy.deepcopy(config_dict or {})
        self.final_config_dict = copy.deepcopy(_COMMON_PARAMETERS)
        self.final_config_dict.update(copy.deepcopy(_DATASET_PARAMETERS[dataset]))
        self.final_config_dict.update(overrides)
        self.final_config_dict.update(model=model, dataset=dataset)
        self._set_default_parameters()
        self._init_device()

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = valid_metric not in smaller_metric
        if 'seed' not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'].append('seed')

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        gpu_id = self.final_config_dict.get('gpu_id', 0)
        self.final_config_dict['gpu_id'] = gpu_id
        self.final_config_dict['device'] = torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() and use_gpu else 'cpu')

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('index must be a str.')
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError('index must be a str.')
        return key in self.final_config_dict

    def __str__(self):
        return '\n' + '\n'.join('{}={}'.format(key, value)
                                  for key, value in self.final_config_dict.items()) + '\n\n'

    def __repr__(self):
        return self.__str__()
