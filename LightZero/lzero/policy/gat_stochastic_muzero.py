import copy
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
from ding.model import model_wrap
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss

from lzero.mcts import StochasticMuZeroMCTSCtree as MCTSCtree
from lzero.mcts import StochasticMuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs
from lzero.policy.stochastic_muzero import StochasticMuZeroPolicy
from lzero.policy.utils import plot_topk_accuracy, visualize_avg_softmax, plot_argmax_distribution


@POLICY_REGISTRY.register('gat_stochastic_muzero')
class GATStochasticMuZeroPolicy(StochasticMuZeroPolicy):
    """
    Overview:
        The policy class for GAT-based Stochastic MuZero.
        Extends the standard StochasticMuZero policy to use Graph Attention Networks.
    """

    def _init_model(self) -> None:
        """
        Overview:
            Initialize the model. Use GAT-based StochasticMuZero model.
        """
        model_info = self._get_model_info()
        model = self._model_wrap(model_info[0])
        self._learn_model = model
        self._collect_model = model
        self._eval_model = model

    def _get_model_info(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return the model name and import path for GAT-based StochasticMuZero.
        """
        return 'GATStochasticMuZeroModel', ['lzero.model.gat_stochastic_muzero_model']

    def _model_wrap(self, model_name: str):
        """
        Overview:
            Create and wrap the GAT StochasticMuZero model.
        """
        from ding.utils import import_module
        from ding.model import model_wrap
        
        # Import the GAT StochasticMuZero model
        import_module(['lzero.model.gat_stochastic_muzero_model'])
        
        # Create model with the config
        model = self._cfg.model.copy()
        model.update(self._cfg.model)
        
        # Use the registered GAT StochasticMuZero model
        model_class = getattr(__import__('lzero.model.gat_stochastic_muzero_model', fromlist=['GATStochasticMuZeroModel']), 'GATStochasticMuZeroModel')
        
        # Remove model_type and model fields that are not constructor parameters
        model_params = model.copy()
        model_params.pop('model_type', None)
        model_params.pop('model', None)
        
        # Create the model instance
        model_instance = model_class(**model_params)
        
        # Wrap the model
        model_wrapped = model_wrap(
            model_instance,
            wrapper_name='base',
            cuda=self._cfg.cuda,
            **self._cfg.model_wrapper_cfg
        )
        
        return model_wrapped

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Initialize the collect model and MCTS utils for GAT StochasticMuZero.
        """
        self._collect_model = self._model_wrap('GATStochasticMuZeroModel')
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Initialize the eval model and MCTS utils for GAT StochasticMuZero.
        """
        self._eval_model = self._model_wrap('GATStochasticMuZeroModel')
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)
