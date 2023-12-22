from typing import Dict, Optional

from torch import Tensor
from mmengine.optim.optimizer import OptimWrapper


class CustomOptimWrapper(OptimWrapper):

    def update_params(self,
                      loss: Tensor,
                      backward_kwargs: Optional[Dict] = None,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        if backward_kwargs is None:
            backward_kwargs = {}

        loss = self.scale_loss(loss)
        self.backward(loss, **backward_kwargs)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)
