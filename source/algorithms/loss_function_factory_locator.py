from algorithms.base_locator import BaseLocator
from algorithms.loss_factory_base import LossFactoryBase


class LossFunctionFactoryLocator(BaseLocator):
    """
    General Loss function factory locator, to return loss function factory
    """

    def __init__(self):
        super().__init__(LossFactoryBase)
