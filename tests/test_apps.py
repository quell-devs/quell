from torchapp.testing import TorchAppTestCase
from quell.apps import Quell


class TestQuell(TorchAppTestCase):
    app_class = Quell
