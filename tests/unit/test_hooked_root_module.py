from unittest.mock import Mock

import torch
from torch import nn

from transformer_lens.hook_points import HookedRootModule, HookPoint


class Bla(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1)
        self.b = nn.Linear(1, 1)
        self.hook_a = HookPoint()
        self.hook_b = HookPoint()
        self.setup()

    def forward(self, x):
        return self.hook_a(self.a(x)) + self.hook_b(self.b(x))


def test_run_with_cache():
    model = Bla()
    inp = torch.tensor([1.0])
    _, cache = model.run_with_cache(inp)
    assert isinstance(cache["hook_a"], torch.Tensor)
    assert isinstance(cache["hook_b"], torch.Tensor)

    model.clear_contexts()

    def save_hook(tensor: torch.Tensor, hook: HookPoint, cache: dict = None):
        # for attention heads the pos dimension is the third from last
        if hook.name is None:
            raise RuntimeError("Hook should have been provided a name")
        if cache is None:
            raise RuntimeError(f"Cache should have been provided for {hook.name}")
        cache[hook.name] = tensor * 2  # double the value
        return tensor

    _, cache2 = model.run_with_cache(inp, fwd_save_hooks=[("hook_a", save_hook)], names_filter=None)

    assert torch.allclose(cache["hook_b"], cache2["hook_b"])
    assert torch.allclose(cache["hook_a"] * 2, cache2["hook_a"])


MODEL_NAME = "solu-2l"


def test_enable_hook_with_name():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
    model.context_level = 5

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook_with_name("linear", hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_called_with(hook, dir="fwd", level=5)


def test_enable_hooks_for_points():
    model = HookedRootModule()
    model.mod_dict = {}
    model.context_level = 5

    hook_points = {
        "linear": Mock(),
        "attn": Mock(),
    }

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "bwd"

    print(hook_points.items())
    model._enable_hooks_for_points(
        hook_points=hook_points.items(), enabled=enabled, hook=hook, dir=dir
    )

    hook_points["attn"].add_hook.assert_called_with(hook, dir="bwd", level=5)
    hook_points["linear"].add_hook.assert_not_called()


def test_enable_hook_with_string_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
    model.context_level = 5

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook("linear", hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_called_with(hook, dir="fwd", level=5)


def test_enable_hook_with_callable_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
    model.hook_dict = {
        "linear": Mock(),
        "attn": Mock(),
    }
    model.context_level = 5

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook(enabled, hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["linear"].add_hook.assert_not_called()
