from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner

class RunnerRegistry:
    def __init__(self):
        self.runner_classes = {}
        self.lazy_runners = {}
    
    def register(self, name: str, runner_class: OnPolicyRunner):
        self.runner_classes[name] = runner_class

    def register_lazy(self, name, module, attribute):
        self.lazy_runners[name] = (module, attribute)
    
    def get_runner_class(self, name: str) -> OnPolicyRunner:
        if name not in self.runner_classes and name not in self.lazy_runners:
            # Also support registry-first imports without a circular runner import.
            import rsl_rl.runners
        if name not in self.runner_classes and name in self.lazy_runners:
            from importlib import import_module
            module, attribute = self.lazy_runners[name]
            self.register(name, getattr(import_module(module), attribute))
        return self.runner_classes[name]

runner_registry = RunnerRegistry()
