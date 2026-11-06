# Integration and arm definitions

`accuracy_harness.build(author_module, original_rollout_function, "settle_only")` returns the selected wrapped rollout for the upstream Push-T runtime. The code transforms the original function to avoid an unused final ensemble transition and adds bounded goal settling. The fixed visible goal and public low-dimensional observation are the only settling inputs; evaluator rewards are not provided to the controller.

Use the untouched author rollout for the authors' arm and exactly the same checkpoint, obstacle scenario and seeded physical start. `speed_harness.py` is the earlier compute-only variant; its metrics must not be attributed to the v4 accuracy variant. Refer to the archived campaign files for argument binding, original imports and checkpoint loading. The original environment and frozen checkpoint remain external dependencies.
