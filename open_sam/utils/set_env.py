# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmseg into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmseg default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmseg`, and all registries will build modules from mmseg's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import open_sam.datasets  # noqa: F401,F403
    import open_sam.modeling  # noqa: F401,F403
    import open_sam.metric

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('open_sam')
        if never_created:
            DefaultScope.get_instance('open_sam', scope_name='open_sam')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'open_sam':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "open_sam", '
                          '`register_all_modules` will force the current'
                          'default scope to be "open_sam". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'open_sam-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='open_sam')
