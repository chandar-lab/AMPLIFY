"""Utility functions related to working with [Hydra](https://hydra.cc)."""

from __future__ import annotations

import functools
import importlib
import inspect
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import fields, is_dataclass
from logging import getLogger as get_logger
from typing import (
    Any,
    TypeVar,
)

import hydra.utils
import hydra_zen.structured_configs._utils
import omegaconf
from hydra_zen import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

if typing.TYPE_CHECKING:
    from project.configs.config import Config

logger = get_logger(__name__)


def get_full_name(object_type: type) -> str:
    return object_type.__module__ + "." + object_type.__qualname__


def get_outer_class(inner_class: type) -> type:
    inner_full_name = get_full_name(inner_class)
    parent_full_name, _, _ = inner_full_name.rpartition(".")
    outer_class_module, _, outer_class_name = parent_full_name.rpartition(".")
    mod = importlib.import_module(outer_class_module)
    return getattr(mod, outer_class_name)


def get_attr(obj: Any, *attributes: str):
    """Recursive version of `getattr` when the attribute is like 'a.b.c'."""

    if not attributes:
        return obj
    for attribute in attributes:
        try:
            return _get_attr(obj, attribute)
        except AttributeError:
            pass
    raise AttributeError(f"Could not find any attributes matching {attributes} on {obj}.")


def _get_attr(obj: Any, potentially_nested_attribute: str):
    """Recursive version of `getattr` when the attribute is like 'a.b.c'."""
    subobj = obj
    for attr in potentially_nested_attribute.split("."):
        subobj = getattr(subobj, attr)
        return subobj


def _has_attr(obj: Any, potentially_nested_attribute: str):
    """Recursive version of `hasattr` when the attribute is like 'a.b.c'."""
    for attribute in potentially_nested_attribute.split("."):
        if not hasattr(obj, attribute):
            return False
        obj = getattr(obj, attribute)
    return True


def _set_attr(obj: Any, potentially_nested_attribute: str, value: Any) -> None:
    """Recursive version of `setattr` when the attribute is like 'a.b.c'."""
    attributes = potentially_nested_attribute.split(".")
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def register_instance_attr_resolver(instantiated_objects_cache: dict[str, Any]) -> None:
    """Registers the `instance_attr` custom resolver with OmegaConf."""
    OmegaConf.register_new_resolver(
        "instance_attr",
        functools.partial(
            instance_attr,
            _instantiated_objects_cache=instantiated_objects_cache,
        ),
        replace=True,
    )


def resolve_dictconfig(dict_config: DictConfig) -> Config:
    """Resolve all interpolations in the `DictConfig`.

    Returns a [`Config`][project.configs.Config] object, which is a simple dataclass used to give
    nicer type hints for the contents of an experiment config.
    """
    # Important: Register this fancy little resolver here so we can get attributes of the
    # instantiated objects, not just the configs!
    instantiated_objects_cache: dict[str, Any] = {}
    register_instance_attr_resolver(instantiated_objects_cache)
    # Convert the "raw" DictConfig (which uses the `Config` class to define it's structure)
    # into an actual `Config` object:

    # TODO: Seems to only be necessary now that the datamodule group is optional?
    # Need to manually nudge OmegaConf so that it instantiates the datamodule first.
    if dict_config["datamodule"]:
        with omegaconf.open_dict(dict_config):
            v = dict_config._get_flag("allow_objects")
            dict_config._set_flag("allow_objects", True)
            if isinstance(dict_config["datamodule"], LightningDataModule):
                dm = dict_config["datamodule"]
            else:
                dm = hydra.utils.instantiate(dict_config["datamodule"])
                dict_config["datamodule"] = dm
            instantiated_objects_cache["datamodule"] = dm
            dict_config._set_flag("allow_objects", v)

    config = OmegaConf.to_object(dict_config)
    from project.configs.config import Config

    assert isinstance(config, Config)
    # If we had to instantiate some of the configs into objects in order to find the interpolated
    # values (e.g. ${instance_attr:datamodule.dims} or similar in order to construct the network),
    # then we don't waste that, put the object instance into the config.
    for attribute, pre_instantiated_object in instantiated_objects_cache.items():
        if not _has_attr(config, attribute):
            logger.debug(
                f"Leftover temporarily-instantiated attribute {attribute} in the instantiated "
                f"objects cache."
            )
            continue
        value_in_config = _get_attr(config, attribute)
        if pre_instantiated_object != value_in_config:
            logger.debug(
                f"Overwriting the config at {attribute} with the already-instantiated "
                f"object {pre_instantiated_object}"
            )
            _set_attr(config, attribute, pre_instantiated_object)

    return config


def instance_attr(
    *attributes: str,
    _instantiated_objects_cache: MutableMapping[str, Any] | None = None,
):
    """Allows interpolations of the instantiated objects attributes (rather than configs).

    !!! note "This is very hacky"

        This is quite hacky and very dependent on the code of Hydra / OmegaConf not changing too
        much in the future. For this reason, consider pinning the versions of these libraries in
        your project if you intend do use this feature.

    This works during a call to `hydra.utils.instantiate`, by looking at the stack trace to find
    the instantiated objects, which are in a variable in that function.

    If there is a `${instance_attr:datamodule.num_classes}` interpolation in a config, this will:

    1. instantiate the `datamodule` config
    2. store it at the key `'datamodule'` in the instantiated objects cache dict (if passed).

        > (This is useful since it makes it possible for us to later reuse this instantiated
        datamodule instead of re-instantiating it.)

    3. Retrieve the value of the attribute (`getattr(datamodule, 'num_classes')`) and return it.
    """
    if not attributes:
        raise RuntimeError("Need to pass one or more attributes to this resolver.")
    assert _being_called_in_hydra_context()
    logger.debug(f"Custom resolver is being called to get the value of {attributes}.")

    current_frame = inspect.currentframe()
    assert current_frame
    assert current_frame.f_back
    frame_infos = inspect.getouterframes(current_frame.f_back)

    # QUITE HACKY: These local variables are defined in the _to_object function inside OmegaConf.

    init_field_items: list[dict[str, Any]] = []
    non_init_field_items: list[dict[str, Any]] = []

    for frame_info in frame_infos:
        if frame_info.function != DictConfig._to_object.__name__:
            continue
        _self_obj: DictConfig = frame_info.frame.f_locals["self"]

        assert "init_field_items" in frame_info.frame.f_locals
        frame_init_field_items = frame_info.frame.f_locals["init_field_items"]
        # logger.debug(
        #     f"Gathered {frame_init_field_items} from the init_field_items variable."
        # )
        init_field_items.append(frame_init_field_items.copy())

        assert "non_init_field_items" in frame_info.frame.f_locals
        frame_non_init_field_items = frame_info.frame.f_locals["non_init_field_items"]
        non_init_field_items.append(frame_non_init_field_items.copy())
        # logger.debug(
        #     f"Gathered {frame_non_init_field_items} from the non_init_field_items variable."
        # )
    assert init_field_items or non_init_field_items
    # Okay so we now have all the *instantiated* attributes! We can do the interpolation by
    # getting its value!
    # BUG: This isn't quite working as I'd like, the interpolation is trying to get the attribute
    # on the object *instance*, but this is getting called during the first OmegaConf.to_object
    # which is only responsible for creating the configs, not the objects!
    all_init_field_items = ChainMap(*reversed(init_field_items))

    # BUG: There are some unexpected attributes being stored in the instantiated objects cache
    # which are fields of the dataclass (e.g. `layer_widths, sigma_gen, batch_size`, etc instead of
    # being configs!
    if _instantiated_objects_cache is not None:
        _instantiated_objects_cache.update(all_init_field_items)

    objects_cache: Mapping[str, Any] = _instantiated_objects_cache or all_init_field_items

    for attribute in attributes:
        logger.debug(f"Looking for instantiated attribute {attribute} from {all_init_field_items}")
        key, *nested_attribute = attribute.split(".")
        if key not in objects_cache:
            continue

        obj = objects_cache[key]

        # Recursive getattr that tries to instantiate configs to objects when missing an attribute
        # values.
        new_instantiated_objects: dict[str, Any] = {}  # NOTE: unused so far.
        for level, attr_part in enumerate(nested_attribute):
            if hasattr(obj, attr_part):
                obj = getattr(obj, attr_part)
                continue

            if not (
                (isinstance(obj, dict | DictConfig) and "_target_" in obj)
                or (is_dataclass(obj) and any(f.name == "_target_" for f in fields(obj)))
            ):
                # attribute not found, and the `obj` isn't a config with a _target_ field, so we
                # won't be able to instantiate it.
                break

            # NOTE: Instantiating the object just to get the value, but we store the instantiated
            # object in the cache dict so it can be reused later to avoid re-instantiating the obj.

            path_so_far = key
            if nested_attribute[:level]:
                path_so_far += "." + ".".join(nested_attribute[:level])
            logger.debug(
                f"Will pro-actively attempt to instantiate {path_so_far} to retrieve "
                f"the {attr_part} attribute."
            )
            try:
                instantiated_obj = instantiate(obj)
            except Exception as err:
                logger.error(
                    f"Unable to instantiate {obj} to get the missing {attr_part} "
                    f"attribute: {err}"
                )
                break
            new_instantiated_objects[path_so_far] = instantiated_obj

            logger.info(
                f"Instantiated the config at {path_so_far!r} while trying to find one of the "
                f"{attributes} attributes."
            )

            if _instantiated_objects_cache is None:
                logger.warning(
                    f"The config {obj} at path {path_so_far} was instantiated while trying to "
                    f"find the {attr_part} portion of the {attribute} attributes in {attributes}."
                    f"This compute is being wasted because {_instantiated_objects_cache=}. "
                    f"If this is an expensive config to instantiate (e.g. Dataset, model, etc), "
                    f"consider passing a dictionary that will be populated with the instantiated "
                    f"objects so they can be reused instead of being discarded and instantiated "
                    f"again."
                )
            else:
                _instantiated_objects_cache[path_so_far] = instantiated_obj

            if not hasattr(instantiated_obj, attr_part):
                logger.debug(
                    f"We instantiated the object at path {path_so_far}, but it doesn't have the "
                    f"{attr_part} attribute. Moving on to the next attribute in attributes."
                )
                # _instantiated_objects_cache[path_so_far] = instantiated_obj
                break

            # Retrieve the attribute from the instantiated object.
            obj = getattr(instantiated_obj, attr_part)
        else:
            # Found the attribute on the object! (didn't break)
            return obj

        logger.debug(f"Trying the next attribute in {attributes}.")

    if not objects_cache:
        if len(attributes) == 1:
            attribute = attributes[0]
            parent, _, attr = attribute.rpartition(".")
            raise RuntimeError(
                f"Could not find attribute {attribute!r} on any instantiated config! "
                f"Did you forget to set a value for the {parent!r} config? Are you sure that the "
                f"object at path {parent!r} has an attribute named {attr!r}?"
            )
        raise RuntimeError(
            f"Could not find any of {attributes} in the configs that were instantiated! "
            f"Did you forget to set a value for one of these configs?"
        )
    raise RuntimeError(
        f"Could not find any of these attributes {attributes} from the instantiated configs: "
        + str({k: type(v) for k, v in objects_cache.items()})
        # + "\n".join([f"- {k}: {type(v)}" for k, v in all_init_field_items.items()])
    )


def _being_called_in_hydra_context() -> bool:
    """Returns `True` if this function is being called indirectly by Hydra/OmegaConf.

    Can be used in a field default factory to change the default value based on whether the config
    is being instantiated by Hydra vs in code. For example, you could have a default value for a
    field `a` of a dataclass `Foo` that is 'a=123` when called in code, for example when doing
    `Foo()` in a python file, but then when called by Hydra, the default value could be
    `a=${some_interpolation}`, so that Hydra/OmegaConf resolve that interpolation.
    """
    import hydra.core.utils
    import omegaconf._utils
    import omegaconf.base

    return _being_called_by(
        omegaconf._utils.get_dataclass_data,
        omegaconf.base.Container._resolve_interpolation_from_parse_tree,
        hydra.core.utils.run_job,
    )


def _being_called_by(*functions: Callable) -> bool:
    current = inspect.currentframe()
    assert current

    caller_frames = inspect.getouterframes(current, context=0)
    if any(
        frame_info.function in [function.__name__ for function in functions]
        for frame_info in caller_frames
    ):
        return True
    return False


Target = TypeVar("Target")


def make_config_and_store(
    target: Callable[..., Target], *, store: hydra_zen.ZenStore, **overrides
):
    """Creates a config dataclass for the given target and stores it in the config store.

    This uses [hydra_zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html)
    to create the config dataclass and stores it at the name `config_name`, or `target.__name__`.
    """
    _current_frame = inspect.currentframe()
    assert _current_frame
    _calling_module = inspect.getmodule(_current_frame.f_back)
    assert _calling_module

    config = hydra_zen.builds(
        target,
        populate_full_signature=True,
        zen_partial=True,
        zen_dataclass={
            "cls_name": f"{target.__name__}Config",
            # BUG: Causes issues, tries to get the config from the module again, which re-creates
            # it?
            "module": _calling_module.__name__,
            # TODO: Seems to be causing issues with `_target_` being overwritten?
            "frozen": False,
        },
        # Seems to make things not pickleable!
        # zen_wrappers=pydantic_parser,
        **overrides,
    )
    name_of_config_in_store = target.__name__
    logger.debug(f"Created a config entry {name_of_config_in_store} for {target.__qualname__}")
    store(config, name=name_of_config_in_store)
    return config
