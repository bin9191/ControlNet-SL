import argparse
import os
from collections import OrderedDict, namedtuple
from typing import Any, IO
import yaml



class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""
        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


class Config:
    """ Retrieving configuration parameters by parsing YAML configuration file """

    _instance = None

    @staticmethod
    def construct_include(loader: Loader, node: yaml.Node) -> Any:
        """Include file referenced at node."""

        filename = os.path.abspath(
            os.path.join(loader.root_path, loader.construct_scalar(node))
        )
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r", encoding="utf-8") as config_file:
            if extension in ("yaml", "yml"):
                return yaml.load(config_file, Loader)
            else:
                return "".join(config_file.readlines())

    def __new__(cls):
        if cls._instance is None:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "-c",
                "--config",
                type=str,
                #default="./config.yml",
                help="configuration file.",
            )
            args = parser.parse_args()
            Config.args = args

  
            cls._instance = super(Config, cls).__new__(cls)

            yaml.add_constructor("!include", Config.construct_include, Loader)

            filename = args.config
            if(filename == None):
                # if no config file is passed
                raise ValueError("A configuration file must be supplied.")
            elif not os.path.isfile(filename):
                # if the given configuration file does not exist, raise an error
                raise ValueError("Provided configuration file path is invalid")
            else:
                with open(filename, "r", encoding="utf-8") as config_file:
                    config = yaml.load(config_file, Loader)
            
            
            Config.train = Config.namedtuple_from_dict(config["train"])
            Config.utils = Config.namedtuple_from_dict(config["utils"])
            Config.valid = Config.namedtuple_from_dict(config["valid"])
            if "params" in config:
                Config.params = Config.namedtuple_from_dict(config["params"])

        return cls._instance
    
    @staticmethod
    def namedtuple_from_dict(obj):
        """Creates a named tuple from a dictionary."""
        if isinstance(obj, dict):
            fields = sorted(obj.keys())
            namedtuple_type = namedtuple(
                typename="Config", field_names=fields, rename=True
            )
            field_value_pairs = OrderedDict(
                (str(field), Config.namedtuple_from_dict(obj[field]))
                for field in fields
            )
            try:
                return namedtuple_type(**field_value_pairs)
            except TypeError:
                # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
                return dict(**field_value_pairs)
        elif isinstance(obj, (list, set, tuple, frozenset)):
            return [Config.namedtuple_from_dict(item) for item in obj]
        else:
            return obj