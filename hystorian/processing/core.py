from collections.abc import Iterable
from dataclasses import dataclass, field
import datetime
import fnmatch
import inspect
import types
from typing import Any


import h5py
import numpy as np
from . import utils


@dataclass(repr=False, frozen=True)
class FileHandler:
    @dataclass(frozen=True)
    class Attributes:
        filename: str

        def __contains__(self, path: str) -> bool:
            with h5py.File(self.filename, "r") as f:
                if path in f.keys():
                    return True
                else:
                    return False

        def __getitem__(self, path: str | None = None) -> dict:
            if path is None:
                path = ""

            with h5py.File(self.filename, "r") as f:
                if path != "":
                    f = f[path]
                return {key: f.attrs[key] for key in f.attrs.keys()}

        def __setitem__(self, path: str | None = None, attributes: dict | None = None) -> None:
            if path is None:
                path = ""

            with h5py.File(self.filename, "r+") as f:
                for key, attribute in attributes.items():
                    f[path].attrs[key] = attribute

    filename: str
    structure: dict = field(init=False)
    attrs: Attributes = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "attrs", self.Attributes(self.filename))
        with h5py.File(self.filename, "r") as f:
            dictionary = self._generate_file_struct("/", f)
        object.__setattr__(self, "structure", dictionary)

    def __getitem__(self, path: str = ""):
        # This function use the structure attribute to avoid opening the hdf5 file if not necessary.
        # Since the structure of the hdf5 is stored in self.structure, the file need to be open only if an h5py.Dataset is asked.

        if path == "":
            return list(self.structure.keys())

        return self._traverse_dict(self.structure, path)

    def __setitem__(self, path: str | list[str], obj: Any, current_path=None):
        if current_path is None:
            current_path = []

        if isinstance(path, str):
            path = path.split("/")

        if len(path) > 1:
            if path[0] not in self.structure.keys():
                with h5py.File(self.filename, "r+") as f:
                    if current_path is []:
                        f.create_group(path[0])
                    else:
                        f["/".join(current_path)].create_group(path[0])

            current_path.append(path[0])
            if not isinstance(obj, Iterable):
                obj = np.array([obj])
            return self.__setitem__(path[1:], obj, current_path)
        else:
            if obj is not None:
                with h5py.File(self.filename, "r+") as f:
                    f["/".join(current_path)].create_dataset(path[0], obj)

        with h5py.File(self.filename, "r+") as f:
            dictionary = self._generate_file_struct("/", f)
        object.__setattr__(self, "structure", dictionary)

    def __delitem__(self, path: str | list[str]):
        with h5py.File(self.filename, "r+") as f:
            del f[path]
            dictionary = self._generate_file_struct("/", f)

        object.__setattr__(self, "structure", dictionary)

    def __contains__(self, path: str) -> bool:
        with h5py.File(self.filename, "r") as f:
            return path in f

    def apply(
        self,
        function: callable,
        inputs: list[str] | str,
        folder_names: list[str] | str | None = None,
        output_names: list[str] | str | None = None,
        use_attrs: list[str] | str | None = None,
        prop_attrs: list[str] | str | None = None,
        increment_proc: bool = True,
        repeat=None,
        **kwargs,
    ) -> Any:
        def convert_to_list(inputs):
            if isinstance(inputs, list):
                return inputs
            return [inputs]

        inputs = convert_to_list(inputs)

        ## TO DO
        inputs = self._path_search(inputs, repeat)
        inputs = list(map(list, zip(*inputs)))
        # Implement smart path searching
        for input in inputs:
            print(input)
            if output_names is None:
                output_names = input[0].rsplit("/", 1)[1]
            output_names = convert_to_list(output_names)

            if folder_names is None:
                folder_names = input[0].rsplit("/", 2)[1]

            if use_attrs is not None:
                use_attrs = convert_to_list(use_attrs)

            if prop_attrs is not None:
                prop_attrs = convert_to_list(prop_attrs)

            data_list = []
            prop_attr_dict = {}
            use_attr_dict = {}

            for path in input:
                data_list.append(self[path])

                if prop_attrs is not None:
                    for prop_attr in prop_attrs:
                        ## TO DO
                        # This should proably be handled when the attribute is initially created so it is consistent

                        # if prop_attr == "scale_m_per_px" and "scale (m/px)" in f[path].attrs:
                        #    prop_attr_dict[prop_attr] = f[path].attrs["scale (m/px)"]
                        if (prop_attr not in prop_attr_dict.keys()) and (prop_attr in self.attrs[path]):
                            prop_attr_dict[prop_attr] = self.attrs[path][prop_attr]

                if use_attrs is not None:
                    for use_attr in use_attrs:
                        if (use_attr not in use_attr_dict.keys()) and (use_attr in self.attrs[path]):
                            use_attr_dict["source_" + use_attr] = self.attrs[path][use_attr]

            kwargs.update(use_attr_dict)

            result = function(*data_list, **kwargs)

            if result is None:
                return None
            if not isinstance(result, tuple):
                result = tuple([result])

            if len(output_names) != len(result):
                raise Exception("Error: Unequal amount of outputs and output names")

            num_proc = len(self.structure["process"].keys())

            if (
                increment_proc
                or f"{str(num_proc).zfill(3)}-{function.__name__}" not in self.structure["process"].keys()
            ):
                num_proc += 1

            out_folder_location = f"{'process'}/{str(num_proc).zfill(3)}-{function.__name__}/{folder_names}"

            for name, data in zip(output_names, result):
                self[f"{out_folder_location}/{name}"] = data

                if prop_attrs is not None:
                    self.attrs[f"{out_folder_location}/{name}"] = prop_attr_dict

                self._write_generic_attributes(f"{out_folder_location}/{name}", input, name, function)
                self._write_kwargs_as_attributes(
                    f"{out_folder_location}/{name}", function, kwargs, first_kwarg=len(input)
                )

    def _write_generic_attributes(
        self, out_folder_location: str, in_paths: list[str] | str, output_name: str, function: callable
    ) -> None:

        if not isinstance(in_paths, list):
            in_paths = [in_paths]

        operation_name = out_folder_location.split("/")[1]
        new_attrs = {
            "path": out_folder_location + output_name,
            "shape": np.shape(self[out_folder_location]),
            "name": output_name,
        }

        if function.__module__ is None:
            new_attrs["operation name"] = "None." + function.__name__
        else:
            new_attrs["operation name"] = function.__module__ + "." + function.__name__

        if function.__module__ == "__main__":
            new_attrs["function code"] = inspect.getsource(function)

        new_attrs["operation number"] = operation_name.split("-")[0]
        new_attrs["time"] = str(datetime.datetime.now())
        new_attrs["source"] = in_paths

        self.attrs[out_folder_location] = new_attrs

    def _write_kwargs_as_attributes(
        self, path: str, func: callable, all_variables: dict[str, Any], first_kwarg: int = 1
    ) -> None:
        """write_kwargs_as_attributes
        Writes all other arguments as attributes to a datset.

        Parameters
        ----------
        path : str
            path to the dataset the attributes are written to
        func : callable
            the function from which attributes are pulled
        all_variables : dict
            all variables from the function call. To call properly, set as locals()
        first_kwarg : int, optional
            First kwarg that is to be written in, by default 1
        """
        attr_dict = {}
        if isinstance(func, types.BuiltinFunctionType):
            attr_dict["BuiltinFunctionType"] = True
        else:
            signature = inspect.signature(func).parameters
            var_names = list(signature.keys())[first_kwarg:]
            for key in var_names:

                if key in all_variables:
                    value = all_variables[key]
                elif isinstance(signature[key].default, np._globals._NoValueType):
                    value = "None"
                else:
                    value = signature[key].default

                if callable(value):
                    value = value.__name__
                elif value is None:
                    value = "None"

                try:
                    attr_dict[f"kwargs_{key}"] = value
                except RuntimeError:
                    RuntimeWarning("Attribute was not able to be saved, probably because the attribute is too large")
                    attr_dict["kwargs_" + key] = "None"

        self.attrs[path] = attr_dict

    def _generate_file_struct(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            return h5py.Dataset

        file_structure = {}
        for key, subobj in obj.items():
            subname = f"{name}/{key}"
            file_structure[key] = self._generate_file_struct(subname, subobj)

        return file_structure

    def _traverse_dict(self, dictionary, current_path, saved_path=None):
        if saved_path is None:
            saved_path = []
        if isinstance(current_path, str):
            current_path = current_path.split("/")
        if len(current_path) > 0:
            if current_path[0] not in dictionary:
                raise KeyError(
                    "Your path is not in the structure attribute. Please check that you wrote it correctly."
                )

            if isinstance(dictionary[current_path[0]], dict):
                saved_path.append(current_path[0])
                return self._traverse_dict(dictionary[current_path[0]], current_path[1:], saved_path)
            else:
                with h5py.File(self.filename, "r") as f:
                    data = f["/".join(saved_path + current_path)][()].copy()
                return data
        else:
            return tuple(dictionary.keys())

    def _path_search(self, criterion: list[str] | str, repeat: str | None = None) -> list[list[str]]:

        file_structure = utils.dict_to_list(self.structure)

        if not isinstance(criterion, list):
            criterion = [criterion]
        for i, criteria in enumerate(criterion):
            if not isinstance(criteria, list):
                criterion[i] = [criteria]
        inputs = criterion.copy()

        for i, criteria in enumerate(criterion):
            inputs[i] = []
            for crit in criteria:
                for elem in file_structure:
                    if fnmatch.fnmatch(elem, crit):
                        inputs[i].append(elem)

        if len(inputs) == 1:
            if len(inputs[0]) == 0:
                raise Exception("No Input Datafiles found!")

        list_length = [len(input) for input in inputs]
        largest_list_length = max(list_length)
        list_multiples = []
        for length in list_length:
            if largest_list_length % length != 0:
                Warning(
                    "At least one path list length is not a factor of the largest path list length. Extra file will be omitted"
                )
            list_multiples.append(largest_list_length // length)

        if repeat in ["block", "b"]:
            for i, multiple in enumerate(list_multiples):
                inputs[i] = np.repeat(inputs[i], multiple)

        if repeat in ["alt", "a"]:
            for i, multiple in enumerate(list_multiples):
                old_inputs = inputs[i]
                new_inputs = []
                for _ in multiple:
                    new_inputs.extend(old_inputs)
                inputs[i] = new_inputs

        else:
            smallest_list_length = min(list_length)
            inputs = [input[:smallest_list_length] for input in inputs]

        return inputs
