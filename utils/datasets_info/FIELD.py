from typing import Any


DATA_SPLITS = [  # Pre-defined splits of the videos, to be used generally
    [[2], [1]],  # Split 0: debugging
    [[2, 3, 5, 6, 10, 13, 14, 16, 17, 19, 20, 21, 22, 23, 26, 27, 29, 33, 36, 40], [1, 9, 24, 30, 37]],  # Split 1
    [[2, 3, 5, 6, 10, 13, 14, 16, 17, 19, 20, 21, 22, 23, 26, 27, 29, 33, 36, 40, 9, 24, 30, 37], [1]],  # Split 2
]

class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5]
}
classes_exp0 = {
    0: 'Background',
    1: 'Upper IRF',
    2: 'Middle IRF',
    3: 'SRF',
    4: 'Top layer',
    5: 'Bottom layer'
}

categories_exp0 = {
    "Background": [0],
    "Fluids": [1, 2, 3],
    "IRF": [1, 2],
    "SRF": [3]
}

class_remapping_exp1 = {
        0: [0],
        1: [1, 2, 3],
        255: [4, 5]
    }
classes_exp1 = {
    0: "Background",
    1: "Fluid",
    255: "Ignore"
}

categories_exp1 = {
    "Background": [0],
    "Fluids": [1]
}

class_remapping_exp2 = {
    0: [0],
    1: [1, 2],
    2: [3],
    255: [4, 5]
}
classes_exp2 = {
    0: "Background",
    1: "IRF",
    2: "SRF",
    255: "Ignore"
}

categories_exp2 = {
    "Background": [0],
    "Fluids": [1, 2],
    "IRF": [1],
    "SRF": [2]
}

class_remapping_exp3 = {
    0: [0],
    1: [1, 2],
    2: [3],
    3: [4],
    4: [5]
}
classes_exp3 = {
    0: 'Background',
    1: 'IRF',
    2: 'SRF',
    3: 'Top layer',
    4: 'Bottom layer'
}

categories_exp3 = {
    "Background": [0],
    "Fluids": [1, 2],
    "IRF": [1],
    "SRF": [2],
    "Top layer": [3],
    "Bottom layer": [4],
}

class_remapping_exp4 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    255: [4, 5]
}
classes_exp4 = {
    0: 'Background',
    1: 'uIRF',
    2: 'mIRF',
    3: 'SRF',
    255: "Ignore"
}

categories_exp4 = {
    "Background": [0],
    "Fluids": [1, 2, 3],
    "IRF": [1, 2],
    "uIRF": [1],
    "mIRF": [2],
    "SRF": [3]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1],
    [class_remapping_exp2, classes_exp2, categories_exp2],
    [class_remapping_exp3, classes_exp3, categories_exp3],
    [class_remapping_exp4, classes_exp4, categories_exp4]
]


CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())],
               [CLASS_INFO[2][1][key] for key in sorted(CLASS_INFO[2][1].keys())],
               [CLASS_INFO[3][1][key] for key in sorted(CLASS_INFO[3][1].keys())],
               [CLASS_INFO[4][1][key] for key in sorted(CLASS_INFO[4][1].keys())]]


# todo fix oversampling resets or remov


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


FIELD_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, DATA_SPLITS=DATA_SPLITS)
