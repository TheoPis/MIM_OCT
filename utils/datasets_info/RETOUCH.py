from typing import Any
import numpy as np

DATA_SPLITS = [  # Pre-defined splits of the videos, to be used generally
    [[1, 26, 49], [3, 35, 50]],  # Split 0: debugging
    [[1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17,
      19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 37, 38, 41, 42, 44, 45, 47, 48, 49, 52, 53, 54, 55,
       56, 57, 59, 61, 62, 64, 65, 66, 67, 68, 69], [3, 9, 13, 14,
                                                     18, 21, 35, 36,
                                                     39, 40, 43, 46, 50, 51,
                                                     58, 60, 63, 70]],  # Split 1

    [[], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]],  # split 2 test

    [[1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 19, 20, 22, 23, 24, 25], [3, 9, 13, 14, 18, 21]],  # cirrus split 2
    [[26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 41, 42, 44, 45, 47, 48], [35, 36, 39, 40, 43, 46]],  # spectr split 3
    [[49, 52, 53, 54, 55, 56, 57, 59, 61, 62, 64, 65, 66, 67, 68, 69], [50, 51, 58, 60, 63, 70]]  # topcon split 4

]
class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
}
classes_exp0 = {
    0: 'Background',
    1: 'IRF',
    2: 'SRF',
    3: 'PED',
}

categories_exp0 = {
    "Background": [0],
    "Fluids": [1, 2, 3],
    "IRF": [1],
    "SRF": [2],
    "PED": [3]
}

class_remapping_exp1 = {
    255: [4],
    0: [0],
    1: [1],
    2: [2],
    3: [3],
}
classes_exp1 = {
    255: "ignore",
    0: 'Background',
    1: 'IRF',
    2: 'SRF',
    3: 'PED',
}

categories_exp1 = {
    "Background": [0],
    "Fluids": [1, 2, 3],
    "IRF": [1],
    "SRF": [2],
    "PED": [3]
}

CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1],
]


CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]
               ]

NUM_CLASSES = [len(CLASS_INFO[0][1])-1 if 255 in CLASS_INFO[0][1] else len(CLASS_INFO[0][1]),
               len(CLASS_INFO[1][1])-1 if 255 in CLASS_INFO[1][1] else len(CLASS_INFO[1][1])]



OVERSAMPLING_PRESETS = {
    'default': [
        [3, 5, 7],            # Experiment 1
        [7, 8, 15, 16],       # Experiment 2
        [19, 20, 22, 24]      # Experiment 3
    ],
    'rare': [  # Same classes as 'rare' category for mIoU metric
        [2],                                            # Experiment 1
        [16, 10, 9, 12, 14],                            # Experiment 2
        [24, 20, 21, 22, 18, 23, 19, 16, 12, 11, 14]    # Experiment 3
    ]
}


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


def scanner2int(scanner_str):
    if scanner_str == 'Cirrus':
        return 1
    elif scanner_str == 'Spectralis':
        return 2
    elif scanner_str == 'Topcon':
        return 3
    else:
        raise ValueError('scanners in RETOUCH dataset are [Cirrus, Spectralis, Topcon] instead got {}')


RETOUCH_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, DATA_SPLITS=DATA_SPLITS,
                        scanner2int=scanner2int, NUM_CLASSES=NUM_CLASSES)