

# todo - add the following to the dataset info:
from typing import Any


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

# todo STAGE just add something as a placeholder so that this exists. Check later if needs to be thought out more.


classes_exp0 = {
    0: "NO",  # control
    1: "AMD",    # age-related macular degeneration
    2: "DME",    # diabetic retinopathy
    3: "ERM",    # epiretinal membrane
    4: "RAO",    # retinal artery occlusion
    5: "RVO",    # retinal vein occlusion
    6: "VID",    # Vitreomacular Interface Disease
}

class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
}


categories_exp0 = {
    "diseases": [1, 2, 3, 4, 5, 6],
    "healthy": [0]
}

# go up to 39
classes_exp1 = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "11", 12: "12",
    13: "13", 14: "14", 15: "15", 16: "16", 17: "17", 18: "18", 19: "19", 20: "20", 21: "21", 22: "22", 23: "23",
    24: "24", 25: "25", 26: "26", 27: "27", 28: "28", 29: "29", 30: "30", 31: "31", 32: "32", 33: "33", 34: "34",
    35: "35", 36: "36", 37: "37", 38: "38", 39: "39"
}

class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6]
}

categories_exp1 = {
    "diseases": [1, 2, 3, 4, 5, 6],
    "healthy": [0]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1]), len(CLASS_INFO[1][1])]


STAGE_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    pass


if __name__ == '__main__':
    label_sanity_check()
