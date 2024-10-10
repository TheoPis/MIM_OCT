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


classes_exp0 = {
    0: "Background",
    1: "arteries",
    2: "veins"
}


class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2]
}

categories_exp0 = {
    "Background": [0],
    "arteries": [1],
    "veins": [2],
    "vessels": [1, 2]
}

classes_exp1 = {
    0: "Background",
    1: "arteries",
    2: "veins"
}

class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2]
}

categories_exp1 = {
    "Background": [0],
    "arteries": [1],
    "veins": [2],
    "vessels": [1, 2]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [
    [CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
    [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]
               ]

NUM_CLASSES = [len(CLASS_INFO[0][1])-1 if 255 in CLASS_INFO[0][1] else len(CLASS_INFO[0][1]),
               len(CLASS_INFO[1][1])-1 if 255 in CLASS_INFO[1][1] else len(CLASS_INFO[1][1])]


RAVIR_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check():
    print(RAVIR_INFO.CLASS_NAMES)
    print(RAVIR_INFO.NUM_CLASSES)


if __name__ == '__main__':
    label_sanity_check()
