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
    0: "vitreous",
    1: "ILM",  # Inner limiting membrane
    2: "OPL",  # Outer plexiform layer (Helnes fiber layer)
    3: "IS_OS",  # IS/OS junction
    4: "IB_RPE",  # Inner boundary RPE
    5: "OBRPE",    # Outer boundary RPE
}


class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5]
}

categories_exp0 = {
    "background":[0],
    "layers": [1, 2, 3, 4],

}

classes_exp1 = {
    0: "vitreous",
    1: "ILM",  # Inner limiting membrane
    2: "OPL",  # Outer plexiform layer (Helnes fiber layer)
    3: "IS_OS",  # IS/OS junction
    4: "IB_RPE",  # Inner boundary RPE
    5: "OBRPE",    # Outer boundary RPE
}

class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5]
}

categories_exp1 = {
    "vitreous": [0],
    'ILM': [1],
    'OPL': [2],
    'IS_OS': [3],
    'IB_RPE': [4],
    'OBRPE': [5],
    "Layers": [1, 2, 3, 4],
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


OCT5K_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)

def label_sanity_check(root=None):
    pass

if __name__ == '__main__':
    label_sanity_check()