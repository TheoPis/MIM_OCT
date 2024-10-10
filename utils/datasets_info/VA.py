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

# VA


classes_exp0 = {
    0: "Dummy1",  # control
    1: "Dummy2",    # age-related macular degeneration
    }

class_remapping_exp0 = {
    0: [0],
    1: [1]
}


categories_exp0 = {
    "dummy_1": [1],
    "dummy_2": [0]
}


classes_exp1 = {
    0: "Dummy1",  # control
    1: "Dummy2"   # age-related macular degeneration
    }

class_remapping_exp1 = {
    0: [0],
    1: [1]
}

categories_exp1 = {
    "diseases": [1],
    "healthy": [0]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1]), len(CLASS_INFO[1][1])]


VA_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    # todo write this check for OCTID
    pass


if __name__ == '__main__':
    label_sanity_check()
