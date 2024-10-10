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

#  NPDR vs PDR


classes_exp0 = {
    0: "NPDR",  # non proliferative diabetic retinopathy
    1: "PDR"    # proliferative diabetic retinopathy
}

class_remapping_exp0 = {
    0: [0],
    1: [1]
}


categories_exp0 = {
    "all": [0, 1]
}


classes_exp1 = {
    0: "NPDR",  # non proliferative diabetic retinopathy
    1: "PDR"    # proliferative diabetic retinopathy
}

class_remapping_exp1 = {
    0: [0],
    1: [1]
}

categories_exp1 = {
    "all": [0, 1]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1]), len(CLASS_INFO[1][1])]


DR_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    # todo write this check for DR
    import cv2
    import warnings
    import pathlib
    import numpy as np
    warning = 0
    warning_msg = []
    if root == None:
        root = pathlib.Path(r"C:/PhD/Segmentation/oct_semantic_pytorch/data/IACL/label_images/")
        root = pathlib.Path(r"C:\Users\Theodoros Pissas\Documents\tresorit\IACL/label_images")
    for path_to_label in root.glob('*.png'):
        i = cv2.imread(str(path_to_label))
        labels_present = np.unique(i)
        print(f'{path_to_label.stem} : {labels_present}')
        if max(labels_present) > 8:
            warnings.warn(f'invalid label found {labels_present}')
            warning += 1
            warning_msg.append(f'invalid label found {labels_present}')
    return warning_msg, warning


if __name__ == '__main__':
    label_sanity_check()
