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
    1: "RNFL_GCL_IPL",  # Retinal Nerve Fibre Layer + Ganglion Cell Layer + Inner Plexiform Layer
    2: "INL_to_OS",     # Inner nuclear layer to Outer photoreceptor segments
    3: "RPE",           # Retinal Pigment Epithelium
    4: "CHOR",          # Choroidal Stroma
    5: "PED",           # pigment epithelial detachment
    6: "SRF",           # subretinal fluid and subretinal hyperreflective material (marked jointly as SRF)
    7: "IRF"            # intraretinal fluid (IRF)
}


class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7]
}

categories_exp0 = {
    "Background"    :[0],
    "RNFL_GCL_IPL"  :[1],
    "INL_to_OS"     :[2],
    "RPE"           :[3],
    "CHOR"          :[4],
    "PED"           :[5],
    "SRF"           :[6],
    "IRF"           :[7]
}

classes_exp1 = {
    0: "Background",
    1: "RNFL_GCL_IPL",  # Retinal Nerve Fibre Layer + Ganglion Cell Layer + Inner Plexiform Layer
    2: "INL_to_OS",     # Inner nuclear layer to Outer photoreceptor segments
    3: "RPE",           # Retinal Pigment Epithelium
    4: "CHOR",          # Choroidal Stroma
    5: "PED",           # pigment epithelial detachment
    6: "SRF",           # subretinal fluid and subretinal hyperreflective material (marked jointly as SRF)
    7: "IRF"            # intraretinal fluid (IRF)
}


class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7]
}

categories_exp1 = {
    "Background": [0],
    "Layers": [1, 2, 3, 4],
    "Fluid": [5, 6, 7],
    "PED": [5],
    "SRF": [6],
    "IRF": [7]
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


AROI_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    import cv2
    import warnings
    import pathlib
    import numpy as np
    warning = 0
    warning_msg = []
    if root == None:
        root = pathlib.Path(r"C:/PhD/Segmentation/oct_semantic_pytorch/data/AROI/label_images/")
        root = pathlib.Path(r"C:\Users\Theodoros Pissas\Documents\tresorit\AROI/label_images")
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