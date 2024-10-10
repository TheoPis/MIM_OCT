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


# Note: We have excluded the class: Inner photoreceptor segments (IS). which is located slightly above OS.
# Difficult to discriminate the two layers (IS and OCT), especially in iOCT.
# So, we incorporate this layer to the OS. Labelmaps are saved using this assumption.

classes_exp0 = {
    0: "Background",
    1: "RNFL",          # Retinal Nerve Fibre Layer
    2: "GCL_IPL",       # Ganglion Cell Layer + Inner Plexiform Layer
    3: "INL",           # Inner nuclear layer
    4: "OPL",           # Outer plexiform layer
    5: "ONL",           # Outer nuclear layer
    6: "OS",            # Outer photoreceptor segments
    7: "RPE",           # Retinal Pigment Epithelium
    8: "CHOR"           # Choroidal Stroma
}

class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8]
}

categories_exp0 = {
    "Background":[0],
    "RNFL"      :[1],
    "GCL_IPL"   :[2],
    "INL"       :[3],
    "OPL"       :[4],
    "ONL"       :[5],
    "OS"        :[6],
    "RPE"       :[7],
    "CHOR"      :[8]
}

classes_exp1 = {
    255: "ignore",
    0: "Background",
    1: "RNFL",          # Retinal Nerve Fibre Layer
    2: "GCL_IPL",       # Ganglion Cell Layer + Inner Plexiform Layer
    3: "INL",           # Inner nuclear layer
    4: "OPL",           # Outer plexiform layer
    5: "ONL",           # Outer nuclear layer
    6: "OS",            # Outer photoreceptor segments
    7: "RPE",           # Retinal Pigment Epithelium
    8: "CHOR"           # Choroidal Stroma
}

class_remapping_exp1 = {
    255: [9],
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8]
}

categories_exp1 = {
    "Background":[0],
    "RNFL"      :[1],
    "GCL_IPL"   :[2],
    "INL"       :[3],
    "OPL"       :[4],
    "ONL"       :[5],
    "OS"        :[6],
    "RPE"       :[7],
    "CHOR"      :[8]
}


#



CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1])-1 if 255 in CLASS_INFO[0][1] else len(CLASS_INFO[0][1]),
               len(CLASS_INFO[1][1])-1 if 255 in CLASS_INFO[1][1] else len(CLASS_INFO[1][1])]

OCTIR_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)
