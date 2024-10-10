

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

#  NPDR vs PDR


classes_exp0 = {
    0: "NORMAL",  # control
    1: "AMD",    # age-related macular degeneration
    2: "DR",    # diabetic retinopathy
    3: "MH",    # macular hole
    4: "CSC",    # central serous chorioretinopathy
}

class_remapping_exp0 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4]
}


categories_exp0 = {
    "diseases": [1, 2, 3, 4],
    "healthy": [0]
}


classes_exp1 = {
    0: "NORMAL",  # control
    1: "AMD",    # age-related macular degeneration
    2: "DR",    # diabetic retinopathy
    3: "MH",    # macular hole
    4: "CSC",    # central serous chorioretinopathy
}

class_remapping_exp1 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4]
}

categories_exp1 = {
    "diseases": [1, 2, 3, 4],
    "healthy": [0]
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

NUM_CLASSES = [len(CLASS_INFO[0][1]), len(CLASS_INFO[1][1])]


OCTID_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES, NUM_CLASSES=NUM_CLASSES)


def label_sanity_check(root=None):
    # todo write this check for OCTID

    # Optical Coherence Tomography Image Database (OCTID)	India	Cirrus HD-OCT machine (Carl Zeiss Meditec)
    #                     Train Val Test Total
    # 			NORMAL: 	115	29	62	 206 (35.83%)
    # 			AMD:		56	15	32	 103 (17.91%)
    # 			DR:			30	8	17	 55 (9.57%)
    # 			MH:     	59	15	33	 107 (18.61%)
    # 			CSC:	    58	15	31	 104 (18.09%)
    # 			in total:	318	82	175	 575 (100%)

    #                     Train Val Test Total
    # 			NORMAL: 	115	29	62	 206  (36.0%)
    # 			AMRD:		30	8	17	 55    (9.6%)
    # 			DR:			60  15  32 	 107  (18.7%)
    # 			MH:     	60	10	32	 102  (17.8%)
    # 			CSR:	    60	10	32	 102  (17.8%)
    # 			in total:	318	82	175	 572 (100.0%)

    pass


if __name__ == '__main__':
    label_sanity_check()
