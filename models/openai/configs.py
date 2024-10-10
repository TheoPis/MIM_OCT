import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Union

from tqdm import tqdm


hf_hub_download = None
_has_hf_hub = False

MODEL_URLS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "vit_base": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


pretrained_configs = {
    "RN50": {
        "embed_dim": 1024,
        "vision_cfg": {
            "image_size": 224,
            "layers": [
                3,
                4,
                6,
                3
            ],
            "width": 64,
            "patch_size": None
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12
        }
    },
    "RN101": {
        "embed_dim": 512,
        "vision_cfg": {
            "image_size": 224,
            "layers": [
                3,
                4,
                23,
                3
            ],
            "width": 64,
            "patch_size": None
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12
        }
    },
    "vit_base": {
        "embed_dim": 512,
        "vision_cfg": {
            "image_size": 224,
            "layers": 12,
            "width": 768,
            "patch_size": 16
        },
        "text_cfg": {
            "context_length": 77,
                "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12
        }
    },
    "vit_large": {
        "embed_dim": 768,
        "vision_cfg": {
            "image_size": 320,
            "layers": 24,
            "width": 1024,
            "patch_size": 16
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 49408,
            "width": 768,
            "heads": 12,
            "layers": 12
        }
    }
}


def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target