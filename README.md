# transformers_hub_download_urls

In case, you don't have stable internet connection and want to download huge models from huggingface like stable diffusion, you can use this simple function to create download links for all the files in the model.


How to use it:
```python
from typing import Optional, Union, List
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.constants import REPO_TYPES


def get_urls(
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
) -> List[str]:
    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")

    _api = HfApi()
    repo_info = _api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    assert repo_info.sha is not None, "Repo info returned from server must have a revision sha."
    filtered_repo_files = list(
        filter_repo_objects(
            items=[f.rfilename for f in repo_info.siblings],
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    dl_urls = [hf_hub_url(repo_id, f) for f in filtered_repo_files]
    return dl_urls
```
Get your token from `hugginface` page, then run the following code:
```python
login_token = "login-token-get-from-huggingfacepage"
urls = get_urls("runwayml/stable-diffusion-v1-5", token=login_token)
print(urls)
```

The output will be a list of download links than can be individually downloaded!
```python
['https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/.gitattributes', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/README.md', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/safety_checker/config.json', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/safety_checker/model.safetensors',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/safety_checker/pytorch_model.bin', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/scheduler_config.json', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.safetensors',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt', 
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-inference.yaml',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/config.json',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin',
 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.safetensors']
```

