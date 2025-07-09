from typing import Optional, Union, List
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.constants import REPO_TYPES
from os.path import join, split, exists
import os
from deep_utils import AsyncDownloadUtils, DownloadUtils


def get_urls(
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        download: bool = False,
        download_path: str = ".",
        filtered_repo_files:str = None,
        resolve:str = "resolve",
cookies=None,
        )    -> List[str]:
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
            items=filtered_repo_files or [f.rfilename for f in repo_info.siblings] ,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    dl_urls = [hf_hub_url(repo_id, f) for f in filtered_repo_files]
    if resolve != "resolve":
        dl_urls = [item.replace("/resolve/",f"/{resolve}/" ) for item in dl_urls]
    if repo_type == "dataset":
        dl_urls = [item.replace("https://huggingface.co", "https://huggingface.co/datasets") for item in dl_urls]
    if download:
        download_path = join(download_path, repo_id.split("/")[1])
        print(f"Downloading to {download_path}")
        os.makedirs(download_path, exist_ok=True)
        remove_to_get_local_file_path = f"https://huggingface.co/{repo_type}s/{repo_id}/{resolve}/main/"
        DownloadUtils.download_urls(dl_urls, download_path, remove_to_get_local_file_path, cookies=cookies)
        print("Download is over, Enjoy :)")
    return dl_urls


async def aget_urls(
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        download: bool = False,
        download_path: str = ".",
        filtered_repo_files:str = None,
        )    -> List[str]:
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
    filtered_repo_files = filtered_repo_files or list(
        filter_repo_objects(
            items=[f.rfilename for f in repo_info.siblings],
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    dl_urls = [hf_hub_url(repo_id, f) for f in filtered_repo_files]
    if repo_type == "dataset":
        dl_urls = [item.replace("https://huggingface.co", "https://huggingface.co/datasets") for item in dl_urls]
    if download:
        download_path = join(download_path, repo_id.split("/")[1])
        print(f"Downloading to {download_path}")
        os.makedirs(download_path, exist_ok=True)
        remove_to_get_local_file_path = f"https://huggingface.co/{repo_type}s/{repo_id}/resolve/main/"
        if not do_async:
            DownloadUtils.download_urls(dl_urls, download_path, remove_to_get_local_file_path)
        else:
            await AsyncDownloadUtils.download_urls(dl_urls, download_path, remove_to_get_local_file_path)
        print("Download is over, Enjoy :)")
    return dl_urls
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-data", default="data.json")
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()
    # import asyncio
    from dotenv import load_dotenv
    repo_or_dataset_id = "ibrahimhamamci/CT-RATE"
    # repo_or_dataset_id = "wanglab/CT_DeepLesion-MedSAM2"
    from deep_utils import JsonUtils

    filtered_repo_files = JsonUtils.load(args.data)
    if args.reverse:
        filtered_repo_files = filtered_repo_files[::-1]
    # filtered_repo_files  =
    load_dotenv()
    login_token = os.getenv("login_token")
    cookies = "/home/aicvi/Desktop/huggingface_cookies.txt"
    from http.cookiejar import MozillaCookieJar
    jar = MozillaCookieJar(cookies)
    jar.load()
    get_urls(repo_or_dataset_id, token=login_token, repo_type="dataset", download=True,
             filtered_repo_files=filtered_repo_files, cookies = jar,
             download_path="/media/aicvi/Elements/CT/Ibrahim-CT-RATE")
    # get_urls(repo_or_dataset_id, token=login_token, repo_type="dataset", download=True, )
    # local_files = asyncio.run(get_urls(repo_or_dataset_id, token=None, repo_type="dataset", download=True))
    # print(local_files)
