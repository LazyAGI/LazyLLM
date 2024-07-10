import requests
import os

def get_pr_files(api_url):
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching PR files: {response.status_code}")
        return []

    return response.json()

def check_submodule_changes(pr_files, submodule_path):
    for file in pr_files:
        if submodule_path in file['filename']:
            return True
    return False

if __name__ == "__main__":
    repository = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('GITHUB_PR_NUMBER')
    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/files"

    pr_files = get_pr_files(api_url)
    submodule_path = "LazyLLM-Env"

    if check_submodule_changes(pr_files, submodule_path):
        print("::set-output name=submodule_updated::true")
    else:
        print("::set-output name=submodule_updated::false")
