import os
import requests
import argparse

def get_changed_files(api_url, headers):
    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching PR files: {response.status_code}")
        print(response.json())
        return []

    return response.json()

def check_submodule_changes(pr_files, submodule_path):
    for file in pr_files:
        if file['filename'].startswith(submodule_path):
            return True
    return False

def main(repository, pr_number):
    api_url = f"https://api.github.com/repos/{repository}/pulls/{pr_number}/files"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    pr_files = get_changed_files(api_url, headers)
    if not pr_files:
        print("::set-output name=submodule_updated::false")
        return

    submodule_path = "LazyLLM-Env"
    if check_submodule_changes(pr_files, submodule_path):
        print("::set-output name=submodule_updated::true")
    else:
        print("::set-output name=submodule_updated::false")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if submodule has been updated in a pull request.')
    parser.add_argument('--repository', type=str, required=True, help='GitHub repository (e.g., owner/repo)')
    parser.add_argument('--pr_number', type=int, required=True, help='Pull request number')
    args = parser.parse_args()

    main(args.repository, args.pr_number)
