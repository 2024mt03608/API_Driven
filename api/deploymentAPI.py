# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# URL - https://app.prefect.cloud/api/docs

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
PREFECT_API_KEY = os.getenv("PREFECT_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID")
GOOD_RUN = os.getenv("GOOD_RUN")
FAILED_RUN = os.getenv("FAILED_RUN")

# Validate that all required environment variables are set
if not all([PREFECT_API_KEY, ACCOUNT_ID, WORKSPACE_ID, DEPLOYMENT_ID, GOOD_RUN, FAILED_RUN]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Correct API URL to get deployment details
PREFECT_WORKSPACE_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}


def get_details(url, request_headers):
    response = requests.get(url, headers=request_headers)

    if response.status_code == 200:
        deployment_info = response.json()
        print(deployment_info)
        return deployment_info

    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
    return None



print(f"Deployment Details:")
DEPLOYMENT_API_URL = f"{PREFECT_WORKSPACE_API_URL}/deployments/{DEPLOYMENT_ID}"
get_details(DEPLOYMENT_API_URL, headers)

print(f"Passed Flow Run Details:")
FLOW_RUNS_API: str = f"{PREFECT_WORKSPACE_API_URL}/flow_runs/{GOOD_RUN}"
get_details(FLOW_RUNS_API, headers)

print(f"Failed Flow Run Details:")
FLOW_RUNS_API_2: str = f"{PREFECT_WORKSPACE_API_URL}/flow_runs/{FAILED_RUN}"
get_details(FLOW_RUNS_API_2, headers)

