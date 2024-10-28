import io
import random
import requests
import json
import os

try:
    from google.colab import _message

    USE_COLAB = True
except ImportError:
    USE_COLAB = False


URL = "https://leaderboard.vermeille.fr"


def _get_current_notebook():
    try:
        notebook_json = _message.blocking_request("get_ipynb", timeout_sec=5)
        if notebook_json and "ipynb" in notebook_json:
            nb_json = notebook_json["ipynb"]
            lines = []
            for cell in nb_json["cells"]:
                if cell["cell_type"] == "code":
                    lines.append("#############")
                    lines.append("### CELL ####")
                    lines.append("#############")
                    lines.extend(cell["source"])
            return "".join(lines)
        else:
            print("Can't get notebook JSON")
            return None
    except Exception as e:
        print(f"Cant get notebook JSON: {e}")
        return None


def _current_python_to_zip():
    """
    Add all the python files in the current directory to a zip file
    in memory
    """
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED, False) as zipf:
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".py") or file.endswith(".ipynb"):
                    zipf.write(os.path.join(root, file))
    buffer.seek(0)
    return buffer


def generate_token(name):
    url = f"{URL}/generate_token"
    data = {"name": name}
    token = os.getenv("LEADERBOARD_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()


def submit_training(accuracy, loss, hyperparameters, tag):
    token = os.getenv("LEADERBOARD_TOKEN")
    url = f"{URL}/train_submission"
    headers = {"Authorization": f"Bearer {token}"}

    data = {
        "accuracy": str(accuracy),
        "loss": str(loss),
        "hyperparameters": json.dumps(hyperparameters),
        "pid": str(os.getpid()),
        "tag": tag,
    }

    if USE_COLAB:
        code = _get_current_notebook()
        if code:
            files = {"code_zip": code}
    else:
        files = {"code_zip": _current_python_to_zip()}

    response = requests.post(url, headers=headers, data=data, files=files)
    response.raise_for_status()
    return response.json()


def submit_test(predictions):
    token = os.getenv("LEADERBOARD_TOKEN")
    url = f"{URL}/test_submission"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"predictions": json.dumps(predictions)}

    print(data)
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()


def view_leaderboard():
    url = f"{URL}/leaderboard"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_code(submission_id, save_path):
    url = f"{URL}/download_code/{submission_id}"
    token = os.getenv("LEADERBOARD_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition:
        filename = content_disposition.split("filename=")[1].strip('"')
    else:
        filename = f"submission_{submission_id}.zip"

    full_path = os.path.join(save_path, filename)
    with open(full_path, "wb") as f:
        f.write(response.content)

    return full_path


# Usage examples
if __name__ == "__main__":
    # Generate a token
    token_response = generate_token("Teacher")
    os.environ["LEADERBOARD_TOKEN"] = token_response["token"]
    print("Token Response:", token_response)
    token = token_response["token"]

    # Submit training data
    accuracy = 0.95 + 0.05 * random.random()
    loss = 0.05
    hyperparameters = {"learning_rate": 0.001}
    tag = "First run"
    training_response = submit_training(accuracy, loss, hyperparameters, tag)
    print("Training Submission Response:", training_response)

    # Submit test predictions
    preds = {
        "img0.png": [random.randint(0, 2)],
        "img1.png": [random.randint(0, 2)],
        "img2.png": [2],
        "img3.png": [random.randint(0, 4)],
        "img4.png": [random.randint(2, 5)],
    }
    try:
        test_response = submit_test(preds)
        print("Test Submission Response:", test_response)
    except Exception as e:
        print(e)
        print("Error submitting test predictions")

    # View leaderboard
    leaderboard = view_leaderboard()
    print("Leaderboard:", json.dumps(leaderboard, indent=2))

    # Download code submission
    submission_id = training_response.get("submission_id")
    save_path = "./downloads"
    os.makedirs(save_path, exist_ok=True)
    downloaded_file = download_code(submission_id, save_path)
    print(f"Downloaded code to {downloaded_file}")
