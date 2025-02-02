from multiprocessing import Process
import datetime
import time
import os
from io import BytesIO
import torch
import torchvision.transforms as TF
from fastapi import Request
from fastapi.responses import StreamingResponse
from vqvae import baseline
from grader import ConvGrader
from core import create_app, SubmissionError

device = "cpu"
model = baseline(256)
ckpt = torch.load("model-hd.pth", map_location=device, mmap=True)["model"]
print(torch.nn.ModuleDict({"model": model}).load_state_dict(ckpt, strict=False))
torch.nn.ModuleDict({"ema": model[2]}).load_state_dict(ckpt, strict=False)
model.eval()
del ckpt

grader = ConvGrader(512, 15)
grader.load_state_dict(torch.load("data/grader.pth", map_location=device, mmap=True))
grader.eval()


def load_ref():
    dataset = {}
    with open("data/dataset-test.txt") as f:
        lines = [l.split(" ") for l in f.readlines()]

    for line in lines:
        dataset[line[0]] = [int(x) for x in line[1:]]
    return dataset


test_ref = load_ref()


def batched(iterable, n=1):
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(iterable))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


def grade_submission(predictions):
    for ref_file in test_ref.keys():
        if ref_file not in predictions:
            raise SubmissionError(f"Missing predictions for {ref_file}")

    for pred_file in predictions.keys():
        if pred_file not in test_ref:
            raise SubmissionError(f"Unexpected predictions for {pred_file}")

    all_tokens = []
    all_cls = []
    for tag, codes in predictions.items():
        if not isinstance(codes, list):
            raise SubmissionError(f"Invalid submission format for {tag}")

        if not all(c1 == c2 for c1, c2 in zip(codes, test_ref[tag])):
            raise SubmissionError(f"Invalid prefix for {tag}")

        if not len(codes) == 257:
            raise SubmissionError(
                (f"Invalid length for {tag} ({len(codes)},")
                + (f"expected 257): {codes}")
            )

        all_tokens.append(codes[1:])
        all_cls.append(codes[0])

    correct = 0
    for batch in batched(zip(all_tokens, all_cls), 32):
        batch_tokens, batch_cls = zip(*batch)
        with torch.no_grad():
            pred = grader(torch.tensor(batch_tokens).to(device))
        correct += (pred.argmax(dim=1) == torch.tensor(batch_cls)).sum().item()
    sum = len(all_cls)
    return correct / sum


def generate_image(codes):
    codes = torch.tensor(codes).view(1, 16, 16)

    with torch.no_grad():
        z = model[1].lookup(codes.to(device))
        imgs = model[2](z).clamp(0.001, 0.999)
    return TF.functional.to_pil_image(imgs[0].cpu().clamp(0, 1))


def save_to_gallery(student, submission, file_date=None):
    import random

    if file_date is None:
        file_date = time.time()
    file_date_str = datetime.datetime.fromtimestamp(file_date).strftime("%m%d")
    files = random.choices(list(submission.keys()), k=8)
    for file in files:
        img = generate_image(submission[file][1:])
        img.save(f"static/gallery/{student.name}-{file_date_str}-{file}.png")


def gen_some_images():
    import json
    import os
    from types import SimpleNamespace

    # find all json files in submissions/test/
    for root, _, files in os.walk("submissions/test/"):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file)) as f:
                        submission = json.load(f)
                        file_date = os.path.getmtime(os.path.join(root, file))
                except json.JSONDecodeError:
                    print(f"Error reading {file}")
                    continue

                try:
                    save_to_gallery(
                        SimpleNamespace(name=root.split("/")[-1]), submission, file_date
                    )
                except Exception as e:
                    print(f"Error processing {file}: {e}")


os.makedirs("static/gallery", exist_ok=True)

app = create_app(grading_fn=grade_submission, after_submission=save_to_gallery)


@app.on_event("startup")
def startup():
    p = Process(target=gen_some_images)
    p.start()


@app.get("/gallery")
def gallery(request: Request):
    return app.templates.TemplateResponse(
        "gallery.html", {"request": request, "files": os.listdir("static/gallery")}
    )


@app.get("/generate-images/")
async def generate_images(codes: str):
    codes = [int(i) for i in codes.split()[2:]]
    img_t = generate_image(codes)
    buf = BytesIO()
    img_t.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
