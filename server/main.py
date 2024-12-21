from io import BytesIO
import torch
import torchvision.transforms as TF
from fastapi.responses import StreamingResponse
from vqvae import baseline

from core import create_app

device = "cpu"
model = baseline()
model.load_state_dict(torch.load("data/weights.pth",
                      map_location=device)["model"])
model.eval()


def grade_submission(predictions):
    return 0.0


app = create_app(grading_fn=grade_submission)


@app.get("/generate-images/")
async def generate_images(codes: str):
    codes = [int(i) for i in codes.split()[2:-2]]
    codes = torch.tensor(codes).view(1, 8, 8)

    with torch.no_grad():
        z = model[1].lookup(codes.to(device))
        imgs = model[2](z)
    img_t = TF.functional.to_pil_image(imgs[0].cpu().clamp(0, 1))
    buf = BytesIO()
    img_t.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
