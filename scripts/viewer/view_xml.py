import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MuJoCo XML to PNG")
    parser.add_argument("xml_path", type=str, help="Path to MuJoCo XML file (scene.xml or model.xml)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (default: image/xml/<stem>.png)")
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640, XMLのoffscreenバッファ上限に注意)")
    parser.add_argument("--height", type=int, default=480, help="Image height (default: 480)")
    parser.add_argument("--camera", type=str, default=None, help="Camera name to use (default: free camera)")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = Path(args.xml_path)

    if not xml_path.exists():
        print(f"[ERROR] XML not found: {xml_path}")
        return 1

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.list_cameras:
        print("Available cameras:")
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  [{i}] {name}")
        return 0

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)

    if args.camera is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cam_id == -1:
            print(f"[ERROR] Camera not found: {args.camera}")
            return 1
        renderer.update_scene(data, camera=args.camera)
    else:
        renderer.update_scene(data)

    img = renderer.render()

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_dir = Path("image/xml")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{xml_path.stem}.png"

    Image.fromarray(img).save(out_path)
    print(f"saved: {out_path}  {img.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
