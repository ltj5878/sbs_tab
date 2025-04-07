import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def validate_resolution(
        width: int, height: int, input_format: str
) -> Tuple[int, int]:
    """验证输入视频分辨率是否符合格式要求"""
    if input_format == "sbs":
        if width % 2 != 0:
            raise ValueError(
                f"无效的SBS输入：宽度必须为偶数（当前值：{width}）"
            )
        return width // 2, height
    elif input_format == "tab":
        if height % 2 != 0:
            raise ValueError(
                f"无效的TAB输入：高度必须为偶数（当前值：{height}）"
            )
        return width, height // 2
    raise ValueError(f"Invalid input format: {input_format}")


def sbs_to_tab(frame: np.ndarray) -> np.ndarray:
    """将 SBS 帧转换为 TAB 格式"""
    h, w = frame.shape[:2]
    left = frame[:, : w // 2]
    right = frame[:, w // 2:]
    return np.vstack((left, right))


def tab_to_sbs(frame: np.ndarray) -> np.ndarray:
    """将 TAB 帧转换为 SBS 格式"""
    h, w = frame.shape[:2]
    top = frame[: h // 2]
    bottom = frame[h // 2:]
    return np.hstack((top, bottom))


def get_output_dimensions(
        input_width: int, input_height: int, in_fmt: str, out_fmt: str
) -> Tuple[int, int]:
    """计算输出视频尺寸"""
    if in_fmt == "sbs" and out_fmt == "tab":
        return (input_width // 2, input_height * 2)
    if in_fmt == "tab" and out_fmt == "sbs":
        return (input_width * 2, input_height // 2)
    raise ValueError("格式一致时无需转换")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在SBS和TAB 3D视频格式之间转换"
    )
    parser.add_argument("input", type=str, help="输入视频文件路径")
    parser.add_argument("output", type=str, help="输出视频文件路径")
    parser.add_argument(
        "-i",
        "--input-format",
        required=True,
        choices=["sbs", "tab"],
        help="输入视频格式（sbs/tab）",
    )
    parser.add_argument(
        "-o",
        "--output-format",
        required=True,
        choices=["sbs", "tab"],
        help="输出视频格式（sbs/tab）",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="视频编解码器（默认：mp4v）",
    )

    try:
        args = parser.parse_args()

        # 输入验证
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件 {args.input} 不存在")

        output_path = Path(args.output)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"目录 {output_dir} 没有写入权限")

        if args.input_format == args.output_format:
            raise ValueError("输入输出格式不能相同")

        # 视频处理
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("打开输入视频失败")

        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 分辨率验证
        validate_resolution(input_width, input_height, args.input_format)
        output_width, output_height = get_output_dimensions(
            input_width, input_height, args.input_format, args.output_format
        )

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (output_width, output_height),
        )
        if not out.isOpened():
            raise RuntimeError("创建视频写入器失败")

        # 帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.output_format == "tab":
                processed = sbs_to_tab(frame)
            else:
                processed = tab_to_sbs(frame)

            out.write(processed)

        print(f"转换成功完成：{output_path}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if "cap" in locals():
            cap.release()
        if "out" in locals():
            out.release()


if __name__ == "__main__":
    main()
