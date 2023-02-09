from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
from typing import List, Optional
import time
import numpy as np
import onnx
from onnx.utils import Extractor
import torch
import tqdm

import furiosa.quantizer.frontend.onnx
import furiosa.quantizer_experimental  # type: ignore[import]
from furiosa.quantizer_experimental import CalibrationMethod, Calibrator, Graph
import furiosa.runtime.session
from ultralytics import YOLO

NUM_TESTING = 50
MILLI_SEC = 1000

def main(argv: Optional[List[str]] = None) -> int:
    # pylint: disable=too-many-locals

    if argv is None:
        argv = sys.argv

    parser = build_argument_parser()
    args = parser.parse_args(argv[1:])

    # https://github.com/ultralytics/ultralytics/blob/main/README.md?plain=1#L93-L105
    model = YOLO(f"{args.model_name}.pt")
    model.export(format="onnx", opset=args.opset)

    onnx_model = onnx.load_model(f"{args.model_name}.onnx")
    onnx_model = extract_model(onnx_model)

    onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)
    onnx.checker.check_model(onnx_model, full_check=True)

    optimized_onnx_model = furiosa.quantizer.frontend.onnx.optimize_model(onnx_model)
    onnx.save_model(optimized_onnx_model, f"{args.model_name}-opt.onnx")
    onnx.checker.check_model(optimized_onnx_model, full_check=True)
    optimized_onnx_model = optimized_onnx_model.SerializeToString()

    calibrator = Calibrator(optimized_onnx_model, CalibrationMethod.MIN_MAX)

    for _ in tqdm.tqdm(range(5), desc="Calibrating", unit="image", mininterval=0.5):
        calibrator.collect_data([[np.ones(shape=(1, 3, 640, 640), dtype=np.float32)]])
    ranges = calibrator.compute_range()

    graph = furiosa.quantizer_experimental.quantize(optimized_onnx_model, ranges)

    rng = np.random.default_rng()
    elapsed_times = []
    with furiosa.runtime.session.create(bytes(graph)) as session:
        for _ in tqdm.tqdm(range(NUM_TESTING), desc="Testing", unit="image", mininterval=0.5):
            pseudo_image = rng.standard_normal(size=(1, 3, 640, 640), dtype=np.float32)
            start = time.time()
            session.run(pseudo_image).numpy()
            elapsed_times.append(time.time() - start)
    print(f"min latency: {min(elapsed_times) * MILLI_SEC: .6f} ms")
    print(f"max latency: {max(elapsed_times) * MILLI_SEC: .6f} ms")
    print(f"avg latency: {sum(elapsed_times) / NUM_TESTING * MILLI_SEC: .6f} ms")
    print(f"qps(queries per second): {NUM_TESTING / sum(elapsed_times): .6f} images/s")


def build_argument_parser() -> ArgumentParser:
    """Build a parser for command-line arguments."""
    parser = ArgumentParser(
        description="End-to-End Testing for ConvNeXt-B",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_name",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOV8 model name",
    )
    parser.add_argument(
        "--opset",
        help="specify an ONNX opset number to use",
        default=furiosa.quantizer.frontend.onnx.__OPSET_VERSION__,
        type=int,
    )
    return parser


def extract_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Cut off the pre/post-processing components."""
    if torch.__version__ < (1, 13, 0):  # type: ignore[operator]
        raise ImportError(f"torch.__version__ >= (1, 13, 0) is required: {torch.__version__}")

    input_to_shape = (("images", (1, 3, 640, 640)),)  # between Concat_91 and Mul_93
    output_to_shape = (
        (
            "/model.22/Reshape_2_output_0",
            (1, 144, 400),
        ),  # between /model.22/Concat_2 and /model.22/Concat_3
        (
            "/model.22/Reshape_1_output_0",
            (1, 144, 1600),
        ),  # between /model.22/Concat_1 and /model.22/Concat_3
        (
            "/model.22/Reshape_output_0",
            (1, 144, 6400),
        ),  # between /model.22/Concat and /model.22/Concat_3
    )

    input_to_shape = {  # type: ignore[assignment]
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size) for dimension_size in shape
        ]
        for tensor_name, shape in input_to_shape
    }
    output_to_shape = {  # type: ignore[assignment]
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size) for dimension_size in shape
        ]
        for tensor_name, shape in output_to_shape
    }

    extracted_model = Extractor(model).extract_model(
        input_names=list(input_to_shape), output_names=list(output_to_shape)
    )
    for value_info in extracted_model.graph.input:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(input_to_shape[value_info.name])
    for value_info in extracted_model.graph.output:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(output_to_shape[value_info.name])
    return extracted_model


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARN"))
    sys.exit(main())
