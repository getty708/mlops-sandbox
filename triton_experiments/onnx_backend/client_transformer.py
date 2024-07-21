import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import tritonclient.grpc as grpcclient
from loguru import logger
from services.toy_models.simple_transformer.const import (
    SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    SIMPLE_TRANSFORMER_INPUT_TENSOR,
    SIMPLE_TRANSFORMER_OUTPUT_TENSOR,
    SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
)
from tqdm import tqdm

from triton_experiments.onnx_backend.utils.client import (
    cleanup_shared_memory,
    convert_results_on_cudashm_to_tensor_dict,
    create_input_output_tensors_with_cudashm,
    create_input_tensors_wo_shared_memory,
    create_logdir,
    print_output_tensor_metas,
    save_client_stats,
    save_triton_inference_stats,
)
from triton_experiments.onnx_backend.utils.const import (
    PIPELINE_ARCH_ENSEMBLE,
    PIPELINE_ARCH_MONOLITHIC,
)

_REPO_ROOT = Path(__file__).parents[2]
SAMPLE_IMAGE_PATH = _REPO_ROOT / "data" / "pexels-anna-tarazevich-14751175-fullhd.jpg"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.remove()
logger.add(sys.stderr, level="INFO")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triton client for Simple CNN")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=SAMPLE_IMAGE_PATH,
        help="Path to input image",
    )
    parser.add_argument(
        "--pipeline-architecture",
        type=str,
        default=PIPELINE_ARCH_MONOLITHIC,
        help="architecture of the pipeline. [monolithic, ensemble]",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="use ORT-TRT optimization.",
    )
    parser.add_argument(
        "-s",
        "--pipeline-step-size",
        type=int,
        default=1,
        help="the number of models in a pipeline.",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        help="number of request send to the server.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size of the single request. Only 1 is supported for now.",
    )
    parser.add_argument(
        "--use-shared-memory",
        action="store_true",
        help="Use shared memory or not.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("./outputs/"),
        help="Path to a log directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose mode.",
    )
    return parser


def prepare_input_tensor(
    image_batch: torch.Tensor,
    use_shared_memory: bool = True,
    client: grpcclient.InferenceServerClient | None = None,
):
    # == Make InferInput ==
    if use_shared_memory:
        if _DEVICE == "cpu":
            raise NotImplementedError
        client.unregister_system_shared_memory()
        client.unregister_cuda_shared_memory()
        input_tensors, output_tensors, shm_handles = create_input_output_tensors_with_cudashm(
            image_batch,
            client,
            input_tensor_name=SIMPLE_TRANSFORMER_INPUT_TENSOR,
            output_tensor_name=SIMPLE_TRANSFORMER_OUTPUT_TENSOR,
        )
    else:
        input_tensors = create_input_tensors_wo_shared_memory(image_batch)
        output_tensors = None
        shm_handles = None

    return input_tensors, output_tensors, shm_handles


def call_triton_model(
    pipeline_architecture: str,
    use_trt: bool,
    step_size: int,
    input_tensors: torch.Tensor | list[grpcclient.InferInput],
    output_tensors: list[grpcclient.InferRequestedOutput] | None = None,
    client: grpcclient.InferenceServerClient | None = None,
) -> tuple[dict[str, torch.Tensor] | list[grpcclient.InferRequestedOutput], float]:
    ts_start = time.time()
    if pipeline_architecture == PIPELINE_ARCH_ENSEMBLE:
        assert step_size == 4
        model_name = f"ensemble_transformer_{step_size}"
    elif pipeline_architecture == PIPELINE_ARCH_MONOLITHIC:
        assert step_size == 1
        model_name = "simple_transformer"
    else:
        raise ValueError(f"Invalid architecture: {pipeline_architecture}")
    if use_trt:
        model_name += "_trt"
    results = client.infer(
        model_name=model_name,
        inputs=input_tensors,
        outputs=output_tensors,
    )

    elapsed_time = time.time() - ts_start
    return results, elapsed_time


def main():
    args = make_parser().parse_args()
    logdir = create_logdir(args)

    # Prepare inputs
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Inference
    logger.info(f"Call Triton with pattern {args.pipeline_architecture}")
    df_stats = []
    for _ in tqdm(range(args.num_requests)):
        input_batch = (
            torch.randn(
                (
                    args.batch_size,
                    SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
                    SIMPLE_TRANSFORMER_EMBEDDING_DIM,
                )
            )
            .float()
            .to(_DEVICE)
        )
        input_tensors, output_tensors, shm_handles = prepare_input_tensor(
            input_batch, use_shared_memory=args.use_shared_memory, client=client
        )
        results, elapsed_time = call_triton_model(
            args.pipeline_architecture,
            args.trt,
            args.pipeline_step_size,
            input_tensors,
            output_tensors,
            client,
        )
        if args.verbose:
            output_tensors = convert_results_on_cudashm_to_tensor_dict(results, shm_handles[1])
            print_output_tensor_metas(output_tensors)
        cleanup_shared_memory(client, shm_handles)
        df_stats.append({"elapsed_time": elapsed_time * 1e3})
    df_stats = pd.DataFrame(df_stats)
    logger.info(
        "Inference Request Completed! (Average Elapsed Time: {elapsed_time:.3f}ms [std: {elapsed_time_std:.3f}])".format(
            elapsed_time=df_stats["elapsed_time"].mean(),
            elapsed_time_std=df_stats["elapsed_time"].std(),
        )
    )

    # Save stats
    save_client_stats(logdir, df_stats)
    save_triton_inference_stats(logdir, client)


if __name__ == "__main__":
    main()
