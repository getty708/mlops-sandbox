import sys
import time
from pathlib import Path

import click
import pandas as pd
import torch
import tritonclient.grpc as grpcclient
from loguru import logger
from services.toy_models.simple_cnn.const import SIMPLE_CNN_INPUT_TENSOR
from tqdm import tqdm

from triton_experiments.common.const import SAMPLE_IMAGE_PATH
from triton_experiments.common.triton_client.utils import (
    cleanup_shared_memory,
    prepare_image_batch_tensor,
    save_client_stats,
    save_triton_inference_stats,
)
from triton_experiments.concurrent_model_exec.client_utils import (
    create_input_output_tensors_with_cudashm,
)

OUTPUT_DIR = Path("./outputs/")

logger.remove()
logger.add(sys.stderr, level="INFO")


def prepare_input_tensor(
    model_name: str,
    image_batch: torch.Tensor,
    client: grpcclient.InferenceServerClient | None = None,
):
    # == Make InferInput ==
    if "single" in model_name:
        output_tensor_names = ["output_image"]
    elif "ensemble_" in model_name:
        output_tensor_names = ["output_image_m1", "output_image_m2", "output_image_m3"]
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    client.unregister_system_shared_memory()
    client.unregister_cuda_shared_memory()
    input_tensors, output_tensors, shm_handles = create_input_output_tensors_with_cudashm(
        image_batch,
        client,
        input_tensor_name=SIMPLE_CNN_INPUT_TENSOR,
        output_tensor_names=output_tensor_names,
    )
    return input_tensors, output_tensors, shm_handles


def call_triton_model(
    pipeline_name: str,
    input_tensors: list[grpcclient.InferInput],
    output_tensors: list[grpcclient.InferRequestedOutput] | None = None,
    client: grpcclient.InferenceServerClient | None = None,
) -> tuple[list[grpcclient.InferRequestedOutput], float]:
    ts_start = time.time()
    results = client.infer(
        model_name=pipeline_name,
        inputs=input_tensors,
        outputs=output_tensors,
    )
    elapsed_time = time.time() - ts_start
    return results, elapsed_time


@click.command()
@click.option("-i", "--input-image", type=Path, default=SAMPLE_IMAGE_PATH)
@click.option("-p", "--pipeline-name", type=str, default="ensemble_single_onnx")
@click.option("-b", "--batch-size", type=int, default=2)
@click.option("-n", "--num-requests", type=int, default=5)
@click.option("--logdir", type=Path, default=OUTPUT_DIR)
def main(
    input_image: Path = SAMPLE_IMAGE_PATH,
    pipeline_name: str = "ensemble_sequential",
    batch_size: int = 2,
    num_requests: int = 5,
    logdir: Path = OUTPUT_DIR,
):
    logdir.mkdir(parents=True, exist_ok=True)

    # Prepare inputs
    image_batch = prepare_image_batch_tensor(input_image, batch_size)
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Inference
    logger.info(f"Call {pipeline_name}")
    df_stats = []
    for _ in tqdm(range(num_requests)):
        input_tensors, output_tensors, shm_handles = prepare_input_tensor(
            pipeline_name, image_batch, client=client
        )
        _, elapsed_time = call_triton_model(
            pipeline_name,
            input_tensors,
            output_tensors,
            client,
        )
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
