"""
Launch YOLO-Pose court keypoint detection training on AWS SageMaker.

Uploads the COCO keypoint dataset to S3, configures a PyTorch training job,
and submits it. The entry point handles COCO->YOLO conversion inside the
container, so you upload the raw COCO-format data.

Prerequisites:
    1. AWS CLI configured (`aws configure`)
    2. SageMaker execution role with S3 access
    3. Dataset at data/raw/Tennis_Court_Keypoint/ (or already on S3)
    4. pip install sagemaker boto3

Usage:
    # Basic launch
    python scripts/sagemaker/launch_court_keypoint.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole

    # Use data already on S3
    python scripts/sagemaker/launch_court_keypoint.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole \
        --s3_data s3://my-bucket/datasets/Tennis_Court_Keypoint

    # Larger model, native resolution, bigger instance
    python scripts/sagemaker/launch_court_keypoint.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole \
        --instance_type ml.g5.xlarge \
        --model yolo11l-pose.pt \
        --imgsz 1280 \
        --epochs 200 \
        --batch 8
"""

import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


def parse_args():
    parser = argparse.ArgumentParser(description="Launch YOLO-Pose training on SageMaker")

    # AWS / SageMaker
    parser.add_argument("--role", type=str, required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region", type=str, default=None, help="AWS region")
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge",
                        help="SageMaker instance type")
    parser.add_argument("--instance_count", type=int, default=1)
    parser.add_argument("--volume_size", type=int, default=50, help="EBS volume size in GB")
    parser.add_argument("--max_runtime", type=int, default=86400, help="Max training time in seconds")
    parser.add_argument("--spot", action="store_true", help="Use spot instances")

    # Data
    parser.add_argument("--s3_data", type=str, default=None,
                        help="S3 URI of COCO keypoint dataset")
    parser.add_argument("--local_data", type=str, default="data/raw/Tennis_Court_Keypoint",
                        help="Local dataset path to upload")
    parser.add_argument("--s3_bucket", type=str, default=None)
    parser.add_argument("--s3_prefix", type=str, default="tennis-analysis/datasets/Tennis_Court_Keypoint")

    # Model output
    parser.add_argument("--s3_output", type=str, default=None)

    # Training hyperparameters
    parser.add_argument("--model", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=8)

    # Job config
    parser.add_argument("--job_name", type=str, default=None)
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--tags", type=str, nargs="*")

    return parser.parse_args()


def upload_data_to_s3(local_path, bucket, prefix, session):
    """Upload local dataset to S3."""
    print(f"Uploading {local_path} -> s3://{bucket}/{prefix}")
    s3_uri = sagemaker.Session(boto_session=session).upload_data(
        path=local_path,
        bucket=bucket,
        key_prefix=prefix,
    )
    print(f"Upload complete: {s3_uri}")
    return s3_uri


def main():
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    bucket = args.s3_bucket or sm_session.default_bucket()

    # Upload data to S3 if needed
    if args.s3_data:
        s3_data_uri = args.s3_data
        print(f"Using existing S3 data: {s3_data_uri}")
    else:
        assert os.path.exists(args.local_data), f"Local data not found: {args.local_data}"
        s3_data_uri = upload_data_to_s3(args.local_data, bucket, args.s3_prefix, boto_session)

    s3_output = args.s3_output or f"s3://{bucket}/tennis-analysis/models/court_keypoint"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = args.model.replace(".pt", "").replace("-pose", "")
    job_name = args.job_name or f"yolo-court-{model_short}-{timestamp}"

    tags = []
    if args.tags:
        for tag in args.tags:
            k, v = tag.split("=", 1)
            tags.append({"Key": k, "Value": v})
    tags.append({"Key": "project", "Value": "tennis-analysis"})
    tags.append({"Key": "task", "Value": "court-keypoint"})

    checkpoint_s3 = f"s3://{bucket}/tennis-analysis/checkpoints/court_keypoint/" if args.spot else None

    print(f"\n{'='*60}")
    print(f"SageMaker Training Job: {job_name}")
    print(f"{'='*60}")
    print(f"Instance:    {args.instance_type} x {args.instance_count}")
    print(f"Spot:        {args.spot}")
    print(f"Data:        {s3_data_uri}")
    print(f"Output:      {s3_output}")
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch:       {args.batch}")
    print(f"Image size:  {args.imgsz}")
    print(f"Pose weight: {args.pose}")
    print(f"{'='*60}\n")

    estimator = PyTorch(
        entry_point="entry_court_keypoint.py",
        source_dir="scripts/sagemaker",
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        output_path=s3_output,
        volume_size=args.volume_size,
        max_run=args.max_runtime,
        use_spot_instances=args.spot,
        max_wait=args.max_runtime * 2 if args.spot else None,
        checkpoint_s3_uri=checkpoint_s3,
        sagemaker_session=sm_session,
        tags=tags,
        hyperparameters={
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "fliplr": args.fliplr,
            "mosaic": args.mosaic,
            "scale": args.scale,
            "pose": args.pose,
            "box": args.box,
            "cls": args.cls,
            "dfl": args.dfl,
            "workers": args.workers,
        },
    )

    estimator.fit(
        inputs={"training": s3_data_uri},
        job_name=job_name,
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if args.wait:
        print(f"\nTraining complete!")
        print(f"Model artifacts: {estimator.model_data}")
    else:
        print(f"\nTraining job submitted: {job_name}")
        print(f"Monitor at: https://{boto_session.region_name}.console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    return estimator


if __name__ == "__main__":
    main()
