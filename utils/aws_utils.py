"""
AWS S3 utilities for caching pattern data
"""

import json
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, BUCKET_NAME


def get_s3_client():
    """Get configured S3 client"""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def save_cache_to_s3(results, filename):
    """Save pattern results to S3 cache"""
    s3 = get_s3_client()
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"cache/{filename}",
        Body=json.dumps(results, indent=4),
        ContentType="application/json"
    )
    return filename


def load_cache_from_s3(filename):
    """Load pattern results from S3 cache"""
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
    return json.loads(obj["Body"].read().decode("utf-8"))


def cache_exists_in_s3(filename):
    """Check if cache file exists in S3"""
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
        return True
    except Exception:
        return False
