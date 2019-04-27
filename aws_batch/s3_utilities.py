import boto3


def get_bucketname_key(uripath):
    assert uripath.startswith("s3://")

    path_without_scheme = uripath[5:]
    bucket_end_index = path_without_scheme.find("/")

    bucket_name = path_without_scheme
    key = "/"
    if bucket_end_index > -1:
        bucket_name = path_without_scheme[0:bucket_end_index]
        key = path_without_scheme[bucket_end_index + 1:]

    return bucket_name, key


def list_files(s3path_prefix):
    assert s3path_prefix.startswith("s3://")
    assert s3path_prefix.endswith("/")

    bucket, key = get_bucketname_key(s3path_prefix)

    s3 = boto3.resource('s3')

    bucket = s3.Bucket(name=bucket)

    return ((o.bucket_name, o.key) for o in bucket.objects.filter(Prefix=key))
