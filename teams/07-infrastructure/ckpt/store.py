"""Where committed checkpoints live: a local folder (tests) or S3 (EC2) (ADR 0001 §2).

In simple words: the checkpoint code only talks to this small interface: put a file, get a file, read and
write a small JSON, list, delete. One version writes to a folder (the Docker test's shared volume), the other
to an S3 bucket. The S3 version is tested against an S3 emulator (moto) in the Docker cluster, so the same
code that will run on AWS has already run once.

    open_store("/s3/ckpt")                 -> LocalStore
    open_store("s3://bucket/prefix")       -> S3Store (endpoint from V5_S3_ENDPOINT, e.g. the emulator)

S3 specifics handled here:
- no rename and no append: JSON pointers are overwritten whole; commits.jsonl is read-modify-written by rank 0
  only, which is the one writer;
- every object carries its sha256 in metadata, and S3 is asked to store a SHA256 checksum it verifies on
  upload. verify(key) either re-reads the object and hashes it ("full") or compares the stored checksum
  ("head", cheap for TB-size checkpoints);
- multipart upload above 64 MB, retries with backoff (botocore standard mode handles 503 SlowDown).
"""
import base64
import hashlib
import json
import os
import shutil
import time


def _sha_file(path: str) -> tuple:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


class LocalStore:
    kind = "local"

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, key):
        return os.path.join(self.root, key)

    def put_file(self, src: str, key: str) -> tuple:
        """Copy through a temp name, so a reader never sees half a file. Returns (sha256, bytes)."""
        dst = self._p(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        h = hashlib.sha256()
        n = 0
        tmp = dst + ".part"
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            for chunk in iter(lambda: fi.read(1 << 20), b""):
                h.update(chunk)
                fo.write(chunk)
                n += len(chunk)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp, dst)
        return h.hexdigest(), n

    def get_file(self, key: str, dst: str) -> tuple:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(self._p(key), dst)
        return _sha_file(dst)

    def sha256(self, key: str, mode: str = "full") -> str:
        return _sha_file(self._p(key))[0] if self.exists(key) else None

    def exists(self, key: str) -> bool:
        return os.path.exists(self._p(key))

    def get_json(self, key: str):
        with open(self._p(key)) as f:
            return json.load(f)

    def put_json(self, key: str, obj) -> None:
        p = self._p(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def get_text(self, key: str) -> str:
        with open(self._p(key)) as f:
            return f.read()

    def append_line(self, key: str, line: str) -> None:
        p = self._p(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def list(self, prefix: str = "") -> list:
        """Every object under prefix: [{key, size, mtime}]."""
        base = self._p(prefix)
        out = []
        for dp, _, files in os.walk(base):
            for n in files:
                full = os.path.join(dp, n)
                out.append({"key": os.path.relpath(full, self.root), "size": os.path.getsize(full),
                            "mtime": os.path.getmtime(full)})
        return out

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._p(prefix), ignore_errors=True)

    def corrupt_for_test(self, key: str, offset: int = 100) -> None:
        with open(self._p(key), "r+b") as f:
            f.seek(offset)
            b = f.read(1)
            f.seek(offset)
            f.write(bytes([b[0] ^ 0xFF]))

    def local_path(self, key: str):
        """A real path DCP can read directly, or None (S3)."""
        return self._p(key)


class S3Store:
    kind = "s3"

    def __init__(self, url: str):
        import boto3
        from botocore.config import Config
        rest = url[len("s3://"):]
        self.bucket, _, self.prefix = rest.partition("/")
        self.prefix = self.prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("V5_S3_ENDPOINT") or None,
                               region_name=os.environ.get("AWS_REGION", "us-east-1"),
                               config=Config(retries={"max_attempts": 10, "mode": "standard"}))
        from boto3.s3.transfer import TransferConfig
        self.tx = TransferConfig(multipart_threshold=64 * 2 ** 20, multipart_chunksize=64 * 2 ** 20,
                                 max_concurrency=int(os.environ.get("V5_S3_CONCURRENCY", "16")))

    def _k(self, key):
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, src: str, key: str) -> tuple:
        sha, n = _sha_file(src)
        self.s3.upload_file(src, self.bucket, self._k(key), Config=self.tx,
                            ExtraArgs={"Metadata": {"sha256": sha}, "ChecksumAlgorithm": "SHA256"})
        return sha, n

    def get_file(self, key: str, dst: str) -> tuple:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        self.s3.download_file(self.bucket, self._k(key), dst, Config=self.tx)
        return _sha_file(dst)

    def sha256(self, key: str, mode: str = "full") -> str:
        """full: stream the object and hash it. head: trust the sha256 stored at upload (cheap)."""
        from botocore.exceptions import ClientError
        try:
            if mode == "head":
                return self.s3.head_object(Bucket=self.bucket, Key=self._k(key))["Metadata"].get("sha256")
            body = self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"]
            h = hashlib.sha256()
            for chunk in iter(lambda: body.read(1 << 20), b""):
                h.update(chunk)
            return h.hexdigest()
        except ClientError:
            return None

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError:
            return False

    def get_json(self, key: str):
        return json.loads(self.get_text(key))

    def put_json(self, key: str, obj) -> None:
        body = json.dumps(obj, indent=1, sort_keys=True).encode()
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=body,
                           ChecksumSHA256=base64.b64encode(hashlib.sha256(body).digest()).decode())

    def get_text(self, key: str) -> str:
        return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read().decode()

    def append_line(self, key: str, line: str) -> None:
        """S3 has no append: read, add, write. Only rank 0 of one job writes these files."""
        old = self.get_text(key) if self.exists(key) else ""
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=(old + line.rstrip("\n") + "\n").encode())

    def list(self, prefix: str = "") -> list:
        out = []
        pfx = self._k(prefix)
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=pfx):
            for o in page.get("Contents", []):
                key = o["Key"][len(self.prefix) + 1:] if self.prefix else o["Key"]
                out.append({"key": key, "size": o["Size"], "mtime": o["LastModified"].timestamp()})
        return out

    def delete_prefix(self, prefix: str) -> None:
        keys = [{"Key": self._k(o["key"])} for o in self.list(prefix.rstrip("/") + "/")]
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys[i:i + 1000], "Quiet": True})

    def corrupt_for_test(self, key: str, offset: int = 100) -> None:
        body = bytearray(self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read())
        body[offset] ^= 0xFF
        meta = self.s3.head_object(Bucket=self.bucket, Key=self._k(key))["Metadata"]
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=bytes(body), Metadata=meta)

    def local_path(self, key: str):
        return None


def open_store(url: str):
    return S3Store(url) if url.startswith("s3://") else LocalStore(url)


def join(*parts) -> str:
    return "/".join(p.strip("/") for p in parts if p)


def wait_for(fn, timeout_s: float, poll_s: float = 0.05):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        v = fn()
        if v:
            return v
        time.sleep(poll_s)
    return None
