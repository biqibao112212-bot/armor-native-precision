from pathlib import Path
import gc,json,time,threading
import numpy as np,torch
from .engines import Engine,CoarseTRTGraph,SharedPackedGraphEngine,SharedPackedFineTRTGraph
from .localizer import QueuedPackedGpuCropGlyphLocalizer
from .quad_graph import ChunkedQuadGraph
from .coarse_scale_nms import scaled_nms

class RuntimePipeline:
    """Native 1440x1080 RGB -> accepted optical-frame armor poses.

    Uses one CUDA device/stream at a time. Do not call from concurrent threads.
    Model inputs are RGB, K, and same-exposure camera-link-to-world wxyz only.
    Startup/warmup is explicit and excluded from steady-state inference timing.
    """
    def __init__(self,bundle,engines,native_library,workers=2):
        self.bundle=Path(bundle);self.engine_dir=Path(engines)
        self.contract=json.loads((self.bundle/"runtime-contract.json").read_text())
        if self.contract["schema"]!="armor-native.runtime-bundle/1":
            raise ValueError("Unsupported runtime bundle")
        self.fine=QueuedPackedGpuCropGlyphLocalizer(
            self.bundle/"glyph_native.pt",native_library=native_library,workers=workers)
        self.fine.model=SharedPackedFineTRTGraph({
            s:self.engine_dir/f"fine{s}.engine" for s in [128,192,256,384,512]})
        self.fine.fallback.model=ChunkedQuadGraph(self.fine.fallback.model,4)
        self.coarse=CoarseTRTGraph(self.engine_dir/"coarse.engine")
        self._lock=threading.Lock();self.warmed=False

    @torch.inference_mode()
    def warmup(self):
        if not self._lock.acquire(blocking=False):raise RuntimeError("Concurrent runtime use")
        try:
            for s in [128,192,256,384,512]:
                for b in range(1,max(1,262144//(s*s))+1):
                    self.fine.model(torch.zeros((b,3,s,s),device="cuda"))
                    torch.cuda.synchronize()
            for b in range(1,5):
                self.fine.fallback.model(torch.zeros((b,4,3,128,128),device="cuda"),
                                         torch.zeros((b,4,2),device="cuda"))
                torch.cuda.synchronize()
            self.coarse(np.zeros((1080,1440,3),np.uint8),include_geometry=True)
            torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
            self.warmed=True
            return {"torch_allocated_bytes":torch.cuda.memory_allocated(),
                    "torch_reserved_bytes":torch.cuda.memory_reserved(),
                    "cuda_free_total_bytes":torch.cuda.mem_get_info()}
        finally:self._lock.release()

    @torch.inference_mode()
    def __call__(self,image,K,camera_quaternion_wxyz):
        if not self.warmed:raise RuntimeError("Call warmup() before steady-state inference")
        if not self._lock.acquire(blocking=False):raise RuntimeError("Concurrent runtime use")
        try:
            start=time.perf_counter()
            if image.shape!=(1080,1440,3) or image.dtype!=np.uint8:
                raise ValueError("Expected native uint8 RGB (1080,1440,3)")
            K=np.asarray(K,np.float64);q=np.asarray(camera_quaternion_wxyz,np.float64)
            if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:
                raise ValueError("Invalid calibrated intrinsics")
            if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
                raise ValueError("Expected unit same-exposure camera quaternion wxyz")
            pp=scaled_nms(self.coarse(image,threshold=self.fine.proposal_threshold,include_geometry=True))
            coarse_end=time.perf_counter()
            self.fine.gpu_image=self.coarse.input[0,:,:1080]
            qq=self.fine(image,pp,K,q)
            torch.cuda.synchronize();end=time.perf_counter()
            return {"proposals":pp,"outputs":qq,
                    "runtime_ms":[(end-start)*1000,(coarse_end-start)*1000,(end-coarse_end)*1000]}
        finally:self._lock.release()

    def close(self):
        if self.fine.executor is not None:self.fine.executor.shutdown(wait=True)
