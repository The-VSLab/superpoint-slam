import os
import numpy as np
import cv2
import torch

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as exc:
    trt = None
    cuda = None
    TRT_IMPORT_ERROR = exc
else:
    TRT_IMPORT_ERROR = None


def build_engine_from_onnx(onnx_path, engine_path, fp16=True, max_workspace_size=1 << 30):
    if trt is None:
        raise RuntimeError(f"TensorRT is not available: {TRT_IMPORT_ERROR}")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    config.max_workspace_size = int(max_workspace_size)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 1, 240, 320), (1, 1, 480, 640), (1, 1, 720, 1280))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    return engine_path


class SuperPointTRTFrontend:
    def __init__(
        self,
        engine_path,
        nms_dist=4,
        conf_thresh=0.015,
        nn_thresh=0.7,
        use_fp16=True,
    ):
        if trt is None:
            raise RuntimeError(f"TensorRT is not available: {TRT_IMPORT_ERROR}")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8
        self.border_remove = 4
        self.use_fp16 = bool(use_fp16)

        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        self.context = self.engine.create_execution_context()

        self.input_index = 0
        self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def _allocate_buffers(self, input_shape):
        if not self.context.set_binding_shape(self.input_index, input_shape):
            raise RuntimeError("Failed to set input shape for TensorRT context")

        bindings = [None] * self.engine.num_bindings
        host_inputs = []
        device_inputs = []
        host_outputs = []
        device_outputs = []

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            size = int(trt.volume(shape))

            if self.engine.binding_is_input(i):
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                host_inputs.append(host_mem)
                device_inputs.append(device_mem)
                bindings[i] = int(device_mem)
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                host_outputs.append((host_mem, shape, dtype))
                device_outputs.append(device_mem)
                bindings[i] = int(device_mem)

        return bindings, host_inputs, device_inputs, host_outputs, device_outputs

    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W)).astype(int)
        inds = np.zeros((H, W)).astype(int)
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        assert img.ndim == 2, "image must be grayscale"
        assert img.dtype == np.float32, "image must be float32"
        H, W = img.shape

        input_shape = (1, 1, H, W)
        bindings, host_inputs, device_inputs, host_outputs, device_outputs = self._allocate_buffers(input_shape)
        stream = cuda.Stream()

        inp = img.reshape(1, 1, H, W)
        if self.use_fp16:
            inp = inp.astype(np.float16)
        host_inputs[0][:] = inp.ravel()

        cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        outputs = []
        for (host_mem, shape, dtype), device_mem in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
            outputs.append((host_mem, shape, dtype))
        stream.synchronize()

        if len(outputs) < 2:
            raise RuntimeError("TensorRT outputs are incomplete")

        semi = outputs[0][0].reshape(outputs[0][1])
        coarse_desc = outputs[1][0].reshape(outputs[1][1])
        if semi.dtype != np.float32:
            semi = semi.astype(np.float32)
        if coarse_desc.dtype != np.float32:
            coarse_desc = coarse_desc.astype(np.float32)

        semi = semi.squeeze()
        semi = semi - np.max(semi, axis=0, keepdims=True)
        dense = np.exp(semi)
        dense = dense / (np.sum(dense, axis=0, keepdims=True) + 1e-5)
        nodust = dense[:-1, :, :]
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]
        # 특징점 수 제한 (상위 N개만 유지)
        max_kpts = 600
        if pts.shape[1] > max_kpts:
            pts = pts[:, :max_kpts]
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            samp_pts = pts[:2, :].copy()
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.0)) - 1.0
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.0)) - 1.0
            samp_pts = samp_pts.transpose(1, 0).astype(np.float32)
            samp_pts = samp_pts.reshape(1, 1, -1, 2)
            samp_pts = torch.from_numpy(samp_pts)
            with torch.no_grad():
                desc = torch.nn.functional.grid_sample(torch.from_numpy(coarse_desc), samp_pts)
                desc = desc.numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap
