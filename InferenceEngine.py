import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    
class InferenceEngine:
	def __init__(self, engine_path):
		ctypes.CDLL('/usr/src/tensorrt/samples/python/uff_ssd/build/libflattenconcat.so')
		trt.init_libnvinfer_plugins(TRT_LOGGER, '')
		with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
			self.engine = runtime.deserialize_cuda_engine(f.read())
		self.inputs, self.outputs, self.bindings, self.stream = self._create_buffers_stream()
		self.context = self.engine.create_execution_context()
		
	def _create_buffers_stream(self):
		inputs = []
		outputs = []
		bindings = []
		stream = cuda.Stream()
		binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}
		for binding in self.engine:
			size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
			dtype = binding_to_type[str(binding)]
			# Allocate host and device buffers
			host_mem = cuda.pagelocked_empty(size, dtype)
			device_mem = cuda.mem_alloc(host_mem.nbytes)
			# Append the device buffer to device bindings.
			bindings.append(int(device_mem))
			# Append to the appropriate list.
			if self.engine.binding_is_input(binding):
				inputs.append(HostDeviceMem(host_mem, device_mem))
			else:
				outputs.append(HostDeviceMem(host_mem, device_mem))
		return inputs, outputs, bindings, stream

	def infer(self, img):
		img_ravel = img.ravel()
		
		np.copyto(self.inputs[0].host, img_ravel)
		#[detection_out, keepCount_out] = common.do_inference(
		#	self.context, bindings=self.bindings, inputs=self.inputs,
		#	outputs=self.outputs, stream=self.stream)
		[cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
		self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
		[cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
		self.stream.synchronize()
		return [out.host for out in self.outputs]
        #return detection_out, keepCount_out
