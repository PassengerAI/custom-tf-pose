
import tensorrt as trt
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
if __name__ == "__main__":
	model_file = '/home/nvidia/output_new.uff'
	try:
		ctypes.CDLL('/usr/src/tensorrt/samples/python/uff_ssd/build/libflattenconcat.so')
	except:
		print(
			"Error: {}\n{}\n{}".format(
				"Could not find {}".format(PATHS.get_flatten_concat_plugin_path()),
				"Make sure you have compiled FlattenConcat custom plugin layer",
				"For more details, check README.md"
			)
		)
		sys.exit(1)
	trt.init_libnvinfer_plugins(TRT_LOGGER, '')
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		parser.register_input("Input", (3, 300, 300))
		parser.register_output("MarkOutput_0")
		parser.parse(model_file, network)
		builder.max_batch_size = 1
		builder.max_workspace_size = 1 << 30
		engine = builder.build_cuda_engine(network)
	with open('/home/nvidia/ssd_new.engine', 'wb') as f:
		f.write(engine.serialize())
