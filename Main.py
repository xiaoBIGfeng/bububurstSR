    model.load_state_dict(checkpoint["state_dict"])
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1667, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for Burstormer:
        size mismatch for conv1.0.weight: copying a param with shape torch.Size([48, 3, 3, 3]) from checkpoint, the shape in current model is torch.Size([48, 4, 3, 3]).
        size mismatch for up3.0.weight: copying a param with shape torch.Size([48, 48, 1, 1]) from checkpoint, the shape in current model is torch.Size([192, 48, 1, 1]).


Traceback (most recent call last):
  File "/mnt/diskb/penglong/dx/code/SR/zff_track2_eva.py", line 14, in <module>
    from Network_Real_burstSR import Base_Model
  File "/mnt/diskb/penglong/dx/code/SR/Network_Real_burstSR.py", line 9, in <module>
    import pytorch_lightning as pl
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/__init__.py", line 20, in <module>
    from pytorch_lightning.callbacks import Callback  # noqa: E402
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/callbacks/__init__.py", line 26, in <module>
    from pytorch_lightning.callbacks.pruning import ModelPruning
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/callbacks/pruning.py", line 31, in <module>
    from pytorch_lightning.core.lightning import LightningModule
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/core/__init__.py", line 16, in <module>
    from pytorch_lightning.core.lightning import LightningModule
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/core/lightning.py", line 39, in <module>
    from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/trainer/__init__.py", line 16, in <module>
    from pytorch_lightning.trainer.trainer import Trainer
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 30, in <module>
    from pytorch_lightning.accelerators import Accelerator, IPUAccelerator
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/accelerators/__init__.py", line 13, in <module>
    from pytorch_lightning.accelerators.accelerator import Accelerator  # noqa: F401
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/accelerators/accelerator.py", line 26, in <module>
    from pytorch_lightning.plugins.precision import ApexMixedPrecisionPlugin, NativeMixedPrecisionPlugin, PrecisionPlugin
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/plugins/__init__.py", line 8, in <module>
    from pytorch_lightning.plugins.plugins_registry import (  # noqa: F401
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/plugins/plugins_registry.py", line 20, in <module>
    from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/__init__.py", line 13, in <module>
    from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin  # noqa: F401
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/tpu_spawn.py", line 27, in <module>
    from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/loggers/__init__.py", line 18, in <module>
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/pytorch_lightning/loggers/tensorboard.py", line 26, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/utils/tensorboard/__init__.py", line 12, in <module>
    from .writer import FileWriter, SummaryWriter  # noqa: F401
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/torch/utils/tensorboard/writer.py", line 9, in <module>
    from tensorboard.compat.proto.event_pb2 import SessionLog
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/proto/event_pb2.py", line 17, in <module>
    from tensorboard.compat.proto import summary_pb2 as tensorboard_dot_compat_dot_proto_dot_summary__pb2
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/proto/summary_pb2.py", line 17, in <module>
    from tensorboard.compat.proto import tensor_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__pb2
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/proto/tensor_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import resource_handle_pb2 as tensorboard_dot_compat_dot_proto_dot_resource__handle__pb2
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/proto/resource_handle_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb2
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/proto/tensor_shape_pb2.py", line 36, in <module>
    _descriptor.FieldDescriptor(
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/google/protobuf/descriptor.py", line 621, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates



Exception in thread Thread-4:                                                                                                                                      
Traceback (most recent call last):
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/threading.py", line 980, in _bootstrap_inner
    self.run()
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/summary/writer/event_file_writer.py", line 233, in run
    self._record_writer.write(data)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/summary/writer/record_writer.py", line 40, in write
    self._writer.write(header + header_crc + data + footer_crc)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/tensorflow_stub/io/gfile.py", line 766, in write
    self.fs.append(self.filename, file_content, self.binary_mode)
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/tensorflow_stub/io/gfile.py", line 160, in append
    self._write(filename, file_content, "ab" if binary_mode else "a")
  File "/mnt/diskb/penglong/anaconda3/envs/dx9826/lib/python3.9/site-packages/tensorboard/compat/tensorflow_stub/io/gfile.py", line 166, in _write
    f.write(compatify(file_content))
OSError: [Errno 28] No space left on device







  File "/mnt/diskb/penglong/dx/code/SR/utils/metrics.py", line 151, in make_patches
    output1 = output.unfold(2,patch_size*8,stride*8).unfold(3,patch_size*8,stride*8).contiguous()
RuntimeError: maximum size for tensor at dimension 2 is 80 but size is 384


