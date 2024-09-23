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
