### 下载权重文件
见[bytetrack官方仓库](https://github.com/ifzhang/ByteTrack)

### pt转trt
python tools/trt.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar

### 推理
python tools/track_3dpose.py video --path test1.mp4 -f exps/example/mot/yolox_s_mix_det.py --trt --save_result --conf 0.6 --nms 0.6
