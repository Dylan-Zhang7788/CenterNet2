from train_net import default_argument_parser, do_test, setup, build_model
import os
import torch
import time
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from fvcore.common.timer import Timer

def main(args):
    cfg = setup(args) # 前头定义的函数
    # MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    # MODEL.DEVICE = "cuda"

    # 这里调用了 META_ARCH_REGISTRY.get()(cfg)
    # 并且使用了注册器，注册了"GeneralizedRCNN"
    # "GeneralizedRCNN"这个类被写在
    model = build_model(cfg)  # modeling.meta_arch.build.py
    # dir=cfg.OUTPUT_DIR + "/eval_result/"
    start_iter=0
    writers = (
        [
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),  # utils.event.py 把指标写进json
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
    )
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        storage.step()
    
        f_list = os.listdir("./output-MY-BiFPN")
        for idx,file in enumerate(f_list):
            if os.path.splitext(file)[1] != '.pth': f_list.remove(file)
        f_list.sort()
        # print f_list
        for i, file in enumerate(f_list):
            # os.path.splitext():分离文件名与扩展名
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join("./output-MY-BiFPN",file), resume=args.resume)
            if cfg.TEST.AUG.ENABLED:
                model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
            storage.step()
            do_test(cfg, model,Writer=storage,num=i) 
            for writer in writers:
                writer.write()
                

if __name__=='__main__':

    args = default_argument_parser() # engine.default.py
    args.add_argument('--manual_device', default='') # python 
    args = args.parse_args() #python 
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device  # 指定训练的显卡
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())   # 分布式训练的什么东西 不懂
    print("Command Line Args:", args)    # engine.launch 第一个是函数，后面全是参数

    launch(
        main,
        # 后面的这些参数全都在 default_argument_parser() 里头定义的 
        # 要改可以 args.xxx=xxx 就行了
        args.num_gpus,        
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )