import torchvision.models as models

from src import VERSION
from src.export import Export
from src.utils import load_weight

from network import shufflenetv2 as classification  # 分类
from network import segmentation_shufflenetplus_v2_x1_0_infer as segmentation  # 分割
from network import multi_task_shufflenetplus_v2_x1_0_infer as multitask  # 多任务

if __name__ == '__main__':
    # Prepare your model -----------------------------------------------------------------------------------------------

    # model = classification(num_classes=4)
    model = segmentation(mask_classes=4)  # 标注类别数+1
    # model = multitask(num_classes=2, mask_classes=4)

    load_weight(model, path=r'D:\心鉴\项目\2023\模型训练\tiny-convert-main\tiny-convert-main\models\koutu_base.pth')

    print(f'Repo Version: {VERSION}.')

    # Convert model ----------------------------------------------------------------------------------------------------
    args = {
        'model': model,
        'mode': 'onnx',
        'shape': (1, 3, 256, 320),  # NCHW
        'opset_version': 12,
        'output': './',
        'input_names': ['images'],
        'output_names': ['output0'],  # only classification or segmentation
        # 'output_names': ['output0', 'output1'],  # multitask mode
        'dynamic_axes': None,
        'is_simplify': True,
    }

    export = Export(**args)
    export.run()
