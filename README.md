# Open SAM


## TODO

- [x] 支持mask2coco，数据全部采用coco格式
- [x] 支持SAM微调训练
- [x] 优化 point 取样方式，提高效率
- [ ] 修改训练为batch训练，而非单张图计算后汇总
- [ ] 支持 gradio 在线 demo
- [ ] 支持 onnx 导出和推理



## SAM 分割精度测试

直接采用数据集真值生成 prompt，然后利用SAM完成分割，计算精度指标

### Whu-building 

val set performance

prompt mode

- bbox: bbox generated from gt mask
- 1 pt: randomly selected from gt mask
- 2 pts:
- 3 pts: 


| Model     | Prompt Mode | building |
| --------- | ----------- | -------- |
| SAM-Tiny  | bbox        | 84.55    |
|           | 1 pt        | 60.93    |
|           | 2 pts       | 69.73    |
|           | 3 pts       | 71.24    |
| SAM-Base  | bbox        | 86.86    |
|           | 1 pt        |          |
|           | 2 pts       |          |
|           | 3 pts       |          |
| SAM-Large | bbox        | **89.1** |
| SAM-Huge  | bbox        | 88.97    |



### LoveDA

| Model     | Prompt Mode | background | building | road  | water | barren | forest | agricultural | mIoU      |
| --------- | ----------- | ---------- | -------- | ----- | ----- | ------ | ------ | ------------ | --------- |
| SAM-Tiny  | bbox        | 61.54      | 68.24    | 41.24 | 76.19 | 64.45  | 55.96  | 73.2         | 62.97     |
|           | 1 pt        | 37.23      | 40.11    | 29.6  | 49.5  | 35.14  | 28.67  | 26.14        | 35.2      |
|           | 2 pts       | 39.86      | 51.96    | 36.76 | 59.68 | 43.94  | 33.51  | 49.4         | 45.02     |
|           | 3 pts       |            |          |       |       |        |        |              |           |
| SAM-Base  | bbox        | 58.08      | 69.62    | 30.12 | 78.91 | 67.61  | 58.94  | 77.11        | 62.91     |
| SAM-Large | bbox        | 65.12      | 70.74    | 41.03 | 78.28 | 66.82  | 59.27  | 78.93        | **65.74** |
| SAM-Huge  | bbox        | 60.11      | 71.96    | 33.00 | 79.66 | 69.93  | 60.20  | 80.67        | 65.08     |



### Potsdam

with clutter

| Model     | Prompt Mode | impre. surf | building | low_veg | tree  | car   | clutter | mIoU      |
| --------- | ----------- | ----------- | -------- | ------- | ----- | ----- | ------- | --------- |
| SAM-Tiny  | bbox        | 71.66       | 75.13    | 77.51   | 66.70 | 95.54 | 71.30   | 76.31     |
| SAM-Base  | bbox        | 73.24       | 77.20    | 79.77   | 65.65 | 94.85 | 73.14   | 77.31     |
| SAM-Large | bbox        | 78.56       | 86.75    | 81.85   | 67.87 | 95.66 | 75.67   | **81.00** |
| SAM-Huge  | bbox        | 77.29       | 83.57    | 81.12   | 67.65 | 96.07 | 75.33   | 80.17     |


## Finetune SAM


| Model    | Prompt Mode | Style             | whu-building | loveda | potsdam | Epoch |
| -------- | ----------- | ----------------- | ------------ | ------ | ------- | ----- |
| SAM-Tiny | bbox        | Finetune(decoder) | 93.68        |        |         |       |
|          | 1 pt        |                   | 86.27        |        |         |       |


main 分支 `multimask_output=False`

| Model    | Finetune dataset | whu-building | loveda | potsdam | Epoch |
| -------- | ---------------- | ------------ | ------ | ------- | ----- |
| SAM-tiny | whu-building     | 93.91        | 58.26  | 68.94   | 12    |
| SAM-tiny | loveda           | 91.71        | 57.20  | 62.34   | 12    |
| SAM-tiny | potsdam          | 90.73        | 59.23  | 72.86   | 12    |


dev 分支 `multimask_output=True`

| Model    | Finetune dataset | whu-building | loveda | potsdam | Mean | Epoch |
| -------- | ---------------- | ------------ | ------ | ------- | ---- | ----- |
| SAM-tiny | whu-building     | 89.22        | 58.25  | 69.05   |      | 12    |
| SAM-tiny | loveda           | 85.08        | 64.1   | 74.34   |      | 12    |
| SAM-tiny | potsdam          | 82.34        | 62.42  | 77.91   |      | 12    |



## Acknowledgement

- [SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [HQ-SAM](https://github.com/SysCV/sam-hq/tree/main)
