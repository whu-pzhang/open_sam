# Open SAM


## SAM 分割精度测试

直接采用数据集真值生成 prompt，然后利用SAM完成分割，计算精度指标

### Whu-building 

val set performance


| Model     | Prompt Type | background | building | mIoU      |
| --------- | ----------- | ---------- | -------- | --------- |
| SAM-Tiny  | boxes       | 97.97      | 85.09    | 91.53     |
| SAM-Base  | boxes       | 98.35      | 87.31    | 92.83     |
| SAM-Large | boxes       | 98.63      | 89.4     | **94.02** |
| SAM-Huge  | boxes       | 98.59      | 89.14    | 93.87     |




### LoveDA

| Model     | Prompt Type | background | building | road  | water | barren | forest | agricultural | mIoU      |
| --------- | ----------- | ---------- | -------- | ----- | ----- | ------ | ------ | ------------ | --------- |
| SAM-Tiny  | boxes       | 61.54      | 68.24    | 41.24 | 76.19 | 64.45  | 55.96  | 73.2         | 62.97     |
| SAM-Base  | boxes       | 58.08      | 69.62    | 30.12 | 78.91 | 67.61  | 58.94  | 77.11        | 62.91     |
| SAM-Large | boxes       | 65.12      | 70.74    | 41.03 | 78.28 | 66.82  | 59.27  | 78.93        | **65.74** |
| SAM-Huge  | boxes       | 60.11      | 71.96    | 33.00 | 79.66 | 69.93  | 60.20  | 80.67        | 65.08     |



### Potsdam

with clutter

| Model     | Prompt Type | impre. surf | building | low_veg | tree  | car   | clutter | mIoU      |
| --------- | ----------- | ----------- | -------- | ------- | ----- | ----- | ------- | --------- |
| SAM-Tiny  | boxes       | 71.66       | 75.13    | 77.51   | 66.70 | 95.54 | 71.30   | 76.31     |
| SAM-Base  | boxes       | 73.24       | 77.20    | 79.77   | 65.65 | 94.85 | 73.14   | 77.31     |
| SAM-Large | boxes       | 78.56       | 86.75    | 81.85   | 67.87 | 95.66 | 75.67   | **81.00** |
| SAM-Huge  | boxes       | 77.29       | 83.57    | 81.12   | 67.65 | 96.07 | 75.33   | 80.17     |



## Acknowledgement

- [SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [HQ-SAM](https://github.com/SysCV/sam-hq/tree/main)
