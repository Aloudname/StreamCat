# StreamCat (stream ver.)
## 概览

LoLA_hsViT的边缘部署部分，用RGB摄像头代替hsi数据流输入。
`run.py`作程序入口，数据流式处理

### 用法
```
bash

# 按config/config.yaml配置运行(webcam + ONNX fallback)
python run_stream.py

# 自定义配置运行，见第3、4、5条：
python run_stream.py -c <自定义参数>

# 后端用Triton：
python run_stream.py -c < --backend / -b > triton < --triton-url / -url > localhost:8001

# 实时视频流做输入：
python run_stream.py -c < --source / -s > /path/to/video.mp4

# 帧间隔(stride)为8，限宽1920：
python run_stream.py -c < --stride / -stride > 8 < --max-width / -mw > 1920

```
### 工作流

```mermaid
graph TD
    A[配置文件
    config.yaml] -->|加载配置| B[核心逻辑层
    core.py]
    B -->|捕获数据| C[数据捕获模块
    capture.py]
    B -->|数据预处理| D[数据预处理模块
    preprocess.py]
    B -->|推理| E[推理模块
    inference.py]
    B -->|后处理| F[数据后处理模块
    postprocess.py]
    B -->|监控| G[监控模块
    monitor.py]
    B -->|显示| H[显示模块
    display.py]
    B -->|数据通信协议| I[数据类模块
    packets.py]
    A -->|初始化| J[运行脚本
    run.py]
    J -->|启动流程| B
```

## 工作结构
```
StreamCat/
├── config/                 # 参数配置
│   ├── __init__.py
│   ├── config.yaml         # 项目配置文件
│   └── loader.py
├── pipeline/               # 工作流类
│   ├── __init__.py
│   ├── core.py             # 封装总线
│   ├── capture.py          # 数据捕获模块
│   ├── preprocess.py       # 数据预处理模块
│   ├── interface.py        # 推理模块
│   ├── postprocess.py      # 数据后处理模块
│   ├── monitor.py          # 栈内外监视服务
│   ├── display.py          # 显示模块
│   └── packets.py          # 模块间通信协议(数据传输类)
├── server/                 # Triton Server目录
│   ├── common_mini         # 推理实例
│   │   ├   1  /
│   │   │   └── model.plan  # 推理引擎文件
│   │   └── config.pdtxt    # 引擎配置文件
│   └── ...
├── src/                    # .md插图资源
│   ├── xxx.png
│   └── ...
├── run.py                  # 用户层入口
├── monitor.py              # 资源监视器 (服务调用)
├── LOG.md
├── README.md
├── requirements.txt
└── ...
```

## bugs & to-do
### bugs
- 作调试用的RGB数据流需要`(3, H, W)` -> `(H, W)`型的推理引擎，需额外引入这样的测试模型。
### to-do
- 做延时剖析，确定瓶颈；
- 队列调优、线程绑定、零拷贝优化；
- 异常帧保护、断流重连、置信度告警；
- 容器封装、开机自启、日志与指标上报、模型热更新、回滚。
