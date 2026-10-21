# 私有配置

## 安全原则

Bohrium 凭据和 VASP POTCAR 都由用户在本机提供。不要把密码、Access Key、
ticket 或 POTCAR 内容粘贴到 Hermes 对话中，也不要提交到 Git。

运行 `phonon-agent setup` 后，默认配置位于 `~/.config/phonon-agent/`。
目录权限为 `0700`，含凭据的文件权限为 `0600`。可用
`PHONON_AGENT_CONFIG_DIR` 为测试或特殊部署选择其他目录。

## DeepMD-kit 与 phonon-kit 环境

DeepMD-kit 体积较大且与 CUDA 环境紧密相关，必须由用户先行安装。
`phonon-agent setup` 接受 Conda/venv 环境目录或其中的 Python 路径，例如：

```text
/home/moli/miniconda3/envs/dpa4
/home/moli/miniconda3/envs/dpa4/bin/python
```

两种输入都会规范化并保存为明确的 Python 路径：

```yaml
environments:
  deepmd:
    python: /home/moli/miniconda3/envs/dpa4/bin/python
tools:
  phonon_kit:
    command: /home/moli/.local/bin/ph
```

setup 会验证该 Python 能导入 DeepMD，但不会安装或升级 DeepMD。验证通过后，
它把 phonon-agent、phonon-kit、QUESTS 和 dpdata 等工具安装到同一环境，并在
`~/.local/bin/` 创建 `phonon-agent`、`ph` 全局命令。这样 phonon-kit worker、
MD 和 CUDA 始终使用同一套 Python 环境。

phonon-kit 包内只携带 DPA4 `.pt` checkpoint。首次使用 `builtin:dpa4` 前建议在
目标机器运行 `ph model freeze`，生成仅供本机使用的 `.pt2`；如果尚未冻结，
`builtin:dpa4` 会自动回退到包内 checkpoint。

运行 MD 时如果当前 `phonon-agent` 不属于已配置的 DeepMD 环境，工作流会停止
并要求使用 `~/.local/bin/phonon-agent`，不会混用另一个环境中的 DeepMD。

## Bohrium 认证

setup 支持三种输入方式：

1. 邮箱和密码，兼容现有 Bohrium/DPDispatcher 后端；
2. 保存于私有 machine.json 的 OpenAPI Access Key；
3. OpenAPI 环境变量，不把密钥写入文件。

环境变量方式需要在启动 Hermes 的同一环境中设置：

```bash
read -rsp "Bohrium Access Key: " BOHRIUM_ACCESS_KEY; echo
export BOHRIUM_ACCESS_KEY
read -rp "Bohrium Project ID: " BOHRIUM_PROJECT_ID
export BOHRIUM_PROJECT_ID
hermes
```

可选 App Key 使用 `BOHRIUM_APP_KEY`。`phonon-agent setup --check` 只报告
配置状态和权限，不显示秘密。旧配置不会自动迁移或复用。

## POTCAR ZIP 或目录

ZIP 示例：

```yaml
potcar:
  source: /private/potentials/potpaw54.zip
  family: potpaw54/potpaw_PBE/PAW_GGA_PBE
```

目录示例：

```yaml
potcar:
  source: /private/potentials/potpaw_PBE
  family: PAW_GGA_PBE
```

两种形式最终都应能定位到 `<family>/<variant>/POTCAR`。如果 source 直接包含
`Ag_pv/POTCAR` 等元素变体目录，family 可以为空。setup 会在能够唯一识别时
自动选择 family；存在多个候选时让用户选择。

每个项目仍需在 `workflow.yaml` 明确指定元素映射：

```yaml
dft:
  potcar:
    variants:
      Ag: Ag_pv
      I: I
```

软件只引用用户提供的来源，不下载或分发 POTCAR。准备 VASP 任务时会为实际
计算组装 POTCAR，但整个运行目录默认被 Git 忽略。

## 16 核默认值和项目覆盖

默认机型为 `c16_m64_cpu`，命令为 `mpirun -n 16 vasp_std`。提交前会同时
比较 machine.json 中的机型核数、配置声明核数和 MPI `-n`；不一致时拒绝提交。

全局配置提供默认值。项目可以显式覆盖 `dft.potcar.source/family` 或
`dft.dispatcher.machine/resources/cores/command`。项目值优先，但覆盖机型时
仍必须保持核数和命令一致。

云端作业名由工作流按 `pa-{project}-{run-id}-vasp` 生成，并在同一运行恢复时
保持不变。私有 machine.json 不会因作业命名而被改写。

## 缺少配置

只有使用相应功能时才要求私有配置：

- `phonon-check`、`md`、`entropy` 和 `selection` 不要求 POTCAR/Bohrium；
- `dft-prepare` 要求 POTCAR 来源和显式元素映射；
- `dft-submit` 还要求 Bohrium machine/resources 和认证信息。

缺少配置时运行 `phonon-agent setup`，完成后重新执行原任务即可。
